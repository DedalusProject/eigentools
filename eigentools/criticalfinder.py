import logging
import numpy as np
from mpi4py import MPI
import h5py
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
from matplotlib import transforms

from dedalus.tools.cache import CachedAttribute

logger = logging.getLogger(__name__.split('.')[-1])

class CriticalFinder:
    """finds critical parameters for eigenvalue problems.

    This class provides simple tools for finding the critical parameters
    for the linear (in)stability of a given flow. The parameter space must
    be 2D; typically this will be (k, Re), where k is a wavenumber and Re
    is some control parameter (e. g. Reynolds or Rayleigh). The parameters
    are defined by the underlying Eigenproblem object.

    Parameters
    ----------
    eigenproblem: Eigenproblem
        An eigentools eigenproblem object over which to find critical
        parameters
    param_names : tuple of str
        The names of parameters to search over
    comm : mpi4py.MPI.Intracomm, optional
        The MPI comm group to share jobs across (default: MPI.COMM_WORLD)
    find_freq : bool, optional
        If True, also find frequency at critical point

    Attributes
    ----------
    parameter_grids:  
        NumPy mesh grids containing the parameter values for the EVP
    evalue_grid: 
        NumPy array of complex values, containing the maximum growth rates
        of the EVP for the corresponding input values.
    roots : ndarray
        Array of roots along axis 1 of parameter_grid
    """
    
    def __init__(self, eigenproblem, param_names, comm=MPI.COMM_WORLD, find_freq=False):
        self.eigenproblem = eigenproblem
        self.param_names = param_names
        self.comm = comm
        self.size = self.comm.size
        self.rank = self.comm.rank
        self.find_freq = find_freq

        self.roots = None

    def grid_generator(self, points, sparse=False):
        """Generates a grid of eigenvalues over the specified parameter
        space of an eigenvalue problem.

        Parameters
        ----------
        points : tuple of ndarray
            The parameter values over which to find the critical value
        """
        self.parameter_grids = np.meshgrid(*points)
        self.evalue_grid = np.zeros(self.parameter_grids[0].shape, dtype=np.complex128)
        dims = self.evalue_grid.shape
        # Split parameter load across processes
        index = np.arange(np.prod(dims))
        load_indices = np.array_split(index,self.size)
        my_indices = load_indices[self.rank]

        # Calculate growth values for local process grid
        local_grid = np.empty(my_indices.size,dtype=np.complex128)
        for n, index in enumerate(my_indices):
            logger.info("Solving Local EVP {}/{}".format(n+1, len(my_indices)))
            unraveled_index = np.unravel_index(index, dims)
            values = [self.parameter_grids[i][unraveled_index] for i,v in enumerate(self.parameter_grids)]

            gr, indx, freq = self._growth_rate(values, sparse=sparse)
            local_grid[n] = gr + 1j*freq

        # Communicate growth modes to root
        data = np.empty(dims, dtype=np.complex128)
        rec_counts = np.array([s.size for s in load_indices])
        displacements = np.cumsum(rec_counts) - rec_counts
        self.comm.Gatherv(local_grid,[data,rec_counts,displacements, MPI.F_DOUBLE_COMPLEX])
        self.evalue_grid = data

    def _growth_rate(self, values, **kwargs):
        """Compute growth rate at values

        Parameters
        ----------
        values : dict
            Dictionary of parameter names and values
        """
        var_dict = {self.param_names[i]: v for i,v in enumerate(values)}
        return self.eigenproblem.growth_rate(var_dict, **kwargs) #solve
        
    @CachedAttribute
    def _interpolator(self):
        """Creates and then uses a 2D grid interpolator for growth rate

        NB: this transposes x and y for the root finding step, because that
        requires the function to be interpolated along the FIRST axis
        """
        xx = self.parameter_grids[0]
        yy = self.parameter_grids[1]
        return interpolate.interp2d(yy.T, xx.T, self.evalue_grid.real.T)

    @CachedAttribute
    def _freq_interpolator(self):
        """Creates and then uses a 2D grid interpolator for growth rate
        """
        xx = self.parameter_grids[0]
        yy = self.parameter_grids[1]
        return interpolate.interp2d(xx, yy, self.evalue_grid.imag)
        
    def load_grid(self, filename):
        """
        Load a grid file, in the format as created in save_grid.

        Parameters
        ----------
        filename : str
            The name of the .h5 file containing the grid data
        """
        with h5py.File(filename,'r') as infile:
            self.parameter_grids = [k[()] for k in infile.values() if 'xyz' in k.name]
            self.N = len(self.parameter_grids)
            logger.info("Read an {}-dimensional grid".format(self.N))
            self.evalue_grid = infile['/grid'][:]
        
    def save_grid(self, filename):
        """
        Saves the grids of all input parameters as well as the growth rate
        grid that has been solved for.

        Parameters
        ----------
        filename : str
           A file stem, which DOES NOT include the file type extension. The
           grid will be saved to a file called filen.h5
        """
        if self.comm.rank == 0:
            with h5py.File(filename+'.h5','w') as outfile:
                outfile.create_dataset('grid',data=self.evalue_grid)
                for i, grid in enumerate(self.parameter_grids):
                    outfile.create_dataset('xyz_{}'.format(i),data=grid)

    def _root_finder(self):
        """Find rooots from interpolated values at each point along zero axis of parameter_grid

        """
        yy = self.parameter_grids[1]
        xx = self.parameter_grids[0]
        self.roots = np.zeros_like(xx[0,:])
        for j,x in enumerate(xx[0,:]):
            try:
                self.roots[j] = optimize.brentq(self._interpolator,yy[0,0],yy[-1,0],args=(x))
            except ValueError:
                self.roots[j] = np.nan

    def crit_finder(self, polish_roots=False, polish_sparse=True, tol=1e-3, method='Powell', maxiter=200, **kwargs):
        """returns parameters at which critical eigenvalue occurs and optionally frequency at that value. 

        The critical parameter is defined as the absolute minimum of the
        growth rate, defined in the Eigenproblem via its grow_func. If
        frequency is to be found also, returns the frequnecy defined in the
        Eigenproblem via its freq_func.

        If find_freq is True, returns (critical parameter 1, critical
        parameter 2, frequency); otherwise returns (critical parameter 1,
        critical parameter 2)

        Parameters
        ----------
        polish_roots : bool, optional
            If true, use optimization routines to polish critical value (default: False)
        polish_sparse : bool, optional
            If true, use the sparse solver when polishing roots (default: True)
        tol : float, optional
            Tolerance for polishing routine (default: 1e-3)
        method : str, optional
            Method for scipy.optimize used for polishing (default: Powell)
        maxiter : int, optional
            Maximum number of optimization iterations used for polishing (default: 200)

        Returns
        ------
        tuple
        """
        if self.rank != 0:
            return
        self._root_finder()
        mask = np.isfinite(self.roots)
        xx_root = self.parameter_grids[0][0,mask]
        rroot = self.roots[mask]

        self.root_fn = interpolate.interp1d(xx_root,rroot,kind='cubic')
        
        mid = xx_root.shape[0]//2
        
        bracket = [xx_root[0],xx_root[mid],xx_root[-1]]
        
        self.opt = optimize.minimize_scalar(self.root_fn,bracket=bracket)

        x_crit = self.opt['x']
        y_crit = self.opt['fun'].item()
        if self.find_freq:
            crit_freq = self._freq_interpolator(x_crit, y_crit)[0]
            crits = (x_crit, y_crit, crit_freq)
            if polish_roots:
                crits = self.critical_polisher(crits, sparse=polish_sparse,
                                               tol=tol, method=method, maxiter=maxiter, **kwargs)

            return crits
        
        crits = (x_crit, y_crit)
        if polish_roots:
            crits = self.critical_polisher(crits, sparse=polish_sparse,
                                           tol=tol, method=method, maxiter=maxiter, **kwargs)

        return crits

    def critical_polisher(self, guess, sparse=True, tol=1e-3, method='Powell', maxiter=200, **kwargs):
        """
        Polishes a guess for the critical value using scipy's optimization
        routines to find a more precise location of the critical value.

        Parameters
        ----------
        guess : complex
            Initial guess for optimization routines
        sparse : bool, optional
            If true, use the sparse solver when polishing roots (default: True)
        tol : float, optional
            Tolerance for polishing routine (default: 1e-3)
        method : str, optional
            Method for scipy.optimize used for polishing (default: Powell)
        maxiter : int, optional
            Maximum number of optimization iterations used for polishing
            (default: 200)
        """
        
        # minimize absolute value of growth rate
        function = lambda args: np.abs(self._growth_rate(args, sparse=sparse)[0])
        if self.find_freq:
            x0 = guess[:-1]
        else:
            x0 = guess
        search_result = optimize.minimize(function, x0, 
                                          tol=tol, options={'maxiter': maxiter}, method=method)

        logger.debug("Optimize results: {}".format(search_result))

        if self.find_freq:
            freq = self._freq_interpolator(search_result.x[0],search_result.x[1])
        
        if search_result.success:
            logger.info('Minimum growth rate of {} found'.format(search_result.fun))
            results = list(search_result.x)
            if self.find_freq:
                results += list(freq)
            return results
        else:
            logger.warning('Optimize results not fully converged, returning crit_finder results.')
            return guess

    def plot_crit(self, axes=None, transpose=False, xlabel = None, ylabel = None, zlabel="growth rate", cmap="viridis"):
        """Create a 2D colormap of the grid of growth rates.  

        If available, the root values that have been found will be plotted
        over the colormap.

        Parameters
        ----------
        transpose : bool, optional
            If True, plot dim 0 on the y axis and dim 1 on the x axis.
        xlabel : str, optional
            If not None, the x-label of the plot. Otherwise, use parameter name from EVP
        ylabel : str, optional 
            If not None, the y-label of the plot. Otherwise, use parameter name from EVP
        zlabel : str, optional
            Label for the colorbar. (default: growth rate)
        cmp : str, optional
            matplotlib colormap name (default: viridis)
        """
        if self.rank != 0:
            return
        
        if axes is None:
            fig = plt.figure(figsize=[8,8])
            ax = fig.add_subplot(111)
        else:
            ax = axes
            fig = axes.figure

        # Grab out grid data for colormap
        if transpose:
            xx = self.parameter_grids[1].T
            yy = self.parameter_grids[0].T
            grid = self.evalue_grid.real.T
        else:
            xx = self.parameter_grids[0]
            yy = self.parameter_grids[1]
            grid = self.evalue_grid.real
        # Plot colormap, only plot 2 stdevs off zero
        biggest_val = 2*np.abs(grid).std()

        # Setup axes
        # Bounds (left, bottom, width, height) relative-to-axes
        pbbox = transforms.Bbox.from_bounds(0.03, 0, 0.94, 0.94)
        cbbox = transforms.Bbox.from_bounds(0.03, 0.95, 0.94, 0.05)
        # Convert to relative-to-figure
        to_axes_bbox = transforms.BboxTransformTo(ax.get_position())
        pbbox = pbbox.transformed(to_axes_bbox)
        cbbox = cbbox.transformed(to_axes_bbox)
        # Create new axes and suppress base axes
        pax = ax.figure.add_axes(pbbox)
        cax = ax.figure.add_axes(cbbox)

        plot = pax.pcolormesh(xx,yy,grid,cmap=cmap,vmin=-biggest_val,vmax=biggest_val)
        ax.axis('off')
        cbar = plt.colorbar(plot, cax=cax, label=zlabel, orientation='horizontal')
        cbar.outline.set_visible(False)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        # Plot root data if they're available
        if self.roots is not None:
            if transpose:
                x = self.roots[:]
                y = self.parameter_grids[0][0,:]
            else:
                x = self.parameter_grids[0][0,:]
                y = self.roots[:]

            if transpose:
                y, x = y[np.isfinite(x)], x[np.isfinite(x)]
            else:
                y, x = y[np.isfinite(y)], x[np.isfinite(y)]
            pax.scatter(x,y, color='k')
        
        # Pretty up the plot, save.
        pax.set_ylim(yy.min(),yy.max())
        pax.set_xlim(xx.min(),xx.max())
        if xlabel is None:
            xlabel = self.param_names[0]
        if ylabel is None:
            ylabel = self.param_names[1]
        pax.set_xlabel(xlabel)
        pax.set_ylabel(ylabel)
        
        return pax,cax
