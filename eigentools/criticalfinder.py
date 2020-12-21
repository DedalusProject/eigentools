import logging
import numpy as np
from mpi4py import MPI
import h5py
from scipy import interpolate, optimize
import matplotlib.pyplot as plt

from dedalus.tools.cache import CachedAttribute

logger = logging.getLogger(__name__.split('.')[-1])

class CriticalFinder:
    """finds critical parameters for eigenvalue problems.

    This class provides simple tools for finding the critical
    parameters for the linear (in)stability of a given flow. The parameter space must be 2D; typically this will be (k, Re), where
    k is a wavenumber and Re is some control parameter (e. g. Reynolds or Rayleigh). However, this is completely user controllable.

    Attributes:
    -----------
    eigenproblem: An eigentools eigenproblem object
    comm:       The MPI comm group to share jobs across
    nproc:      The size of comm
    rank:       The local processor's rank in comm
    parameter_grids:  NumPy mesh grids containing the input values of the EVP over which
                the criticalfinder will search for the critical value.
    evalue_grid: A NumPy array of complex values, containing the maximum growth rates
                of the EVP for the corresponding input values. 
    """
    
    def __init__(self, eigenproblem, param_names, comm, find_freq=False):
        self.eigenproblem = eigenproblem
        self.param_names = param_names
        self.comm = comm
        self.size = self.comm.size
        self.rank = self.comm.rank
        self.find_freq = find_freq

        self.roots = None

    def grid_generator(self, points, sparse=False):
        """
        Generates a 2-dimensional grid over the specified parameter
        space of an eigenvalue problem.  
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

            gr, indx, freq = self.growth_rate(values)
            local_grid[n] = gr + 1j*freq

        # Communicate growth modes to root
        data = np.empty(dims, dtype=np.complex128)
        rec_counts = np.array([s.size for s in load_indices])
        displacements = np.cumsum(rec_counts) - rec_counts
        self.comm.Gatherv(local_grid,[data,rec_counts,displacements, MPI.F_DOUBLE_COMPLEX])
        self.evalue_grid = data

    def growth_rate(self, values, **kwargs):
        var_dict = {self.param_names[i]: v for i,v in enumerate(values)}
        return self.eigenproblem.growth_rate(var_dict, **kwargs) #solve
        
    @CachedAttribute
    def _interpolator(self):
        """Creates and then uses a 2D grid interpolator for growth rate

        NB: this transposes x and y for the root finding step, because that requires the function to be interpolated along the FIRST axis
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

        Inputs:
        -------
            filename:   The name of the .h5 file containing the grid data
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

        Inputs:
        -------
            filen   -- A file stem, which DOES NOT specify the file type. The
                       grid will be saved to a file called filen.h5
        Example:
        --------
        file_name = 'my_grid'
        my_cf.save_grid(file_name) #creates a file called my_grid.h5
        """
        if self.comm.rank == 0:
            with h5py.File(filename+'.h5','w') as outfile:
                outfile.create_dataset('grid',data=self.evalue_grid)
                for i, grid in enumerate(self.parameter_grids):
                    outfile.create_dataset('xyz_{}'.format(i),data=grid)

    def _root_finder(self):
        yy = self.parameter_grids[1]
        xx = self.parameter_grids[0]
        self.roots = np.zeros_like(xx[0,:])
        for j,x in enumerate(xx[0,:]):
            try:
                self.roots[j] = optimize.brentq(self._interpolator,yy[0,0],yy[-1,0],args=(x))
            except ValueError:
                self.roots[j] = np.nan

    def crit_finder(self, polish_roots=False, polish_sparse=True, tol=1e-3, method='Powell', maxiter=200, **kwargs):
        """returns a tuple of the x value at which the minimum (critical value
        occurs), and the y value. 
        output
        ------
        (x_crit, y_crit) 
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

    def critical_polisher(self, guess, tol=1e-3, method='Powell', maxiter=200, sparse=True, **kwargs):
        """
        Polishes a guess for the critical value using scipy's
        optimization routines to find a more precise location of the critical value.

        Inputs:
        -------
            tol, method, maxiter -- All inputs to the scipy.optimize.minimize function
        """
        
        # minimize absolute value of growth rate
        function = lambda args: np.abs(self.growth_rate(args, sparse=sparse)[0])
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

    def plot_crit(self, title='growth_rates', transpose=False, xlabel = None, ylabel = None, zlabel="growth rate", cmap="viridis"):
        """Create a 2D colormap of the grid of growth rates.  If available, the
            root values that have been found will be plotted over the colormap

            Inputs:
            -------
            title       - The name of the plot, which will be saved out to "title".png
            transpose   - If True, plot dim 0 on the y axis and dim 1 on the x axis.
                          Otherwise, plot it the other way around.
            xlabel      - The x-label of the plot
            ylabel      - The y-label of the plot
        """
        if self.rank != 0:
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)
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
        plt.pcolormesh(xx,yy,grid,cmap=cmap,vmin=-biggest_val,vmax=biggest_val)
        plt.colorbar(label=zlabel)

        # Plot root data if they're available
        if self.roots is not None:
            if transpose:
                x = self.parameter_grids[1][0,:]
                y = self.roots[:]
            else:   
                x = self.roots[:]
                y = self.parameter_grids[1][0,:]

            if transpose:
                y, x = y[np.isfinite(y)], x[np.isfinite(y)]
            else:
                y, x = y[np.isfinite(x)], x[np.isfinite(x)]
            plt.scatter(x,y, color='k')
        
        # Pretty up the plot, save.
        plt.ylim(yy.min(),yy.max())
        plt.xlim(xx.min(),xx.max())
        if xlabel is None:
            xlabel = self.param_names[0]
        if ylabel is None:
            ylabel = self.param_names[1]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        fig.savefig('{}.png'.format(title))
        
        return fig
