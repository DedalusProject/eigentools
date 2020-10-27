import logging
import numpy as np
from mpi4py import MPI
import h5py
from scipy import interpolate, optimize
import matplotlib.pyplot as plt

from dedalus.tools.cache import CachedAttribute

logger = logging.getLogger(__name__.split('.')[-1])

def load_balance(dims, nproc):
    """
    Evenly splits up tasks across the specified number of processes.

    Inputs:
    -------
        dims    - An N-length array, containing the size of each dimension
                  of the eigenvalue problem.
        nproc   - The number of processes to split up the problem over
    
    Outputs:
    --------
        An array of arrays, where the 0th array contains the task indices for the
        0th process, and so on.

    Example:
    -------
        # Three dimensional critical-finding problem
        nx, ny, nz  = 5, 10, 20
        nproc       = MPI.COMM_WORLD.size
        dims        = np.array((nx, ny, nz))
        tasks       = load_balance(dims, nproc)
    """
    index = np.arange(np.prod(dims))
    return np.array_split(index,nproc)

def index2indices(index, dims):
    """
    Converts a single number index into its corresponding N-dimensional index in
    the N-dimensional space that the problem is searching in.

    Inputs:
    -------
        index   - An integer, containing the scalar index you want to convert to
                  a full index in N-dimensional space.
        dims    - A NumPy array, of N-elements, containing the size of each dimension.

    Example:
    --------
    # searching on a 5 x 10 x 20 grid, dims = (5, 10, 20)
    dims = np.array( (5, 10, 20) )
    # index is between 0 and dims.prod() (1000 in this example)
    index = 543
    indxs = index2indices(index, dims)
    print(indxs) #should return [2, 7, 3]
    """
    indices = []
    for i in range(len(dims)):
        if i < len(dims) - 1:
            indices.append(np.int(np.floor(index/np.prod(dims[i+1:]))))
            index -= indices[-1]*np.prod(dims[i+1:])
        else:
            indices.append(np.int(index % dims[i]))
            index -= indices[-1]
        if indices[i] >= dims[i]:
            raise Exception("Index {} too large for dimension {}".format(indices[i], dims[i]))

    if index != 0:
        raise Exception("Something went wrong converting index {} to indices".format(index))
    return indices

class CriticalFinder:
    """CriticalFinder

    This class provides simple tools for finding the critical
    parameters for the linear (in)stability of a given flow. Here, the
    parameter space must be 2D; typically this will be (k, Re), where
    k is a wavenumber and Re is some control parameter (Reynolds or
    Rayleigh or some such number). 

    Attributes:
    -----------
    func:       a function of two variables (x,y) that provides growth rates
                at each (x, y). CriticalFinder will find the roots of the growth
                rate in x for each value of y.
    comm:       The MPI comm group to share jobs across
    nproc:      The size of comm
    rank:       The local processor's rank in comm
    logs:       A list (or NumPy array) of boolean values, each of which corresponds
                to an input dimension over which the eigenvalue problem is being
                solved.  "True" dimensions are in log space
    N:          The dimensionality of the problem / grid space
    xyz_grids:  NumPy mesh grids containing the input values of the EVP over which
                the criticalfinder will search for the critical value.
    grid:       A NumPy array of complex values, containing the maximum growth rates
                of the EVP for the corresponding input values.

    """
    
    def __init__(self, eigenproblem, param_names, comm):
        self.eigenproblem = eigenproblem
        self.param_names = param_names
        self.comm = comm
        self.size = self.comm.size
        self.rank = self.comm.rank

    def grid_generator(self, points):
        """
        Generates a 2-dimensional grid over the specified parameter
        space of an eigenvalue problem.  
        """
        self.parameter_grids = np.meshgrid(*points, indexing='ij')
        self.evalue_grid = np.zeros(self.parameter_grids[0].shape, dtype=np.complex128)
        self.N = self.evalue_grid.ndim
        dims = self.evalue_grid.shape
        # Split parameter load across processes
        load_indices = load_balance(dims, self.size)
        my_indices = load_indices[self.rank]

        # Calculate growth values for local process grid
        local_grid = np.empty(my_indices.size,dtype=np.complex128)
        for n, index in enumerate(my_indices):
            logger.info("Solving Local EVP {}/{}".format(n+1, len(my_indices)))
            unraveled_index = np.unravel_index(index, dims)
            values = [self.parameter_grids[i][unraveled_index] for i,v in enumerate(self.parameter_grids)]
            var_dict = {self.param_names[i]: values[i] for i,v in enumerate(values)}
            gr, indx, freq = self.eigenproblem.growth_rate(var_dict) #solve
            local_grid[n] = gr + 1j*freq

        # Communicate growth modes to root
        data = np.empty(dims, dtype=np.complex128)
        rec_counts = np.array([s.size for s in load_indices])
        displacements = np.cumsum(rec_counts) - rec_counts
        self.comm.Gatherv(local_grid,[data,rec_counts,displacements, MPI.F_DOUBLE_COMPLEX])
        self.evalue_grid = data
        
    @CachedAttribute
    def _interpolator(self):
        """Creates and then uses a 2D grid interpolator
        """
        xx = self.parameter_grids[0]
        yy = self.parameter_grids[1]
        return interpolate.interp2d(xx, yy, self.evalue_grid.real)

    def load_grid(self, filename):
        """
        Load a grid file, in the format as created in save_grid.

        Inputs:
        -------
            filename:   The name of the .h5 file containing the grid data
        """
        with h5py.File(filename,'r') as infile:
            self.parameter_grids = [k.value for k in infile.values() if 'xyz' in k.name]
            self.N = len(self.parameter_grids)
            logger.info("Read an {}-dimensional grid".format(self.N))
            self.evalue_grid = infile['/grid'][:]
        
    def save_grid(self, filen):
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
            with h5py.File(filen+'.h5','w') as outfile:
                outfile.create_dataset('grid',data=self.evalue_grid)
                for i, grid in enumerate(self.parameter_grids):
                    outfile.create_dataset('xyz_{}'.format(i),data=grid)

    def _root_finder(self):
        yy = self.parameter_grids[1]
        xx = self.parameter_grids[0]
        self.roots = np.zeros_like(yy[0,:])
        for j,y in enumerate(yy[0,:]):
            try:
                self.roots[j] = optimize.brentq(self._interpolator,xx[0,0],xx[-1,0],args=(y))
            except ValueError:
                self.roots[j] = np.nan

    def crit_finder(self, find_freq=False):
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
        yy_root = self.parameter_grids[1][0,mask]
        rroot = self.roots[mask]
        self.root_fn = interpolate.interp1d(yy_root,rroot,kind='cubic')
        
        mid = yy_root.shape[0]//2
        
        bracket = [yy_root[0],yy_root[mid],yy_root[-1]]
        
        self.opt = optimize.minimize_scalar(self.root_fn,bracket=bracket)

        x_crit = self.opt['x']
        y_crit = np.asscalar(self.opt['fun'])

        if find_freq:
            freq_interp = interpolate.interp2d(self.parameter_grids[0],self.parameter_grids[1],self.evalue_grid.imag)
            crit_freq = freq_interp(x_crit, y_crit)[0]
            return (x_crit, y_crit, crit_freq)

        return (x_crit, y_crit)

    def critical_polisher(self, tol=1e-3, method='Powell', maxiter=200, **kwargs):
        """
        Polishes the critical value.  Runs the self.crit_finder function
        to get a good initial guess for where the crit is, then uses scipy's
        optimization routines to find a more precise location of the critical value.

        Inputs:
        -------
            tol, method, maxiter -- All inputs to the scipy.optimize.minimize function
        """
        if self.rank != 0:
            return [None]*len(self.parameter_grids)
        crits = self.crit_finder(method=method, **kwargs)
        if np.isnan(crits[0]):
            logger.warning("crit_finder returned NaN, cannot find exact crit")
            return crits

        # Create a lambda function that wraps the object's function, and returns
        # the absolute value of the growth rate.  Minimize that function.
        function = lambda *args: np.abs(self.func(*tuple([i*x for i,x in zip(args[0], crits)])).real)
        search_result = optimize.minimize(function, [1.0]*len(self.parameter_grids), 
                                          tol=tol, options={'maxiter': maxiter})

        logger.info("Optimize results: {}".format(search_result))
        logger.info("Best values found by optimize: {}".format([np.asscalar(x*c) for x,c in zip(search_result.x, crits)]))

        if search_result.success:
            logger.info('Minimum growth rate of {} found'.format(search_result.fun))
            return [np.asscalar(x*c) for x,c in zip(search_result.x, crits)]
        else:
            logger.warning('Optimize results not fully converged, returning crit_finder results.')
            return crits

    def plot_crit(self, title='growth_rates', transpose=False, xlabel = "", ylabel = ""):
        """Create a 2D colormap of the grid of growth rates.  If in 3D, create one
            of these grids for each value in the 3rd dimension.  If available, the
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
        if self.N <= 3: 
            if len(self.parameter_grids) == 3:
                num_iters = self.parameter_grids[2].shape[2]
            else:
                num_iters = 1
            for i in range(num_iters):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                # Grab out grid data for colormap
                if self.N == 2:
                    if transpose:
                        xx = self.parameter_grids[1].T
                        yy = self.parameter_grids[0].T
                        grid = self.evalue_grid.real.T
                    else:
                        xx = self.parameter_grids[0]
                        yy = self.parameter_grids[1]
                        grid = self.evalue_grid.real
                elif self.N == 3:
                    if transpose:
                        xx = self.parameter_grids[1][:,:,i].T
                        yy = self.parameter_grids[0][:,:,i].T
                        grid = self.evalue_grid[:,:,i].real.T
                    else:
                        xx = self.parameter_grids[0][:,:,i]
                        yy = self.parameter_grids[1][:,:,i]
                        grid = self.evalue_grid[:,:,i].real

                # Plot colormap, only plot 2 stdevs off zero
                biggest_val = 2*np.abs(grid).std()
                plt.pcolormesh(xx,yy,grid,cmap='RdYlBu_r',vmin=-biggest_val,vmax=biggest_val)
                plt.colorbar()

                # Grab root data if they're available, plot them.
                if self.N == 2:
                    if transpose:
                        x = self.parameter_grids[1][0,:]
                        y = self.roots[:]
                    else:   
                        x = self.roots[:]
                        y = self.parameter_grids[1][0,:]
                elif self.N == 3:
                    if transpose:
                        x = self.parameter_grids[1][0,:,0]
                        y = self.roots[:,i]
                    else:   
                        x = self.roots[:,i]
                        y = self.parameter_grids[1][0,:,0]
                if transpose:
                    y, x = y[np.isfinite(y)], x[np.isfinite(y)]
                else:
                    y, x = y[np.isfinite(x)], x[np.isfinite(x)]
                plt.scatter(x,y)

                # Pretty up the plot, save.
                plt.ylim(yy.min(),yy.max())
                plt.xlim(xx.min(),xx.max())
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                if self.N == 2:
                    fig.savefig('{}.png'.format(title))
                else:
                    plt.title('z = {:.5g}'.format(self.parameter_grids[2][0,0,i]))
                    fig.savefig('{}_{:04d}.png'.format(title,i))
                plt.close(fig)
        else:
            logger.info("Plot is not implemented for > 3 dimensions")
