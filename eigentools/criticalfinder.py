import numpy as np
from mpi4py import MPI
import h5py
from scipy import interpolate, optimize
import matplotlib.pyplot as plt

from dedalus.tools.cache import CachedAttribute

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
    index = np.arange(dims.prod())
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
    
    def __init__(self, func, comm):
        self.func = func
        self.comm = comm
        self.nproc = self.comm.size
        self.rank = self.comm.rank

    def grid_generator(self, mins, maxs, dims, logs=None):
        """
        Generates an N-dimensional grid over the specified parameter
        space of an eigenvalue problem.  Once you get > 3 dimensions,
        memory issues will likely begin to be a problem.

        Inputs:
        -------
            mins    - The minimum values (in parameter space) to search in, for
                      each dimension of the problem.  An N-element NumPy array.
            maxs    - The maximum values (in parameter space) to search in, for
                      each dimension of the problem.  An N-element NumPy array.
            dims    - The number of points to use in each dimension of the grid in
                      the problem.  An N-element NumPy array.
            logs    - A boolean value, for each dimension.  If True, go between
                      the min and max value in that dimension in log-space.

        Example:
        --------
            # search from wavenumbers kx = [1, 10], ky = [1, 10] (20 pts each)
            # over Rayleigh Numbers Ra = [10, 1e5] (40 pts, log space)
            mins = np.array((1, 1, 10))
            maxs = np.array((10, 10, 1e5))
            dims = np.array((20, 20, 40))
            logs = np.array((False, False, True))
            obj = CriticalFinder(func, comm)
            obj.grid_generator(mins, maxs, dims, logs=logs)
        """
        # If logs aren't specified, do everything in linear space
        if logs is None:
            self.logs = np.array([False]*len(mins))
        else:
            self.logs = logs
        
        # Create parameter mesh grids for EVP solves
        ranges = []
        self.N = len(mins)
        for i in range(self.N):
            if self.logs[i]:
                ranges.append(np.logspace(np.log10(mins[i]), np.log10(maxs[i]),
                                          dims[i], dtype=np.float64))
            else:
                ranges.append(np.linspace(mins[i], maxs[i], dims[i], 
                                          dtype=np.float64))
        self.xyz_grids = np.meshgrid(*ranges, indexing='ij')
        self.grid = np.zeros_like(self.xyz_grids[0])
        
        # Split parameter load across processes
        load_indices = load_balance(dims, self.nproc)
        my_indices = load_indices[self.rank]

        # Calculate growth values for local process grid
        local_grid = np.empty(my_indices.size,dtype=np.complex128)
        for ii, index in enumerate(my_indices):
            if self.rank == 0:
                print("#######################################################")
                print("###### SOLVING LOCAL EVP {}/{} ON PROCESS {}".format(\
                                              ii+1, len(my_indices), self.rank))
                print("#######################################################")
            indices = index2indices(index, dims)
            values = []
            # Loop over each dimension to get N correct parameters
            for i, indx in enumerate(indices):
                zeros_before = i
                zeros_after = len(indices) - i - 1
                this_indx = [0]*zeros_before + [indx] + [0]*zeros_after
                values.append(self.xyz_grids[i][tuple(this_indx)])
            local_grid[ii] = np.complex128(self.func(*tuple(values))) #solve

        # Communicate growth modes across processes
        data = np.empty(dims.prod(), dtype='complex128')
        rec_counts = np.array([s.size for s in load_indices])
        displacements = np.cumsum(rec_counts) - rec_counts
        self.comm.Gatherv(local_grid,[data,rec_counts,displacements, MPI.F_DOUBLE_COMPLEX])
        data = data.reshape(*dims)
        self.comm.Bcast(data, root = 0)
        self.grid = data

    @CachedAttribute
    def interpolator(self):
        """
        On the first call, creates an interpolator of an N+1 dimensional function,
        where the first N dimensions are the grids created in grid_generator for all
        dimensions being explored, and the last dimension is the real component of the
        eigenvalues found.  Subsequent calls use the interpolator function, rather than
        recreating it.
        """
        if len(self.xyz_grids) == 2:
            # In two dimensions, interp2d works very well.
            xs, ys = self.xyz_grids[0][:,0], self.xyz_grids[1][0,:]
            if self.logs[0]:
                xs = np.log10(xs)
            if self.logs[1]:
                ys = np.log10(ys)
            xx, yy = np.meshgrid(xs, ys, indexing='ij')
            good_is = np.isfinite(self.grid.real)
            xs, ys, zs = xx[good_is], yy[good_is], self.grid.real[good_is]
            return interpolate.interp2d(xs, ys, zs)
            #return interpolate.interp2d(xx, yy, self.grid.real)
        else:
            # In N-dimensions, take advantage of regularly spaced grid
            # and use RegularGridInterpolator
            grids = []
            for i,g in enumerate(self.xyz_grids):
                # Run strings which slice out the parameter dimension
                indx = [0]*i + [range(g.shape[i])] + [0]*(self.N-i-1)
                if not self.logs[i]:
                    grids.append(g[indx])
                else:
                    grids.append(np.log10(g[indx]))
            interp = interpolate.RegularGridInterpolator(grids, self.grid.real)
            return lambda *args: interp(args)

    def use_interpolator(self, *args):
        """
        This function is a wrapper around the cached interpolator attribute.
        Input data along axes in log-space first have the log10 taken of them,
        then the interpolator is called on the logged data

        Inputs:
            *args       - An N-length data point at which to find the grid value.
        """
        args = list(args)
        for i, l in enumerate(self.logs):
            if l:
                args[i] = np.log10(np.array(args[i]))
        return self.interpolator(*args)

    def load_grid(self, filename, logs = None):
        """
        Load a grid file, in the format as created in save_grid.

        Inputs:
        -------
            filename:   The name of the .h5 file containing the grid data
            logs:       An N-length array of boolean values, used to
                        set self.logs.  If None, all axes are in linear space.
        """
        with h5py.File(filename,'r') as infile:
            self.xyz_grids = [k.value for k in infile.values() if 'xyz' in k.name]
            self.N = len(self.xyz_grids)
            print("Read an {}-dimensional grid".format(self.N))
            self.grid = infile['/grid'][:]
        if logs is None:
            self.logs = np.array([False]*self.N)
        else:
            self.logs = logs
        
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
                outfile.create_dataset('grid',data=self.grid)
                for i, grid in enumerate(self.xyz_grids):
                    outfile.create_dataset('xyz_{}'.format(i),data=grid)
    
    def root_finder(self):
        """
        For an N-dimensional problem, find the grid value at which the first dimension
        has a corresponding grid value of zero.

        For example, in a 3-dimensional grid of growth rates, and with the
        3 dimensions ranging over (a, A), (b, B), and (c, C), this
        function looks at all combinations of values in (b, B) and (c, C).  For
        each combination, it finds the value in (a, A) at which the growth rate
        grid passes through zero.  If the grid does not pass through zero in (a, A),
        then it returns NaN for that combination.
        """
        n_root_vals = np.prod(self.xyz_grids[0].shape[1:])
        self.roots = np.zeros(n_root_vals)
        for i in range(n_root_vals):
            indices = index2indices(i, self.xyz_grids[0].shape[1:])
            first_indx = tuple([0] + indices)
            last_indx  = tuple([-1] + indices)
            args = [self.xyz_grids[i+1][first_indx] for i in range(self.N-1)]
            try:
                self.roots[i] = optimize.brentq(self.use_interpolator, self.xyz_grids[0][first_indx],
                                                self.xyz_grids[0][last_indx], args=tuple(args))
            except ValueError:
                self.roots[i] = np.nan
        self.roots.reshape(self.xyz_grids[0].shape[1:])

    def crit_finder(self, find_freq=False, method='Powell'):
        """
        Using the root values found in self.root_finder, this function 
        determines the critical value of the first parameter dimension.

        For example, in a 2-dimensional problem, where the parameters
        range from (a, A) and (b, B), this function uses the information
        in roots to figure out the minimum value in the range (a, A) that contains
        a root.  It also finds the value in (b, B) that corresponds to 
        that critical value.

        ***Currently, this function only works for 2- and 3- dimensional grids
        TODO: Implement in higher dimensions.  interpolate.Rbf is an N-dim
              interpolator, but doesn't work well.


        inputs:
        ------
        find_freq       - If True, also return the imaginary component of the
                          critical value
        method          - The minimization method to use in 3+ dimensions

        output
        ------
        An N-length tuple of the critical values.  For example, in 2-D,
        (x_crit, y_crit) 

        """
        self.root_finder()
        mask = np.isfinite(self.roots)

        good_values = [array[0,mask] for array in self.xyz_grids[1:]]
        rroot = self.roots[mask]

        try:
            # Interpolate and find the minimum
            if self.N == 2:
                self.root_fn = interpolate.interp1d(good_values[0],rroot,kind='cubic')
                mid = rroot.argmin()
                if mid == len(good_values[0])-1 or mid == 0:
                    bracket = good_values[0][0], good_values[0][-1]
                else:
                    bracket = [good_values[0][0],good_values[0][mid],good_values[0][-1]]
                
                result = optimize.minimize_scalar(self.root_fn,bracket=bracket)
                # Often minimize_scalar forgets to return this, but if it doesn't
                # crash during the optimization, it's a success.
                result['success'] = True
            elif self.N == 3:
                self.root_fn = interpolate.interp2d(*good_values, rroot, kind='cubic')
                min_func = lambda arr: self.root_fn(*arr)
                guess_arg = rroot.argmin()
                init_guess = [arr[guess_arg] for arr in good_values]
                bound_vals = [(arr.min(), arr.max()) for arr in good_values]
                result = optimize.minimize(min_func, init_guess, bounds=bound_vals, method=method)
            else:
                raise ValueError("Critical find is not currently implemented in 4+ dimensions")

            # Store the values of parameters at which the minimum occur
            if result.success:
                crits = [np.asscalar(result['fun'])]
                try: #3+ dims
                    for i, x in enumerate(result['x']): crits.append(np.asscalar(x))
                except: #2 dims
                    crits.append(result['x'])
            else:
                crits = [np.nan]*self.N
           
            # If looking for the frequency, also get the imaginary value
            if find_freq:
                if result.success:
                    if self.N == 2:
                        freq_interp = interpolate.interp2d(self.xyz_grids[1],self.xyz_grids[0],self.grid.imag.T)
                        crits.append(freq_interp(*crits)[0])
                    elif self.N == 3: 
                        #In higher dims, just solve at crit to avoid bad interpolants
                       crits.append(self.func(*crits).imag)
                else:
                    crits.append(np.nan)
            return crits
        except:
            if self.rank == 0:
                print("An error occured while finding the critical value. Root values used are provided.")
                print("roots for all but first dim: {}".format(good_values))
                print("roots for first-dim (corresponding to previous array): {}".format(rroot))
                print("Returning NaN")
            raise
            return [np.nan]*self.N

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
            return [None]*len(self.xyz_grids)
        crits = self.crit_finder(method=method, **kwargs)
        if np.isnan(crits[0]):
            print("crit_finder returned NaN, cannot find exact crit")
            return crits

        # Create a lambda function that wraps the object's function, and returns
        # the absolute value of the growth rate.  Minimize that function.
        function = lambda *args: np.abs(self.func(*tuple([i*x for i,x in zip(args[0], crits)])).real)
        search_result = optimize.minimize(function, [1.0]*len(self.xyz_grids), 
                                          tol=tol, options={'maxiter': maxiter})

        print("Optimize results are as follows:")
        print(search_result)
        print("Best values found by optimize: {}".\
              format([np.asscalar(x*c) for x,c in zip(search_result.x, crits)]))

        if search_result.success:
            print('Minimum growth rate of {} found'.format(search_result.fun))
            return [np.asscalar(x*c) for x,c in zip(search_result.x, crits)]
        else:
            print('Optimize results not fully converged, returning crit_finder results.')
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
            if len(self.xyz_grids) == 3:
                num_iters = self.xyz_grids[2].shape[2]
            else:
                num_iters = 1
            for i in range(num_iters):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                # Grab out grid data for colormap
                if self.N == 2:
                    if transpose:
                        xx = self.xyz_grids[1].T
                        yy = self.xyz_grids[0].T
                        grid = self.grid.real.T
                    else:
                        xx = self.xyz_grids[0]
                        yy = self.xyz_grids[1]
                        grid = self.grid.real
                elif self.N == 3:
                    if transpose:
                        xx = self.xyz_grids[1][:,:,i].T
                        yy = self.xyz_grids[0][:,:,i].T
                        grid = self.grid[:,:,i].real.T
                    else:
                        xx = self.xyz_grids[0][:,:,i]
                        yy = self.xyz_grids[1][:,:,i]
                        grid = self.grid[:,:,i].real

                # Plot colormap, only plot 2 stdevs off zero
                biggest_val = 2*np.abs(grid).std()
                plt.pcolormesh(xx,yy,grid,cmap='RdYlBu_r',vmin=-biggest_val,vmax=biggest_val)
                plt.colorbar()

                # Grab root data if they're available, plot them.
                if self.N == 2:
                    if transpose:
                        x = self.xyz_grids[1][0,:]
                        y = self.roots[:]
                    else:   
                        x = self.roots[:]
                        y = self.xyz_grids[1][0,:]
                elif self.N == 3:
                    if transpose:
                        x = self.xyz_grids[1][0,:,0]
                        y = self.roots[:,i]
                    else:   
                        x = self.roots[:,i]
                        y = self.xyz_grids[1][0,:,0]
                if transpose:
                    y, x = y[np.isfinite(y)], x[np.isfinite(y)]
                else:
                    y, x = y[np.isfinite(x)], x[np.isfinite(x)]
                plt.scatter(x,y)

                # Pretty up the plot, save.
                plt.ylim(yy.min(),yy.max())
                plt.xlim(xx.min(),xx.max())
                if transpose:
                    if self.logs[1]:
                        plt.xscale('log')
                    if self.logs[0]:
                        plt.yscale('log')
                else:
                    if self.logs[0]:
                        plt.xscale('log')
                    if self.logs[1]:
                        plt.yscale('log')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                if self.N == 2:
                    fig.savefig('{}.png'.format(title))
                else:
                    plt.title('z = {:.5g}'.format(self.xyz_grids[2][0,0,i]))
                    fig.savefig('{}_{:04d}.png'.format(title,i))
                plt.close(fig)
        else:
            print("Plot is not implemented for > 3 dimensions")
