import numpy as np
from mpi4py import MPI
import h5py
from scipy import interpolate, optimize
import matplotlib.pyplot as plt

from dedalus.tools.cache import CachedAttribute

def load_balance(dims, nproc):
    """
    Evenly splits up tasks across the specified number of processes

    Inputs:
    -------
        dims    - An N-length array, containing the size of each dimension
                  of the eigenvalue problem.
        nproc   - The number of processes to split up the problem over
    
    Outputs:
    --------
        An array of arrays, each containing the task "numbers" for each process.

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
    indices = []
    for i in range(len(dims)):
        if i < len(dims) - 1:
            indices.append(np.int(np.floor(index/np.prod(dims[i+1:]))))
        else:
            indices.append(np.int(index % dims[i]))
    return indices

class CriticalFinder:
    """CriticalFinder

    This class provides simple tools for finding the critical
    parameters for the linear (in)stability of a given flow. Here, the
    parameter space must be 2D; typically this will be (k, Re), where
    k is a wavenumber and Re is some control parameter (Reynolds or
    Rayleigh or some such number). 

    Inputs: 

    func: a function of two variables (x,y) that provides growth rates
    at each (x, y). CriticalFinder will find the roots of the growth
    rate in x for each value of y.

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
            obj.grid_generator(mins, maxs, dims, logs)
        """
        if logs == None:
            logs = np.array([False]*len(mins))
        ranges = []
        for i in range(len(mins)):
            if logs[i]:
                ranges.append(np.logspace(np.log10(mins[i]), np.log10(maxs[i]),
                                          dims[i], dtype=np.float64))
            else:
                ranges.append(np.linspace(mins[i], maxs[i], dims[i], 
                                          dtype=np.float64))
        self.xyz_grids = np.meshgrid(*ranges)

        self.grid = np.zeros_like(self.xyz_grids[0])
        load_indices = load_balance(dims, self.nproc)
        my_indices = load_indices[self.rank]

        # work on parameters
        local_grid = np.empty(my_indices.size,dtype='complex128')

        for ii, index in enumerate(my_indices):
            indices = index2indices(index, dims)
            values = []
            for i, indx in enumerate(indices):
                zeros_before = len(indices) - i - 1
                zeros_after = i
                this_indx = [0]*zeros_before + [indx] + [0]*zeros_after
                values.append(self.xyz_grids[i][tuple(this_indx)])
            local_grid[ii] = self.func(*values)

        data = np.empty(dims.prod(), dtype='complex128')

        rec_counts = np.array([s.size for s in load_indices])
        displacements = np.cumsum(rec_counts) - rec_counts

        self.comm.Gatherv(local_grid,[data,rec_counts,displacements, MPI.F_DOUBLE_COMPLEX])

        data = data.reshape(*dims)
        self.comm.Bcast(data, root = 0)

        self.grid = data

    @CachedAttribute
    def interpolator(self):
        if len(self.xyz_grids) == 2:
            return interpolate.interp2d(self.xyz_grids[0][0,:], self.xyz_grids[1][:,0], self.grid.real)
        else:
            print("Creating N-dimensional interpolant function.  This may take a while...")
            return interpolate.Rbf(*self.xyz_grids, self.grid.real)

    def load_grid(self, filename):
        infile = h5py.File(filename,'r')
        self.xyz_grids = []
        try:
            count = 0
            while True:
                self.xyz_grids.append(infile['/xyz_{}'.format(i)][:])
                count += 1
        except:
            print("Read in {}-dimensional grid".format(len(self.xyz_grids)))
        self.grid = infile['/grid'][:]
        
    def save_grid(self, filen):
        if self.comm.rank == 0:
            outfile = h5py.File(filen+'.h5','w')
            outfile.create_dataset('grid',data=self.grid)
            for i, grid in enumerate(self.xyz_grids):
                outfile.create_dataset('xyz_{}'.format(i),data=grid)
            outfile.close()
    
    def root_finder(self):
        """
        For an N-dimensional problem, looking from numbers (a, A) in dim1,
        (b, B) in dim2, (c, C) in dim3, and so on, finds the value of dim1
        at which the eigenvalues cross zero for each other combination of values
        in each other dimension.

        TODO: Make this parallel.  Right now it's a huge (N-1)-dimensional for-loop
        """
        # This is mega-ugly, and needs to be improved to work better in N-dimensions than
        # N-1 nested for-loops.
        self.roots = np.zeros(self.xyz_grids[1].shape[1:])
        nested_loop_string = ""
        cap_a_char = 65 #chr(65) = A, and we can increment it to get diff characters
        low_a_char = 97
        for i in range(len(self.xyz_grids)-1):
            indx = i + 1
            colons_before = indx
            colons_after  = len(self.xyz_grids) - 1 - indx
            index_string = "0,"*colons_before + ":" + ",0"*colons_after
            nested_loop_string += '{}for {},{} in enumerate(self.xyz_grids[{}][{}]):\n'.\
                                            format("\t"*i,chr(cap_a_char+i), chr(low_a_char+i),\
                                                   indx, index_string)
        indxs = np.arange(len(self.xyz_grids)-1) + cap_a_char
        indxs = [chr(i) for i in indxs]
        args  = np.arange(len(self.xyz_grids)-1) + low_a_char
        args  = [chr(i) for i in args]
        indx_str = (len(indxs)*"{},").format(*indxs)
        args_str = (len(args)*"{},").format(*args)
        indx_str, args_str = indx_str[:-1], args_str[:-1]
        grid_indx_str = ",0"*(len(self.xyz_grids)-1)
        nested_loop_string += """
{0:}try:
{0:}    self.roots[{1:}] = optimize.brentq(self.interpolator,self.xyz_grids[0][0{3:}],self.xyz_grids[0][-1{3:}],args=({2:}))
{0:}except ValueError:
{0:}    self.roots[{1:}] = np.nan
        """.format("\t"*(len(self.xyz_grids)-1), indx_str, args_str, grid_indx_str)
        #print up string to debug it and make sure it's doing what we expect.
        #print(nested_loop_string)
        
        exec(nested_loop_string) #This is where the meat of the function actually happens

    def crit_finder(self, find_freq=False, method='Powell'):
        """returns a tuple of the x value at which the minimum (critical value
        occurs), and the y value. 

        output
        ------
        (x_crit, y_crit) 

        """
        self.root_finder()
        #print("my roots are", self.roots)
        mask = np.isfinite(self.roots)

        good_values = [array[0,mask] for array in self.xyz_grids[1:]]
        rroot = self.roots[mask]

        #Interpolate and find the minimum
        if len(self.xyz_grids) == 2:
            self.root_fn = interpolate.interp1d(good_values[0],rroot,kind='cubic')
            
            mid = int(len(good_values)/2)
            bracket = [good_values[0][0],good_values[0][mid],good_values[0][-1]]
            
            result = optimize.minimize_scalar(self.root_fn,bracket=bracket)
        else:
            print("Creating (N-1)-dimensional interpolant function for root finding. This may take a while...")
            self.root_fn = interpolate.Rbf(*good_values, rroot)
            min_func = lambda arr: self.root_fn(*arr)

            guess_arg = rroot.argmin()
            init_guess = [arr[guess_arg] for arr in good_values]
            bound_vals = [(arr.min(), arr.max()) for arr in good_values]

            # Not necessarily sure what the best method is to use here for a general problem.
            result = optimize.minimize(min_func, init_guess, bounds=bound_vals, method=method)

        #Store the values of parameters at which the minimum occur
        if result.success:
            crits = [np.asscalar(result.fun)]
            for x in result.x: crits.append(np.asscalar(x))
        else:
            crits = [np.nan]*len(self.xyz_grids)
       
        #If looking for the frequency, also get the imaginary value at the crit point
        if find_freq:
            if result.success:
                if len(self.xyz_grid) == 2:
                    freq_interp = interpolate.interp2d(self.yy,self.xx,self.grid.imag)
                else:
                    print("Creating (N)-dimensional interpolant function for frequency finding. This may take a while...")
                    freq_interp = interpolate.Rbf(*self.xyz_grid, self.grid.imag)
                crits.append(freq_interp(*crits)[0])
            else:
                crits.append(np.nan)
        return crits

    def plot_crit(self, title='growth_rates',transpose=True, xlabel = "", ylabel = ""):
        """make a simple plot of the growth rates and critical curve

        """
        if len(self.xyz_grid) > 2:
            raise Exception("Plot is not implemented for > 2 dimensions")
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if transpose:
            xx = self.xyz_grid[1][:,0].T
            yy = self.xyz_grid[0][0,:].T
            grid = self.grid.T
            x = self.xyz_grid[1][:,0]
            y = self.roots
        else:
            xx = self.xyz_grid[0]
            yy = self.xyz_grid[1]
            grid = self.grid
            x = self.roots
            y = self.xyz_grid[1][:,0]
        plt.pcolormesh(xx,yy,grid,cmap='autumn')#,vmin=-1,vmax=1)
        plt.colorbar()
        plt.scatter(x,y)
        plt.ylim(yy.min(),yy.max())
        plt.xlim(xx.min(),xx.max())
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fig.savefig('{}.png'.format(title))
