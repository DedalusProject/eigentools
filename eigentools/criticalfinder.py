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

    def grid_generator(self, mins, maxs, dims, logs):
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
        ranges = []
        for i in range(len(mins)):
            if logs[i]:
                ranges.append(np.logspace(np.log10(mins[i]), np.log10(maxs[i]),
                                          dims[i], dtype=np.complex128))
            else:
                ranges.append(np.linspace(mins[i], maxs[i], dims[i], 
                                          dtype=np.complex128))
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
                indices = [0]*zeros_before + [indx] + [0]*zeros_after
                values.append(self.xyz_grids[i][*indices])
            local_grid[ii] = self.func(*values)[0]

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
            return interpolate.Rbf(*self.xyz_grids, self.grid.real)

    def load_grid(self, filename):
        infile = h5py.File(filename,'r')
        self.xyz_grid = []
        try:
            count = 0
            while True:
                self.xyz_grid.append(infile['/xyz_{}'.format(i)][:])
                count += 1
        except:
            print("Read in {}-dimensional grid".format(len(self.xyz_grid))
        self.grid = infile['/grid'][:]
        
    def save_grid(self, filen):
        if self.comm.rank == 0:
            outfile = h5py.File(filen+'.h5','w')
            outfile.create_dataset('grid',data=self.grid)
            for i, grid in enumerate(self.xyz_grids):
                outfile.create_dataset('xyz_{}'.format(i),data=grid)
            outfile.close()
    
    def root_finder(self):
        
        # This is mega-ugly, and needs to be improved to work better in N-dimensions than
        # N-1 nested for-loops.
        self.roots = np.zeros(self.xyz_grid[1].shape[1:])
        nested_loop_string = ""
        cap_a_char = 65 #chr(65) = A, and we can increment it to get diff characters
        low_a_char = 97
        for i in range(len(self.roots)-1):
            indx = i + 1
            colons_before = indx
            colons_after  = len(self.roots) - 1 - indx
            index_string = "0,"*colons_before + ":" + ",0"*colons_after
            nested_loop_string += '{}for {},{} in enumerate(self.xyz_grid[{}][{}]:\n'.\
                                            format("\t"*i,chr(cap_a_char+i), chr(low_a_char+i),\
                                                   indx, index_string)
        indxs = np.arange(len(self.roots)-1) + cap_a_char
        indxs = [chr(i) for i in indxs]
        args  = np.arange(len(self.roots)-1) + low_a_char
        args  = [chr(i) for i in args]
        indx_str = (len(indxs)*"{},").format(*indxs)
        args_str = (len(args)*"{},").format(*args)
        indx_str, args_str = indx_str[:-1], args_str[:-1]
        grid_indx_str = ",0"*(len(self.roots)-1)
        nested_loop_string += """
            {0:}try:
            {0:}    self.roots[{1:}] = optimize.brentq(self.interpolator,self.xyz_grid[0][0{3:}],self.xyz_grid[0][-1{3:}],args=({2:}))
            {0:}except ValueError:
            {0:}    self.roots[{1:}] = np.nan
        """.format("\t"*(len(self.roots)-1), indx_str, args_str, grid_indx_str)
        #print up string to debug it and make sure it's doing what we expect.
        exec(nested_loop_string) #This is where the meat of the function actually happens

    def crit_finder(self, find_freq=False):
        """returns a tuple of the x value at which the minimum (critical value
        occurs), and the y value. 

        output
        ------
        (x_crit, y_crit) 

        """
        self.root_finder()
        #print("my roots are", self.roots)
        mask = np.isfinite(self.roots)
        yy_root = self.yy[mask,0]
        rroot = self.roots[mask]
        #print("interpolating over yyroot ", yy_root, "and rroot", rroot)
        self.root_fn = interpolate.interp1d(yy_root,rroot,kind='cubic')
        
        mid = yy_root.shape[0]/2
        
        bracket = [yy_root[0],yy_root[mid],yy_root[-1]]
        
        self.opt = optimize.minimize_scalar(self.root_fn,bracket=bracket)

        x_crit = self.opt['x']
        y_crit = np.asscalar(self.opt['fun'])

        if find_freq:
            freq_interp = interpolate.interp2d(self.yy,self.xx,self.grid.imag)
            crit_freq = freq_interp(x_crit, y_crit)[0]
            return (x_crit, y_crit, crit_freq)

        return (x_crit, y_crit)

    def plot_crit(self, title='growth_rates',transpose=True, xlabel = "", ylabel = ""):
        """make a simple plot of the growth rates and critical curve

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if transpose:
            xx = self.yy.T
            yy = self.xx.T
            grid = self.grid.T
            x = self.yy[:,0]
            y = self.roots
        else:
            xx = self.xx
            yy = self.yy
            grid = self.grid
            x = self.roots
            y = self.yy[:,0]

        plt.pcolormesh(xx,yy,grid,cmap='autumn')#,vmin=-1,vmax=1)
        plt.colorbar()
        plt.scatter(x,y)
        plt.ylim(yy.min(),yy.max())
        plt.xlim(xx.min(),xx.max())
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fig.savefig('{}.png'.format(title))
