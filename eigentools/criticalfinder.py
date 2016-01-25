import numpy as np
from mpi4py import MPI
from scipy import interpolate, optimize

comm = MPI.COMM_WORLD

def load_balance(nx, ny, nproc):
# this is probably not a very good load balance for nprocs not exactly
# equal to a square of the total grid size...
    index = np.arange(nx*ny)

    return np.array_split(index,nproc)

def index2ji(index, nx):
    j = np.int(index / nx)
    i = index % nx
    return j, i 

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
        
    def grid_generator(self, xmin, xmax, ymin, ymax, nx, ny):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny

        self.yy,self.xx = np.mgrid[self.ymin:self.ymax:self.ny*1j,
                                   self.xmin:self.xmax:self.nx*1j]
        
        self.grid = np.zeros_like(self.xx)
        indices = load_balance(nx, ny, self.nproc)
        my_indices = indices[self.rank]

        # work on parameters
        local_grid = np.empty(my_indices.size,dtype='double')

        for ii, index in enumerate(my_indices):
            j, i = index2ji(index, nx)
            x = self.xx[0,i]
            y = self.yy[j,0]
            local_grid[ii] = self.func(y,x)

        data = np.empty(nx*ny, dtype='double')

        rec_counts = np.array([s.size for s in indices])
        displacements = np.cumsum(rec_counts) - rec_counts

        comm.Gatherv(local_grid,[data,rec_counts,displacements, MPI.DOUBLE])

        data = data.reshape(ny,nx)
        comm.Bcast(data, root = 0)

        self.grid = data
        self.interpolator = interpolate.interp2d(self.xx[0,:], self.yy[:,0], self.grid)
    
    def root_finder(self):
        self.roots = np.zeros_like(self.yy[:,0])
        for j,y in enumerate(self.yy[:,0]):
            try:
                self.roots[j] = optimize.brentq(self.interpolator,self.xx[0,0],self.xx[0,-1],args=(y))
            except ValueError:
                self.roots[j] = np.nan

    def crit_finder(self):
        """returns a tuple of the x value at which the minimum (critical value
        occurs), and the y value. 

        output
        ------
        (x_crit, y_crit) 

        """
        self.root_finder()
        
        mask = np.isfinite(self.roots)
        yy_root = self.yy[mask,0]
        rroot = self.roots[mask]
        self.root_fn = interpolate.interp1d(yy_root,rroot,kind='cubic')
        
        mid = yy_root.shape[0]/2
        
        bracket = [yy_root[0],yy_root[mid],yy_root[-1]]
        
        self.opt = optimize.minimize_scalar(self.root_fn,bracket=bracket)
        return (self.opt['x'], np.asscalar(self.opt['fun']))
