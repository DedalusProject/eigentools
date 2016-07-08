import numpy as np
from mpi4py import MPI
import h5py
from scipy import interpolate, optimize
import matplotlib.pyplot as plt

from dedalus.tools.cache import CachedAttribute

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

        self.comm.Gatherv(local_grid,[data,rec_counts,displacements, MPI.DOUBLE])

        data = data.reshape(ny,nx)
        self.comm.Bcast(data, root = 0)

        self.grid = data

    @CachedAttribute
    def interpolator(self):
        return interpolate.interp2d(self.xx[0,:], self.yy[:,0], self.grid)

    def load_grid(self, filename):
        infile = h5py.File(filename,'r')
        self.xx = infile['/xx'][:]
        self.yy = infile['/yy'][:]
        self.grid = infile['/grid'][:]
        
    def save_grid(self, filen):
        if self.comm.rank == 0:
            outfile = h5py.File(filen+'.h5','w')
            outfile.create_dataset('grid',data=self.grid)
            outfile.create_dataset('xx',data=self.xx)
            outfile.create_dataset('yy',data=self.yy)
            outfile.close()
    
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
        #print("my roots are", self.roots)
        mask = np.isfinite(self.roots)
        yy_root = self.yy[mask,0]
        rroot = self.roots[mask]
        #print("interpolating over yyroot ", yy_root, "and rroot", rroot)
        self.root_fn = interpolate.interp1d(yy_root,rroot,kind='cubic')
        
        mid = yy_root.shape[0]/2
        
        bracket = [yy_root[0],yy_root[mid],yy_root[-1]]
        
        self.opt = optimize.minimize_scalar(self.root_fn,bracket=bracket)
        return (self.opt['x'], np.asscalar(self.opt['fun']))

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
