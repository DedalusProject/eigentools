import numpy as np
from scipy import interpolate, optimize

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
    
    def __init__(self, func):
        self.func = func
        
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
        
        for j,y in enumerate(self.yy[:,0]):
            for i,x in enumerate(self.xx[0,:]):
                self.grid[j,i] = self.func(y,x)


        self.interpolator = interpolate.interp2d(self.xx[0,:], self.yy[:,0], self.grid)
    
    def root_finder(self):
        self.roots = np.zeros_like(self.yy[:,0])
        for j,y in enumerate(self.yy[:,0]):
            try:
                self.roots[j] = optimize.brentq(self.interpolator,self.xx[0,0],self.xx[0,-1],args=(y))
            except ValueError:
                self.roots[j] = np.nan

    def crit_finder(self):
        self.root_finder()
        
        mask = np.isfinite(self.roots)
        yy_root = self.yy[mask,0]
        rroot = self.roots[mask]
        self.root_fn = interpolate.interp1d(yy_root,rroot,kind='cubic')
        
        mid = yy_root.shape[0]/2
        
        bracket = [yy_root[0],yy_root[mid],yy_root[-1]]
        
        self.opt = optimize.minimize_scalar(self.root_fn,bracket=bracket)
        return (self.opt['x'], self.opt['fun'])
