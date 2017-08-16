"""finds the critical Renoylds number and wave number for the
Orr-Somerfeld eigenvalue equation.


"""
import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np
import matplotlib.pylab as plt

comm = MPI.COMM_WORLD


# Define the Orr-Somerfeld problem in Dedalus: 

z = de.Chebyshev('z',50)
d = de.Domain([z],comm=MPI.COMM_SELF)

orr_somerfeld = de.EVP(d,['w','wz','wzz','wzzz'],'sigma')
orr_somerfeld.parameters['alpha'] = 1.
orr_somerfeld.parameters['Re'] = 10000.

orr_somerfeld.add_equation('dz(wzzz) - 2*alpha**2*wzz + alpha**4*w - sigma*(wzz-alpha**2*w)-1j*alpha*(Re*(1-z**2)*(wzz-alpha**2*w) + 2*Re*w) = 0 ')
orr_somerfeld.add_equation('dz(w)-wz = 0')
orr_somerfeld.add_equation('dz(wz)-wzz = 0')
orr_somerfeld.add_equation('dz(wzz)-wzzz = 0')

orr_somerfeld.add_bc('left(w) = 0')
orr_somerfeld.add_bc('right(w) = 0')
orr_somerfeld.add_bc('left(wz) = 0')
orr_somerfeld.add_bc('right(wz) = 0')

# create an Eigenproblem object
EP = Eigenproblem(orr_somerfeld)

# create a shim function to translate (x, y) to the parameters for the eigenvalue problem:

def shim(x,y):
    gr, indx, freq = EP.growth_rate({"Re":x,"alpha":y})
    ret = gr+1j*freq
    if type(ret) == np.ndarray:
        return ret[0]
    else:
        return ret

cf = CriticalFinder(shim, comm)

# generating the grid is the longest part
start = time.time()
mins = np.array((5500, 0.95))
maxs = np.array((6000, 1.15))
nums = np.array((20  , 20))
try:
    cf.load_grid('orr_sommerfeld_growth_rates.h5')
except:
    cf.grid_generator(mins, maxs, nums)
    if comm.rank == 0:
        cf.save_grid('orr_sommerfeld_growth_rates')
end = time.time()
if comm.rank == 0:
    print("grid generation time: {:10.5f} sec".format(end-start))

cf.root_finder()
crit = cf.crit_finder(find_freq = True)

if comm.rank == 0:
    print("crit = {}".format(crit))
    print("critical wavenumber alpha = {:10.5f}".format(crit[1]))
    print("critical Re = {:10.5f}".format(crit[0]))
    print("critical omega = {:10.5f}".format(crit[2]))

    cf.plot_crit()

