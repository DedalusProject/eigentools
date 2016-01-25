"""finds the critical Renoylds number and wave number for the
Orr-Somerfeld eigenvalue equation.

"""
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np
import matplotlib.pylab as plt

comm = MPI.COMM_WORLD


# Define the Orr-Somerfeld problem in Dedalus: 

z = de.Chebyshev('z',32)
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
    return EP.growth_rate({"alpha":x,"Re":y})

cf = CriticalFinder(shim, comm)

# generating the grid is the longest part
start = time.time()
cf.grid_generator(5500,6000,0.95,1.15,40,40)
end = time.time()
print("grid generation time: {:10.5f} sec".format(end-start))

cf.root_finder()
crit = cf.crit_finder()

if comm.rank == 0:
    print("critical wavenumber alpha = {:10.5f}".format(crit[0]))
    print("critical Re = {:10.5f}".format(crit[1]))
