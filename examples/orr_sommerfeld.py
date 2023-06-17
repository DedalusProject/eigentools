"""finds the critical Renoylds number, wave number, and frequency for the
Orr-Somerfeld eigenvalue equation.

NB: This formulation uses a slightly different scaling of the eigenvalue than Orszag (1971). In order to convert, use 

sigma = -1j*alpha*Re*lambda,

where sigma is our eigenvalue and Lambda is Orszag's.
"""
import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np
import matplotlib.pylab as plt
import sys
import logging
logger = logging.getLogger(__name__.split('.')[-1])

file_name = sys.argv[0].strip('.py')
comm = MPI.COMM_WORLD


# Define the Orr-Somerfeld problem in Dedalus: 

z = de.Chebyshev('z', 50)
d = de.Domain([z],comm=MPI.COMM_SELF)

orr_somerfeld = de.EVP(d,['w', 'wz', 'wzz', 'wzzz'], 'sigma')
orr_somerfeld.parameters['alpha'] = 1.
orr_somerfeld.parameters['Re'] = 10000.

orr_somerfeld.add_equation('dz(wzzz) - 2*alpha**2*wzz + alpha**4*w - sigma*(wzz-alpha**2*w)-1j*alpha*(Re*(1-z**2)*(wzz-alpha**2*w) + 2*Re*w) = 0 ')
orr_somerfeld.add_equation('dz(w) - wz = 0')
orr_somerfeld.add_equation('dz(wz) - wzz = 0')
orr_somerfeld.add_equation('dz(wzz) - wzzz = 0')

orr_somerfeld.add_bc('left(w) = 0')
orr_somerfeld.add_bc('right(w) = 0')
orr_somerfeld.add_bc('left(wz) = 0')
orr_somerfeld.add_bc('right(wz) = 0')

# create an Eigenproblem object
EP = Eigenproblem(orr_somerfeld)

# create a shim function to translate (x, y) to the parameters for the eigenvalue problem:

cf = CriticalFinder(EP,("alpha", "Re"), comm, find_freq=True)

# generating the grid is the longest part
start = time.time()
nx = 20
ny = 20
xpoints = np.linspace(1.0, 1.1, nx)
ypoints = np.linspace(5500, 6000, ny)
try:
    cf.load_grid('{}.h5'.format(file_name))
except:
    cf.grid_generator((xpoints, ypoints), sparse=True)
    if comm.rank == 0:
        cf.save_grid(file_name)
end = time.time()
if comm.rank == 0:
    logger.info("grid generation time: {:10.5f} sec".format(end-start))

crit = cf.crit_finder(polish_roots=True, tol=1e-5, method='Nelder-Mead')

Re_orszag = 5772.22
alpha_orszag = 1.02056
omega_orszag = -1555.2070

if comm.rank == 0:
    alpha = crit[0]
    Re = crit[1]
    omega = crit[2]

    Re_err = (Re-Re_orszag)/Re_orszag
    alpha_err = (alpha-alpha_orszag)/alpha_orszag
    L2 = np.sqrt((Re-Re_orszag)**2 + (alpha-alpha_orszag)**2)
    logger.info("critical wavenumber alpha = {:10.5f}".format(alpha))
    logger.info("critical Re = {:10.5f}".format(Re))
    logger.info("critical omega = {:10.5f}".format(omega))
    logger.info("critical Re error = {:10.5e}".format(Re_err))
    logger.info("critical alpha error = {:10.5}".format(alpha_err))
    logger.info("L2 norm from Orszag 71 solution = {:10.5e}".format(L2))

    cf.save_grid('orr_sommerfeld_growth_rates')
    cf.plot_crit()
