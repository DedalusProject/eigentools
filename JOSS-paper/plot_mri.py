"""
finds the critical magnetic Renoylds number and wave number for the magnetorotational instability (MRI).

This script can be run in parallel by using 

$ mpirun -np 4 python3 mri.py

It will parallelize over the grid generation portion and save that 

"""
import sys
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np
import matplotlib.pylab as plt

import logging

plt.style.use('prl')
logger = logging.getLogger(__name__.split('.')[-1])

comm = MPI.COMM_WORLD


# Define the MRI problem in Dedalus: 

x = de.Chebyshev('x',64)
d = de.Domain([x],comm=MPI.COMM_SELF)

mri = de.EVP(d,['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],'sigma')


Rm = 4.879
Pm = 0.001
mri.parameters['q'] = 1.5
mri.parameters['beta'] = 25.0
mri.parameters['iR'] = Pm/Rm
mri.parameters['Rm'] = Rm
mri.parameters['Q'] = 0.748
mri.substitutions['iRm'] = '1/Rm'

mri.add_equation("sigma*psixx - Q**2*sigma*psi - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - (2/beta)*1j*Q*dx(Ax) + (2/beta)*Q**3*1j*A = 0")
mri.add_equation("sigma*u - iR*dx(ux) + iR*Q**2*u - (q - 2)*1j*Q*psi - (2/beta)*1j*Q*B = 0") 
mri.add_equation("sigma*A - iRm*dx(Ax) + iRm*Q**2*A - 1j*Q*psi = 0") 
mri.add_equation("sigma*B - iRm*dx(Bx) + iRm*Q**2*B - 1j*Q*u + q*1j*Q*A = 0")

mri.add_equation("dx(psi) - psix = 0")
mri.add_equation("dx(psix) - psixx = 0")
mri.add_equation("dx(psixx) - psixxx = 0")
mri.add_equation("dx(u) - ux = 0")
mri.add_equation("dx(A) - Ax = 0")
mri.add_equation("dx(B) - Bx = 0")

mri.add_bc("left(u) = 0")
mri.add_bc("right(u) = 0")
mri.add_bc("left(psi) = 0")
mri.add_bc("right(psi) = 0")
mri.add_bc("left(A) = 0")
mri.add_bc("right(A) = 0")
mri.add_bc("left(psix) = 0")
mri.add_bc("right(psix) = 0")
mri.add_bc("left(Bx) = 0")
mri.add_bc("right(Bx) = 0")

# create an Eigenproblem object
EP = Eigenproblem(mri)

cf = CriticalFinder(EP, ("Q", "Rm"), comm, find_freq=False)

# generating the grid is the longest part
nx = 20
ny = 20
xpoints = np.linspace(0.5, 1.5, nx)
ypoints = np.linspace(4.6, 5.5, ny)

file_name = 'mri_growth_rate'
try:
    cf.load_grid('{}.h5'.format(file_name))
except:
    start = time.time()
    cf.grid_generator((xpoints, ypoints), sparse=True)
    end = time.time()

    if comm.rank == 0:
        cf.save_grid(file_name)
        logger.info("grid generation time: {:10.5f} sec".format(end-start))

crit = cf.crit_finder(polish_roots=False)

if comm.rank == 0:
    logger.info("critical Rm = {:10.5f}, Q = {:10.5f}".format(crit[1], crit[0]))

    x_unit = 1/23
    y_unit = 1/9
    aspect = x_unit/y_unit
    width = 12
    height = aspect*width
    fig = plt.figure(figsize=[width,height])

    left = 2*x_unit
    bottom = 2*y_unit
    width = 5*x_unit
    height = 5*y_unit

    crit_ax = fig.add_axes([left, bottom, width, height])
    spec_ax = fig.add_axes([left+width+2*x_unit, bottom, width, height])
    drft_ax = fig.add_axes([left+2*width+4*x_unit, bottom, width, height])
    # create plot of critical parameter space
    pax,cax = cf.plot_crit(axes=crit_ax)

    # add an interpolated critical line
    x_lim = cf.parameter_grids[0][0,np.isfinite(cf.roots)]
    x_hires = np.linspace(x_lim[0], x_lim[-1], 100)
    pax.plot(x_hires, cf.root_fn(x_hires), color='k')

    # plot the spectrum for the critical mode
    logger.info("solving dense eigenvalue problem for critical parameters")
    EP.solve(parameters = {"Q": crit[0], "Rm": crit[1]}, sparse=False)
    EP.plot_spectrum(axes=spec_ax, ylog=False)

    # mark critical mode
    eps = 1e-2
    mask = np.abs(EP.evalues.real) < eps
    spec_ax.scatter(EP.evalues[mask].real, EP.evalues[mask].imag, c='red')

    # plot drift ratio for critical mode
    EP.plot_drift_ratios(axes=drft_ax)
    fig.savefig('mri.png', dpi=300)
