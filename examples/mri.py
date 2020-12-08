"""finds the critical magnetic Renoylds number and wave number for the
MRI eigenvalue equation.

"""
import sys
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np
import matplotlib.pylab as plt

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
EP = Eigenproblem(mri, sparse=True)

cf = CriticalFinder(EP, ("Q", "Rm"), comm, find_freq=False)

# generating the grid is the longest part
start = time.time()
nx = 10
ny = 10
xpoints = np.linspace(0.5, 1.5, nx)
ypoints = np.linspace(4.6, 5.5, ny)
#cf.load_grid('mri_growth_rates.h5')
cf.grid_generator((xpoints, ypoints))
if comm.rank == 0:
    cf.save_grid('mri_growth_rates')
end = time.time()
print("grid generation time: {:10.5f} sec".format(end-start))
crit = cf.crit_finder(polish_roots=True)

if comm.rank == 0:
    print("critical Re = {:10.5f}".format(crit[1]))
    cf.plot_crit()
    cf.save_grid('mri_growth_rates')
