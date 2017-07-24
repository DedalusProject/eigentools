"""finds the critical Renoylds number and wave number for the
Orr-Somerfeld eigenvalue equation.

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

x = de.Chebyshev('x',24)
d = de.Domain([x],comm=MPI.COMM_SELF)

mri = de.EVP(d,['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],'sigma')


Rm = 4.879
Pm = 0.001
mri.parameters['q'] = 1.5
mri.parameters['beta'] = 25.0
mri.parameters['Re'] = Rm
mri.parameters['Pr'] = Pm
mri.substitutions['iR'] = '(Pr/Re)'
mri.substitutions['iRm'] = '(1./Re)'
#mri.parameters['iR'] = Pm/Rm
#mri.parameters['iRm'] = 1./Rm
mri.parameters['Q'] = 0.748

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

# create a shim function to translate (x, y) to the parameters for the eigenvalue problem:
def shim(x,y):
    gr, indx, freq = EP.growth_rate({"Re":x,"Q":y})
    return gr

cf = CriticalFinder(shim, comm)

# generating the grid is the longest part
start = time.time()
mins = np.array((4.6, 0.74))
maxs = np.array((5.0, 0.76))
ns   = np.array((4, 4))
cf.grid_generator(mins, maxs, ns)
for i, g in enumerate(cf.xyz_grids): print(g)
print(cf.grid)
end = time.time()
print("grid generation time: {:10.5f} sec".format(end-start))

cf.root_finder()
print(cf.roots)
crit = cf.crit_finder()

if comm.rank == 0:
    print("critical wavenumber alpha = {:10.5f}".format(crit[0]))
    print("critical Re = {:10.5f}".format(crit[1]))

    cf.plot_crit()
    cf.save_grid('mri_growth_rates')
