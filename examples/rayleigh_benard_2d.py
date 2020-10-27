"""
Finds the critical Rayleigh number and wavenumber for the 2-dimensional,
incompressible, Boussinesq Navier-Stokes equations in order to determine
the onset of convection in such a system.
"""
import matplotlib
matplotlib.use('Agg')
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np

comm = MPI.COMM_WORLD


no_slip = False
stress_free = True
file_name = 'rayleigh_benard_growth_rates'
if no_slip:
    file_name += '_no_slip'
elif stress_free:
    file_name += '_stress_free'

Nz = 16
z = de.Chebyshev('z',Nz, interval=(0, 1))
d = de.Domain([z],comm=MPI.COMM_SELF)

rayleigh_benard = de.EVP(d,['p', 'b', 'u', 'w', 'bz', 'uz', 'wz'], eigenvalue='omega')
rayleigh_benard.parameters['k'] = 3.117 #horizontal wavenumber
rayleigh_benard.parameters['Ra'] = 1708. #Rayleigh number, rigid-rigid
rayleigh_benard.parameters['Pr'] = 1  #Prandtl number
rayleigh_benard.parameters['dzT0'] = 1
rayleigh_benard.substitutions['dt(A)'] = 'omega*A'
rayleigh_benard.substitutions['dx(A)'] = '1j*k*A'

#Boussinesq eqns -- nondimensionalized on thermal diffusion timescale
#Incompressibility
rayleigh_benard.add_equation("dx(u) + wz = 0")
#Momentum eqns
rayleigh_benard.add_equation("dt(u) - Pr*(dx(dx(u)) + dz(uz)) + dx(p)           = -u*dx(u) - w*uz")
rayleigh_benard.add_equation("dt(w) - Pr*(dx(dx(w)) + dz(wz)) + dz(p) - Ra*Pr*b = -u*dx(w) - w*wz")
#Temp eqn
rayleigh_benard.add_equation("dt(b) - w*dzT0 - (dx(dx(b)) + dz(bz)) = -u*dx(b) - w*bz")
#Derivative defns
rayleigh_benard.add_equation("dz(u) - uz = 0")
rayleigh_benard.add_equation("dz(w) - wz = 0")
rayleigh_benard.add_equation("dz(b) - bz = 0")



#fixed temperature
rayleigh_benard.add_bc('left(b) = 0')
rayleigh_benard.add_bc('right(b) = 0')
#Impenetrable
rayleigh_benard.add_bc('left(w) = 0')
rayleigh_benard.add_bc('right(w) = 0')


if no_slip:
    rayleigh_benard.add_bc('left(u) = 0')
    rayleigh_benard.add_bc('right(u) = 0')
elif stress_free:
    rayleigh_benard.add_bc('left(uz) = 0')
    rayleigh_benard.add_bc('right(uz) = 0')

# create an Eigenproblem object
EP = Eigenproblem(rayleigh_benard, sparse=True)

cf = CriticalFinder(EP, ("Ra", "k"), comm)

# generating the grid is the longest part
start = time.time()
if no_slip:
    nx = 20
    ny = 20
    xpoints = np.linspace(1000, 3000, nx)
    ypoints = np.linspace(2, 4, ny)
elif stress_free:
    #657.5, 2.221
    nx = 10
    ny = 10
    xpoints = np.linspace(550, 700, nx)
    ypoints = np.linspace(2, 2.4, ny)

try:
    cf.load_grid('{}.h5'.format(file_name))
except:
    cf.grid_generator((xpoints, ypoints))
    cf.save_grid(file_name)

end = time.time()
if comm.rank == 0:
    print("grid generation time: {:10.5f} sec".format(end-start))

crit = cf.crit_finder(find_freq = True)

if comm.rank == 0:
    print("crit = {}".format(crit))
    print("critical wavenumber k = {:10.5f}".format(crit[0]))
    print("critical Ra = {:10.5f}".format(crit[1]))
    print("critical freq = {:10.5f}".format(crit[2]))

    cf.plot_crit(title=file_name, transpose=True, xlabel='kx', ylabel='Ra')
