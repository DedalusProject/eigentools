"""mixed_layer.py

This script implements the eigenvalue problem posed in 

Boccaletti, Ferrari, & Fox-Kemper, 2007: 
Journal of Physical Oceanography, 37, 2228-2250.

This demonstrates the use of the grow_func option to Eigenproblem to specify a custom calculation method for determining the growth rate of a given eigenvalue.

Boccaletti, Ferrari, & Fox-Kemper definite the time dependence as

exp(i sigma t),

so the growth rate is -Im(sigma).

This script also gives an example of using `project_mode` to compute a 2D visualization of the most unstable eigenmode.
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

from dedalus.extras import plot_tools

logger = logging.getLogger(__name__.split('.')[-1])

file_name = sys.argv[0].strip('.py')
comm = MPI.COMM_WORLD

# problem parameters
kx = 0.8
ky = 0
δ = 0.1
Ri = 2
L = 1
Hy = 0.05
ΔB = 10

# discretization parameters
nz = 64
nx = 64

# Only need the z direction to compute eigenvalues
z = de.Chebyshev('z',nz, interval=[-1,0])
d = de.Domain([z],comm=MPI.COMM_SELF, grid_dtype=np.complex128)

# define a 2D domain to plot the eigenmode
x = de.Fourier('x',nx)
d_2d = de.Domain([x,z], grid_dtype=np.float64)

evp = de.EVP(d, ['u','v','w','p','b'],'sigma')

evp.parameters['Ri'] = Ri
evp.parameters['δ'] = δ
evp.parameters['L'] = L
evp.parameters['Hy'] = Hy
evp.parameters['ΔB'] = ΔB
evp.parameters['kx'] = kx
evp.parameters['ky'] = ky
evp.substitutions['dt(A)'] = '1j*sigma*A'
evp.substitutions['dx(A)'] = '1j*kx*A'
evp.substitutions['dy(A)'] = '1j*ky*A'
evp.substitutions['U'] = 'z + L'

evp.add_equation('dt(u) + U*dx(u) + w - v + Ri*dx(p) = 0')
evp.add_equation('dt(v) + U*dx(v)     + u + Ri*dy(p) = 0')
evp.add_equation('Ri*δ**2*(dt(w) + U*dx(w)) - Ri*b  + Ri*dz(p) = 0')
evp.add_equation('dt(b) + U*dx(b) - v/Ri + w = 0')
evp.add_equation('dx(u) + dy(v) + dz(w) = 0')

evp.add_bc('right(w) = 0')
#evp.add_bc('left(w) = 0')
evp.add_bc('left(w + dt(p/ΔB) + Hy*v) = 0')

EP = Eigenproblem(evp, sparse=False, grow_func=lambda x: -x.imag, freq_func=lambda x: x.real)

rate, indx, freq = EP.growth_rate()
print("fastest growing mode: {} @ freq {}".format(rate, freq))

# produce a 1-D plot of the most unstable eigenmode
fig = EP.plot_mode(indx)
fig.savefig('mixed_layer_unstable_1D.png')


# produce a 2-D plot of the most unstable eigenmode
# reproduces figure 7 from Boccaletti, Ferrari, and Fox-Kemper (2007)
fs = EP.project_mode(indx, d_2d, (1,))

scale=2.5
nrows = 4
ncols = 1
image = plot_tools.Box(4, 1)
pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)
mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
fig = mfig.figure

for var in ['b','u','v','w','p']:
    fs[var]['g']

# b
axes = mfig.add_axes(0,0, [0,0,1,1])
plot_tools.plot_bot_2d(fs['b'], axes=axes)

# u,w
axes = mfig.add_axes(1,0, [0,0,1,1])
data_slices = (slice(None), slice(None))
xx,zz = d_2d.grids()
xx,zz = np.meshgrid(xx,zz)
axes.quiver(xx[::2,::2],zz[::2,::2], fs['u']['g'][::2,::2].T, fs['w']['g'][::2,::2].T,zorder=10)
axes.set_xlabel('x')
axes.set_ylabel('z')
axes.set_title('u-w vectors')

# v
axes = mfig.add_axes(2,0, [0,0,1,1])
plot_tools.plot_bot_2d(fs['v'], axes=axes)

# eta
axes = mfig.add_axes(3,0, [0,0,1,1])
axes.plot(xx[0,:],fs['p']['g'][:,0])
axes.set_xlabel('x')
axes.set_ylabel(r'$\eta$')

fig.savefig('mixed_layer_unstable_2D.png',dpi=300)
