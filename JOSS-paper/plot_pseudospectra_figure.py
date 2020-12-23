import matplotlib.pyplot as plt
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import dedalus.public as de
import numpy as np
plt.style.use('prl')
z = de.Chebyshev('z', 128)
d = de.Domain([z],comm=MPI.COMM_SELF)

alpha = 1.
Re = 10000

primitive = de.EVP(d,['u','w','uz','wz', 'p'],'c')
primitive.parameters['alpha'] = alpha
primitive.parameters['Re'] = Re

primitive.substitutions['umean'] = '1 - z**2'
primitive.substitutions['umeanz'] = '-2*z'
primitive.substitutions['dx(A)'] = '1j*alpha*A' 
primitive.substitutions['dt(A)'] = '-1j*alpha*c*A'
primitive.substitutions['Lap(A,Az)'] = 'dx(dx(A)) + dz(Az)'

primitive.add_equation('dt(u) + umean*dx(u) + w*umeanz + dx(p) - Lap(u, uz)/Re = 0')
primitive.add_equation('dt(w) + umean*dx(w) + dz(p) - Lap(w, wz)/Re = 0')
primitive.add_equation('dx(u) + wz = 0')
primitive.add_equation('uz - dz(u) = 0')
primitive.add_equation('wz - dz(w) = 0')
primitive.add_bc('left(w) = 0')
primitive.add_bc('right(w) = 0')
primitive.add_bc('left(u) = 0')
primitive.add_bc('right(u) = 0')

prim_EP = Eigenproblem(primitive)

# define the energy norm 
def energy_norm(Q1, Q2):
    u1 = Q1['u']
    w1 = Q1['w']
    u2 = Q2['u']
    w2 = Q2['w']
    
    field = (np.conj(u1)*u2 + np.conj(w1)*w2).evaluate().integrate()
    return field['g'][0]
    
# Calculate pseudospctrum
k = 100 # size of invariant subspace

psize = 100 # number of points in real, imaginary points
real_points = np.linspace(0,1, psize)
imag_points = np.linspace(-1,0.1, psize)
prim_EP.calc_ps(k, (real_points, imag_points), inner_product=energy_norm)

# plot
x_unit = 1/15
y_unit = 1/7
aspect = x_unit/y_unit
width = 10
height = aspect *width
fig = plt.figure(figsize=[width,height])

left = x_unit
bottom = y_unit
width = 2*x_unit
height = 2*y_unit
c_width = 5*x_unit
c_height = 5*y_unit
mode_axes = []
mode_axes.append(fig.add_axes([left,bottom,width,height]))
mode_axes.append(fig.add_axes([left,bottom+height+y_unit,width,height]))
mode_axes.append(fig.add_axes([left+width+c_width+4*x_unit,bottom,width,height]))
mode_axes.append(fig.add_axes([left+width+c_width+4*x_unit,bottom+height+y_unit,width,height]))
spec_axes = fig.add_axes([left+width+2*x_unit,bottom, c_width, c_height])


CS = spec_axes.contour(prim_EP.ps_real,prim_EP.ps_imag, np.log10(prim_EP.pseudospectrum),levels=np.arange(-8,0),linestyles='solid',colors='k')
spec_axes.clabel(CS,inline=1,fmt='%d')
spec_axes.scatter(prim_EP.evalues.real, prim_EP.evalues.imag,color='blue',marker='o',zorder=2,alpha=0.7)
spec_axes.set_xlim(0,1)
spec_axes.set_ylim(-1,0.1)
spec_axes.axhline(0,color='k',alpha=0.2)
spec_axes.set_xlabel(r"$c_r$")
spec_axes.set_ylabel(r"$c_i$")

modes = [0,2, 31, -1]
spec_axes.scatter(prim_EP.evalues[modes].real, prim_EP.evalues[modes].imag,color='black',marker='o',facecolors='none',zorder=3)
z = prim_EP.grid()
for i, m in enumerate(modes):
    state = prim_EP.eigenmode(m)
    evalue = prim_EP.evalues[m]
    norm = np.abs(state['w']['g']).max()
    mode_axes[i].plot(z, np.abs(state['u']['g'])/norm, label=r'$|u|$')
    mode_axes[i].plot(z, np.abs(state['w']['g'])/norm, label=r'$|w|$')

    mode_axes[i].set_title(r'$(c_r, c_i) = ({:4.3f},{:4.3f})$'.format(evalue.real, evalue.imag), fontsize=10)

mode_axes[0].legend(fontsize=12)
mode_axes[0].set_xlabel(r"$z$")
mode_axes[0].set_ylabel(r"$|\mathbf{u}|$")
fig.savefig("pseudospectra.png", dpi=300)
