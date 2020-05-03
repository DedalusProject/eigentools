import pytest
import dedalus.public as de
import eigentools as eig
import numpy as np

@pytest.mark.parametrize('Nz', [16,32])
@pytest.mark.parametrize('sparse', [True, False])
def test_rbc_growth(Nz, sparse):
    z = de.Chebyshev('z',Nz, interval=(0, 1))
    d = de.Domain([z])

    rayleigh_benard = de.EVP(d,['p', 'b', 'u', 'w', 'bz', 'uz', 'wz'], eigenvalue='omega')
    rayleigh_benard.parameters['k'] = 3.117 #horizontal wavenumber
    rayleigh_benard.parameters['Ra'] = 1708. #Rayleigh number, rigid-rigid
    rayleigh_benard.parameters['Pr'] = 1  #Prandtl number
    rayleigh_benard.parameters['dzT0'] = 1
    rayleigh_benard.substitutions['dt(A)'] = 'omega*A'
    rayleigh_benard.substitutions['dx(A)'] = '1j*k*A'

    rayleigh_benard.add_equation("dx(u) + wz = 0")
    rayleigh_benard.add_equation("dt(u) - Pr*(dx(dx(u)) + dz(uz)) + dx(p)           = -u*dx(u) - w*uz")
    rayleigh_benard.add_equation("dt(w) - Pr*(dx(dx(w)) + dz(wz)) + dz(p) - Ra*Pr*b = -u*dx(w) - w*wz")
    rayleigh_benard.add_equation("dt(b) - w*dzT0 - (dx(dx(b)) + dz(bz)) = -u*dx(b) - w*bz")
    rayleigh_benard.add_equation("dz(u) - uz = 0")
    rayleigh_benard.add_equation("dz(w) - wz = 0")
    rayleigh_benard.add_equation("dz(b) - bz = 0")
    rayleigh_benard.add_bc('left(b) = 0')
    rayleigh_benard.add_bc('right(b) = 0')
    rayleigh_benard.add_bc('left(w) = 0')
    rayleigh_benard.add_bc('right(w) = 0')
    rayleigh_benard.add_bc('left(u) = 0')
    rayleigh_benard.add_bc('right(u) = 0')

    EP = eig.Eigenproblem(rayleigh_benard, sparse=sparse)

    growth, index, freq = EP.growth_rate({})
    assert np.allclose([growth, freq[0]], [0.0018125573647729994,0.]) 
 
