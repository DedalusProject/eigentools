import pytest
import dedalus.public as de
import eigentools as eig
import numpy as np
from mpi4py import MPI

def rbc_problem(problem_type, domain, stress_free=False):
    problems = {'EVP': de.EVP, 'IVP': de.IVP}

    try:
        args = [domain,['p', 'b', 'u', 'w', 'bz', 'uz', 'wz']]
        if problem_type == 'EVP':
             args.append('omega')
        rayleigh_benard = problems[problem_type](*args)
    except KeyError:
        raise ValueError("problem_type must be one of 'EVP' or 'IVP', not {}".format(problem))

    rayleigh_benard.parameters['k'] = 3.117 #horizontal wavenumber
    rayleigh_benard.parameters['Ra'] = 1708. #Rayleigh number, rigid-rigid
    rayleigh_benard.parameters['Pr'] = 1  #Prandtl number
    rayleigh_benard.parameters['dzT0'] = 1
    if problem_type == 'EVP':
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
    if stress_free:
        rayleigh_benard.add_bc('left(uz) = 0')
        rayleigh_benard.add_bc('right(uz) = 0')
    else:
        rayleigh_benard.add_bc('left(u) = 0')
        rayleigh_benard.add_bc('right(u) = 0')

    return rayleigh_benard

@pytest.mark.parametrize('z', [de.Chebyshev('z',16, interval=(0, 1)), de.Compound('z',(de.Chebyshev('z',10, interval=(0, 0.5)),de.Chebyshev('z',10, interval=(0.5, 1))))])
@pytest.mark.parametrize('sparse', [True, False])
def test_rbc_growth(z, sparse):
    d = de.Domain([z])

    rayleigh_benard = rbc_problem('EVP',d)

    EP = eig.Eigenproblem(rayleigh_benard)

    growth, index, freq = EP.growth_rate(sparse=sparse)
    assert np.allclose((growth, freq), (0.0018125573647729994,0.)) 
 
@pytest.mark.parametrize('z', [de.Chebyshev('z',16, interval=(0, 1))])
def test_rbc_output(z):
    d = de.Domain([z])
    rb_evp = rbc_problem('EVP',d)
    EP = eig.Eigenproblem(rb_evp)

    growth, index, freq = EP.growth_rate(sparse=False)

    x = de.Fourier('x', 32)
    ivp_domain = de.Domain([x,z],grid_dtype=np.float64)

    fields = EP.project_mode(index, ivp_domain, [1,])
    EP.write_global_domain(fields)
    
    rb_IVP = rbc_problem('IVP', ivp_domain)
    solver =  rb_IVP.build_solver(de.timesteppers.RK222)
    solver.load_state("IVP_output/IVP_output_s1.h5",-1)

@pytest.mark.parametrize('z', [de.Chebyshev('z',16, interval=(0, 1))])
def test_rbc_crit_find(z):
    d = de.Domain([z], comm=MPI.COMM_SELF)
    rb_evp = rbc_problem('EVP', d, stress_free=True)
    EP = eig.Eigenproblem(rb_evp)
    comm = MPI.COMM_WORLD
    cf = eig.CriticalFinder(EP, ("k", "Ra"), comm, find_freq=True)

    nx = 10
    ny = 10
    xpoints = np.linspace(2, 2.4, nx)
    ypoints = np.linspace(550, 700, ny)

    cf.grid_generator((xpoints, ypoints),sparse=True)
    crit = cf.crit_finder(polish_roots=True, tol=1e-6, method='Powell')

    Rac = 27*np.pi**4/4.
    kc = 2*np.pi/2**1.5

    assert np.allclose(crit, [kc, Rac, 0.], rtol=1e-5)
