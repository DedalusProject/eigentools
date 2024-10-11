import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from eigentools import Eigenproblem

import logging
logger = logging.getLogger(__name__)

def wave_on_string_EVP(Nx, use_legendre=True):

    Lx = 1
    dtype = np.complex128

    # Bases
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)
    if use_legendre:
        xbasis = d3.Legendre(xcoord, size=Nx, bounds=(0, Lx))
    else:
        xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx))

    # Fields
    u = dist.Field(name='u', bases=xbasis)
    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    s = dist.Field(name='s')

    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)
    lift_basis = xbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    ux = dx(u) + lift(tau_1) # First-order reduction
    uxx = dx(ux) + lift(tau_2)

    # Problem
    problem = d3.EVP([u, tau_1, tau_2], eigenvalue=s, namespace=locals())
    problem.add_equation("s*u + uxx = 0")
    problem.add_equation("u(x=0) = 0")
    problem.add_equation("u(x=Lx) = 0")

    return problem

def run_rejection(Nx, method):
    if method == 'resolution':
        Nx_hi = int(1.5*Nx)
        lo_res_EVP = wave_on_string_EVP(Nx)
        hi_res_EVP = wave_on_string_EVP(Nx_hi)
        ep = Eigenproblem(lo_res_EVP, reject='distance', EVP_secondary=hi_res_EVP)
    elif method == 'basis':
        cheb_EVP = wave_on_string_EVP(Nx, use_legendre=False)
        leg_EVP = wave_on_string_EVP(Nx, use_legendre=True)
        ep = Eigenproblem(cheb_EVP, reject='distance', EVP_secondary=leg_EVP)

    ep.solve()
    num_evals_lo = len(ep.evalues_primary)
    num_evals_kept = len(ep.evalues)

    num_rejected = num_evals_lo - num_evals_kept
    print(f"{method} : {num_rejected} rejected eigenmodes.")

if __name__ == "__main__":
    Nx = 128
    run_rejection(Nx, 'resolution')
    run_rejection(Nx, 'basis')

    
