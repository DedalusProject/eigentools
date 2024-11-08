import sys
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from eigentools import Eigenproblem, ResidualPair

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
        ep = Eigenproblem(leg_EVP, reject='distance', EVP_secondary=cheb_EVP)
    elif method == 'tau':
        EVP = wave_on_string_EVP(Nx)
        rp1 = ResidualPair(tau='tau_1', var='u')
        rp2 = ResidualPair(tau='tau_2', var='u')
        for rp in (rp1, rp2):
            print(f"var = {rp.var}, tau = {rp.tau}")
        ep = Eigenproblem(EVP, reject='tau', tau_residual_pairs=(rp1, rp2), rejection_tolerance=1e-4)
    elif method =='truncation':
        EVP = wave_on_string_EVP(Nx)
        ep = Eigenproblem(EVP, reject='truncation', rejection_tolerance=1e-5)
    ep.solve()
    num_evals_lo = len(ep.evalues_primary)
    num_evals_kept = len(ep.evalues)

    num_rejected = num_evals_lo - num_evals_kept
    print(f"{method} : {num_rejected} rejected eigenmodes.")

    return ep

def plot_mode(eigenproblem, index, kept):
    plt.clf()
    eigenproblem.solver.set_state(index)

    u = eigenproblem.EVP.namespace['u']
    tau_1 = eigenproblem.EVP.namespace['tau_1']
    tau_2 = eigenproblem.EVP.namespace['tau_2']
    eval = eigenproblem.evalues_primary[index].real
    sigma = np.sqrt(eval)/np.pi
    x = u.domain.bases[0].local_grid(u.dist,1)
    plt.subplot(211)
    plt.semilogy(np.abs(u['c']), label = f"tau_1 = {tau_1['c'][0].real:.3e}, tau_2 = {tau_2['c'][0].real:.3e}")
    plt.axhline(1e-5, color='k', alpha=0.7)
    plt.legend()
    plt.ylabel('coeff')
    plt.title(f"sigma = {sigma:.0f}, kept= {kept}")
    plt.subplot(212)
    plt.plot(x, u['g'].real)
    plt.ylabel('grid')
    print(f"saving mode {index}.")
    plt.tight_layout()
    plt.savefig(f"mode_{index}.png",dpi=300)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Nx = 128
    ep_c = run_rejection(Nx, 'truncation')
    ep_r = run_rejection(Nx, 'resolution')
    ep_b = run_rejection(Nx, 'basis')
    ep_t = run_rejection(Nx, 'tau')
    n = 1 + np.arange(Nx)
    true_evals = (n * np.pi)**2
    
    plt.semilogy(np.abs(ep_r.evalues.real-true_evals[slice(0,len(ep_r.evalues))]), label='resolution', alpha=0.5, marker='o')
    plt.semilogy(np.abs(ep_b.evalues.real-true_evals[slice(0,len(ep_b.evalues))]),label='basis', alpha=0.5, marker='x')
    plt.semilogy(np.abs(np.sort(ep_t.evalues.real)-true_evals[slice(0,len(ep_t.evalues))]),label='tau', alpha=0.5, marker='+')
    plt.semilogy(np.abs(np.sort(ep_c.evalues.real)-true_evals[slice(0,len(ep_c.evalues))]),label='truncation', alpha=0.5, marker='+')

    plt.xlabel("number")
    plt.ylabel("eigenvalue error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evals_resolution_basis_tau.png", dpi=300)

    # tau_sort = np.argsort(ep_t.evalues_primary.real)
    # print(f"tau evals = {np.sqrt(ep_t.evalues_primary[tau_sort])/np.pi}")
    
    # print(f"tau errors= {ep_t.error[tau_sort]}")

    plt.clf()
    trunc_sort = np.argsort(ep_c.evalues.real)
    total_trunc_sort = np.argsort(ep_c.evalues_primary.real)
    print(ep_c.evalues_index[total_trunc_sort])
    plt.plot(np.abs(ep_c.evalues[trunc_sort].real-ep_c.evalues_primary[total_trunc_sort].real[:len(trunc_sort)]), label='kept modes', marker='o')
    # plt.semilogy(true_evals, label='true')
    # plt.semilogy(ep_c.evalues_primary[total_trunc_sort].real, label='all modes')
    # plt.legend()
    plt.xlim(0,60)
    plt.xlabel("sort order")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig("evalus_trunc_errors.png",dpi=300)

    index = total_trunc_sort[58]
    plot_mode(ep_c,index, ep_c.evalues_index[index])
