import pytest
import dedalus.public as de
from dedalus.tools.parallel import Sync
import eigentools as eig
import numpy as np
from mpi4py import MPI

def rbc_problem(problem_type, domain, stress_free=False, Ra = 1708., k=3.117, Pr = 1):
    problems = {'EVP': de.EVP, 'IVP': de.IVP}

    try:
        args = [domain,['p', 'b', 'u', 'w', 'bz', 'uz', 'wz']]
        if problem_type == 'EVP':
             args.append('omega')
        rayleigh_benard = problems[problem_type](*args)
    except KeyError:
        raise ValueError("problem_type must be one of 'EVP' or 'IVP', not {}".format(problem))

    rayleigh_benard.parameters['k'] = k #horizontal wavenumber
    rayleigh_benard.parameters['Ra'] = Ra #Rayleigh number, rigid-rigid is 1708
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
    if problem_type == 'IVP':
        rayleigh_benard.add_bc('right(w) = 0', condition='(nx != 0)')
        rayleigh_benard.add_bc('right(p) = 0', condition='(nx == 0)')
    else:
        rayleigh_benard.add_bc('right(w) = 0')
    if stress_free:
        rayleigh_benard.add_bc('left(uz) = 0')
        rayleigh_benard.add_bc('right(uz) = 0')
    else:
        rayleigh_benard.add_bc('left(u) = 0')
        rayleigh_benard.add_bc('right(u) = 0')

    return rayleigh_benard

if __name__=="__main__":
    import time
    from dedalus.extras import flow_tools
    from dedalus.extras import plot_tools
    import logging
    logger = logging.getLogger(__name__.split('.')[-1])

    comm = MPI.COMM_WORLD
    kc = 3.117  #horizontal wavenumber
    Rac = 1708. #Rayleigh number, rigid-rigid is 1708
    Lx = 2*np.pi/kc
    Ra = 1e6
    with Sync(comm) as sync:
        if sync.comm.rank == 0:
            z_evp = de.Chebyshev('z',32, interval=(0, 1))
            d_evp = de.Domain([z_evp],comm=MPI.COMM_SELF)
            rb_evp = rbc_problem('EVP',d_evp)
            EP = eig.Eigenproblem(rb_evp)

            x = de.Fourier('x', 32, interval = (0, Lx)) 
            z = de.Chebyshev('z',32, interval=(0, 1))
            output_domain = de.Domain([x,z],grid_dtype=np.float64,comm=MPI.COMM_SELF)

            logger.info("projecting mode at Ra = {}, k = {}".format(Ra,kc))
            growth, index, freq = EP.growth_rate(parameters={'k':kc,'Ra':Ra},sparse=False)

            logger.info("growth {}, index {}, freq {}".format(growth, index, freq))
            fields = EP.project_mode(index, output_domain, [1,])

            EP.write_global_domain(fields)
            logger.info("Done!")

    logger.info("setting up IVP")
    x = de.Fourier('x', 512, interval=(0,Lx))
    z = de.Chebyshev('z',64, interval=(0, 1))
    ivp_domain = de.Domain([x,z],grid_dtype=np.float64)

    rb_IVP = rbc_problem('IVP', ivp_domain, Ra=Ra)
    solver =  rb_IVP.build_solver(de.timesteppers.RK222)
    
    # load intial conditions from most critical mode
    logger.info("loading most critical mode")
    solver.load_state("IVP_output/IVP_output_s1.h5",-1)

    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=2.5e-4, max_writes=50)
    snapshots.add_system(solver.state)
    timeseries = solver.evaluator.add_file_handler('timeseries', sim_dt=2.5e-4, max_writes=np.inf)
    timeseries.add_task('sqrt(integ(b**2))',name='b_rms')

    # CFL
    CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=1, safety=0.5,
                         max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
    flow.add_property("sqrt(u*u + w*w) / Pr", name='Re')

    # Main loop
    try:
        logger.info('Starting loop')
        start_time = time.time()
        while solver.proceed:
            dt = CFL.compute_dt()
            dt = solver.step(dt)
            if (solver.iteration-1) % 10 == 0:
                logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
                logger.info('Max Re = %e' %flow.max('Re'))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()
        logger.info('Iterations: %i' %solver.iteration)
        logger.info('Sim end time: %f' %solver.sim_time)
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*ivp_domain.dist.comm_cart.size))
