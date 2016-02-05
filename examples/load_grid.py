"""finds the critical Renoylds number and wave number for the
Orr-Somerfeld eigenvalue equation.

"""
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import h5py

def fake(x,y):
    return x,y

comm = MPI.COMM_WORLD

cf = CriticalFinder(fake, comm)
cf.load_grid("orr_sommerfeld_growth_rates.h5")

cf.root_finder()
crit = cf.crit_finder()

if comm.rank == 0:
    print("critical wavenumber alpha = {:10.5f}".format(crit[0]))
    print("critical Re = {:10.5f}".format(crit[1]))

    cf.plot_crit()

