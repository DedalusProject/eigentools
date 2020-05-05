"""test problem for eigentools for non-constant coefficients:

problem from equation 26 of 

Huang, Chen, and Luo, Applied Mathematics Letters (2013)
https://www.sciencedirect.com/science/article/pii/S0893965913000748

y''''(x) - 0.02 x^2 y'' - 0.04 x y' + (0.0001 x^4 - 0.02) y = lambda y

with boundary conditions

y(0) = y(5) = y'(0) = y'(5) = 0

this is their Case 1

NB: I corrected a typo 

Table 3 from that paper gives
  0.86690250239956
  6.35768644786998
 23.99274694653769
 64.97869559403952
144.2841396045761

NB: THIS IS NOT A HIGH PRECISION TEST! It's unclear from the reference what the "true" values actually are. We agree much more closely with their reference 14, but I'm not sure if that is a more trustworthy calculation anyway.

"""
import pytest
import numpy as np
import dedalus.public as de
from eigentools import Eigenproblem, CriticalFinder

@pytest.mark.parametrize('Nx', [50])
@pytest.mark.parametrize('sparse', [False])
def test_non_constant(Nx, sparse):
    x = de.Chebyshev('x',Nx,interval=(0,5))
    d = de.Domain([x,])

    prob = de.EVP(d,['y','yx','yxx','yxxx'],'sigma')

    prob.add_equation("dx(yxxx) -0.02*x*x*yxx -0.04*x*yx + (0.0001*x*x*x*x - 0.02)*y - sigma*y = 0")
    prob.add_equation("dx(yxx) - yxxx = 0")
    prob.add_equation("dx(yx) - yxx = 0")
    prob.add_equation("dx(y) - yx = 0")

    prob.add_bc("left(y) = 0")
    prob.add_bc("right(y) = 0")
    prob.add_bc("left(yx) = 0")
    prob.add_bc("right(yx) = 0")

    EP = Eigenproblem(prob, sparse=sparse)

    EP.solve()
    indx = EP.evalues_good.argsort()

    five_evals = EP.evalues_good[indx][0:5]
    print("First five good eigenvalues are: ")
    print(five_evals)
    print(five_evals[-1])

    reference = np.array([0.86690250239956+0j, 6.35768644786998+0j, 23.99274694653769+0j, 64.97869559403952+0j, 144.2841396045761+0j])

    assert np.allclose(reference, five_evals,rtol=1e-4)
