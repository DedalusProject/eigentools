"""test problem for eigentools for non-constant coefficients:

problem from equation 26 of 

Huang, Chen, and Luo, Applied Mathematics Letters (2013)
https://www.sciencedirect.com/science/article/pii/S0893965913000748

y''''(x) - 0.02 x^2 y'' - 0.04 x y' + (0.0001 x^4 - 0.02) y = lambda y

with boundary conditions

y(0) = y(5) = y'(0) = y'(5) = 0


NB: I corrected a typo 


"""
import dedalus.public as de
from eigentools import Eigenproblem, CriticalFinder

x = de.Chebyshev('x',50,interval=(0,5))
d = de.Domain([x,])

prob = de.EVP(d,['y','yx','yxx','yxxx'],'sigma')

c3 = d.new_field(name='c3')
c1 = d.new_field(name='c1')
c01 = d.new_field(name='c01')
xx = x.grid()

# this is not a reasonable way to do this; it's just to test non-constant coefficients
c3['g'] = -0.02 * xx**2
c1['g'] = -0.04 * xx
c01['g'] = 0.0001 * xx**4
prob.parameters['c3'] = c3
prob.parameters['c1'] = c1
prob.parameters['c01'] = c01
prob.parameters['c02'] = -0.02

prob.add_equation("dx(yxxx) + c3*yxx + c1*yx + (c01 + c02)*y - sigma*y = 0")
prob.add_equation("dx(yxx) - yxxx = 0")
prob.add_equation("dx(yx) - yxx = 0")
prob.add_equation("dx(y) - yx = 0")

prob.add_bc("left(y) = 0")
prob.add_bc("right(y) = 0")
prob.add_bc("left(yx) = 0")
prob.add_bc("right(yx) = 0")

EP = Eigenproblem(prob, sparse=False)

EP.solve()
EP.reject_spurious()
indx = EP.evalues_good.argsort()

print("First five good eigenvalues are: ")
print(EP.evalues_good[indx][0:5])

