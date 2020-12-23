---
title: 'eigentools: A Python package for studying eigenvalue problems with an emphasis on stability'
tags:
  - eigenvalue problems
  - partial differential equations
  - fluid dynamics
  - magnetohydrodynamics
  - pseudospectra
  - Python
authors:
  - name: Jeffrey S. Oishi^[Corresponding Author]
    orcid: 0000-0001-8531-6570
    affiliation: 1
  - name: Keaton J. Burns
    affiliation: 2
  - name: S. E. Clark
    affiliation: 3
  - name: Evan H. Anders
    affiliation: 4
  - name: Benjamin P. Brown
    affiliation: 5
  - name: Geoffrey M. Vasil
    affiliation: 6
  - name: Daniel Lecoanet
    affiliation: "4, 7"
affiliations:
 - name: Department of Physics and Astronomy, Bates College
   index: 1
 - name: Department of Mathematics, MIT
   index: 2
 - name: Department of Physics, Stanford University
   index: 3
 - name: CIERA, Northwestern University
   index: 4
 - name: Department of Astrophysical and Planetary Sciences, University of Colorado, Boulder
   index: 5
 - name: School of Mathematics and Statistics, University of Sydney
   index: 6
 - name: Department Engineering Sciences and Applied Mathematics, Northwestern University
   index: 7

date: 21 December 2020
bibliography: paper.bib

# Summary
In numerous fields of science, engineering, and applied mathematics, eigenvalue analysis is an invaluable tool. It is used to define bases in quantum mechanics, to find orbitals in chemistry, assess the stability of vibrations in fluids and solids, to understand the stability and robustness of timestepping schemes, to understand material properties in the Earth and Sun via seismological techniques, among a plethora of other uses. Concomitantly,  nearly every computational package contains tools for computing eigenvalues for both sparse and dense matrices. However, studying these eigenvalues is not without significant peril: many important systems, particularly discretized partial differential equations (PDEs), are poorly conditioned and not all the numerical eigenvalues reported by such routines are reliable. Additionally, it is far from trivial to start with a set of PDEs and *construct* a discretized matrix in the first place. In order to solve these problems, we present `eigentools`, a Python package that extends the eigenvalue problem capabilities of the Dedalus project [@Dedalus] to provide a complete analysis toolkit for eigenvalue problems.

# Statement of need
Linear stability analysis of PDEs is a fundamental tool in chaotic dynamics, fluid dynamics, biophysics, and many other scientific disciplines. `eigentools` provides a convenient, parallelized interface for both modal and non-modal stability analyses for nearly arbitrary systems of PDEs.

In addition to the traditional venues for eigenvalue analysis such as fluid and solid mechanics, a wide variety of new continuum models is emerging from soft condensed matter, particularly the study of active matter [see @doi:10.1146/annurev-conmatphys-031119-050611 and @2020NatRP...2..181S for recent reviews]. These models are encoded as PDEs and evaluating their stability is important for understanding the rich behavior they exhibit. `eigentools` provides a toolkit that requires very little user input in order to take a model, find robust eigenvalues and eigemodes, and find critical parameter values for stability. The only thing a user needs to do is find a state to linearize about, and cast the PDE in that form. Once the linear PDE is derived, one constructs a Dedalus `EigenvalueProblem` object, and passes that to `eigentools`. `eigentools` provides robust spurious eigenvalue rejection, spectrum and eigenmode visualization, and the ability to project a given eigenmode onto an 2- or 3-dimensional domain and save it as a Dedalus-formatted HDF5 file to use as an initial condition for an initial value problem (i.e. simulation) of the same system. 

# Pseudospectra

One of the most important features of `eigentools` is the ability to calculate $\epsilon-$pseudospectra, a generalization of the spectrum that reveals non-normality of the underlying matrix [@trefethen2005spectra].
These can be used for non-modal stability analysis in fluid dynamics, and may also play a role in the unusual properties of $\mathcal{PT}-$symmetric quantum mechanics [@doi:10.1063/1.4934378]. The $\epsilon-$pseudospectra was originally developed for standard eigenvalue problems, 

$$\mathbf{L} \mathbf{x} = \lambda \mathbf{x}$$.

Using a newly developed algorithm [@doi:10.1137/15M1055012] to allow the computation of pseudospectra for differential-algebraic equation systems, we implement (for the first time, to our knowledge) 

# Example
As an example of many of the features of `eigentools`, here is a script that computes the spectra and pseudospectra for the classic Orr-Sommerfeld problem in hydrodynamic stability theory. As a twist on the standard problem, we demonstrate Dedalus and `eigentools` ability to solve the problem using the standard Navier-Stokes equations linearized about a background velocity, rather than in the traditional, single fourth-order equation for wall-normal velocity. This is not possible without using the generalized eigenvalue pseudospectra algorithm implemented above.

```python
import matplotlib.pyplot as plt
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import dedalus.public as de
import numpy as np

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
P_CS = plt.contour(prim_EP.ps_real,prim_EP.ps_imag, np.log10(prim_EP.pseudospectrum),levels=np.arange(-8,0),linestyles='solid')
plt.scatter(EP.evalues.real, EP.evalues.imag,color='blue',marker='x',zorder=4)

plt.axis('square')
plt.xlim(0,1)
plt.ylim(-1,0.1)
plt.axhline(0,color='k',alpha=0.2)
plt.xlabel(r"$c_r$")
plt.ylabel(r"$c_i$")
plt.tight_layout()
plt.savefig("OS_pseudospectra.png", dpi=300)
```

# Related Work
There are a few other packages dedicated to the automatic construction of eigenvalue problems, including [Chebfun](https://www.chebfun.org/), which can also produce pseudospectra. Chebfun, while itself released under the standard 3-Clause BSD license, is written in the proprietary MATLAB langauge.
[`Pseudospectra.jl`](https://github.com/RalphAS/Pseudospectra.jl) and [`EigTool`](https://github.com/eigtool/eigtool) and [`PSOPT`](http://www.psopt.org/) and 


# Performace

Using Orr-Somerfeld test problem 

OMP parallelization leavitt
real	1m9.802s
user	10m13.894s
sys	19m24.646s

Single core no OMP leavitt
real	0m48.659s
user	0m43.606s
sys	0m0.947s

4 core MPI no OMP leavitt
real	0m28.251s
user	0m52.939s
sys	0m4.548s

openblas OMP 8 threads (default OMP_NUM_THREADS)
2020-12-07 14:43:18,278 __main__ 0/1 INFO :: grid generation time:   68.02126 sec

openblas OMP 4 threads (default OMP_NUM_THREADS)
2020-12-07 14:45:15,796 __main__ 0/1 INFO :: grid generation time:   34.12971 sec

openblas OMP 1 threads (default OMP_NUM_THREADS)
2020-12-07 14:46:19,026 __main__ 0/1 INFO :: grid generation time:   29.15382 sec

MKL 1 threads (default OMP_NUM_THREADS)
2020-12-07 14:41:32,328 __main__ 0/1 INFO :: grid generation time:   30.47119 sec

MKL dense
2020-12-07 14:50:12,748 __main__ 0/1 INFO :: grid generation time:  148.12831 sec



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

# Acknowledgements
Eigentools was developed with support from the Research Corporation under award Scialog Collaborative Award (TDA) ID# 24231.

# References
