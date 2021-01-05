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
---
# Summary
In numerous fields of science, engineering, and applied mathematics, eigenvalue analysis is an invaluable tool. It is used to define bases in quantum mechanics, to find orbitals in chemistry, assess the stability of vibrations in fluids and solids, to understand the stability and robustness of timestepping schemes, to understand material properties in the Earth and Sun via seismological techniques, among a plethora of other uses. Concomitantly,  nearly every computational package contains tools for computing eigenvalues for both sparse and dense matrices. However, studying these eigenvalues is not without significant peril: many important systems, particularly discretized partial differential equations (PDEs), are poorly conditioned and not all the numerical eigenvalues reported by such routines are reliable. Additionally, it is far from trivial to start with a set of PDEs and *construct* a discretized matrix in the first place. In order to solve these problems, we present `eigentools`, a Python package that extends the eigenvalue problem (EVP) capabilities of the Dedalus project [@PhysRevResearch.2.023068] to provide a complete analysis toolkit for EVPs.

# Statement of need
Linear stability analysis of PDEs is a fundamental tool in chaotic dynamics, fluid dynamics, biophysics, and many other scientific disciplines. `eigentools` provides a convenient, parallelized interface for both modal and non-modal stability analyses for nearly arbitrary systems of PDEs.

In addition to the traditional venues for eigenvalue analysis such as fluid and solid mechanics, a wide variety of new continuum models is emerging from soft condensed matter, particularly the study of active matter [see @doi:10.1146/annurev-conmatphys-031119-050611; and @2020NatRP.2.181S for recent reviews]. These models are encoded as PDEs and evaluating their stability is important for understanding the rich behavior they exhibit. `eigentools` provides a toolkit that requires very little user input in order to take a model, find robust eigenvalues and eigemodes, and find critical parameter values for stability. The only thing a user needs to do is find a state to linearize about and cast the PDE in that form. Once the linear PDE is derived, one constructs a Dedalus `EigenvalueProblem` object, and passes that to `eigentools`. `eigentools` provides robust spurious eigenvalue rejection [@boyd2001chebyshev], spectrum and eigenmode visualization, $\epsilon-$pseudospectra, and the ability to project a given eigenmode onto an 2- or 3-dimensional domain and save it as a Dedalus-formatted HDF5 file to use as an initial condition for an initial value problem (i.e. simulation) of the same system. 

# Critical Parameter finding
One of the original motivations for `eigentools` was to quickly and easily find critical parameters for eigenvalue stability problems.
In order to do so, `eigentools` provides an object `CriticalFinder`, which allows users to specify an `Eigenproblem` and a tuple of parameters.
The user then provides a grid of points for those two parameters, and `CriticalFinder` finds the maximum growth rate for the EVP at each point in the grid, exploiting MPI parallelism on multiprocessor systems.
It then interpolates to find the zero crossings of one parameter, and finally minimizes over the remaining parameter to find approximate critical values.
`CriticalFinder` also provides simple visualization tools, root polishing algorithms to further refine the critical values, and the ability to save and load the grid of eigenvalues.
`CriticalFinder` allows the user to choose how "growth" is defined via custom functions; this allows the use of positive imaginary parts (e.g. $e^{\sigma t}$, $\sigma \in \mathbb{C}$), negative real parts (e.g. $e^{i(k x - \omega t)}$, $\omega \in \mathbb{C}$), or any other choice.

\autoref{fig:mri} demonstrates three core features of `eigentools`: the ability to find critical parameters, the ability to use sparse and dense eigenvalue solvers, and the ability to reject spurious eigenvalues.
In the right panel, the growth rate of the magnetorotational instability (defined as the positive real part of the eigenvalue $\sigma$) is plotted on a $20 \times 20$ grid of magnetic Reynolds number $\mathrm{Rm}$ and wavenumber $Q$, finding the critical values $\mathrm{Rm_c} = 4.88, Q = 0.747$; in \autoref{fig:mri}, we used 4 cores each performing 100 *sparse* eigenvalue solves finding the 15 modes with $\sigma$ closest to zero.
The middle panel shows the spectrum at the critical parameters; this was solved using a *dense* eigenvalue solver to find all modes.
The unstable mode is a rotationally modified Alfv\'en wave highlighted in red.
Finally, the rightmost panel shows a plot of the **inverse drift ratio** for both ordinal and nearest comparisons.
When `eigentools` solves an EVP, by default it will perform mode rejection by solving the same problem twice, once at 1.5 times the resolution (this is user configurable).
In order to ascertain which modes are good, the inverse drift ratio is computed one of two ways.
For simple problems with only one mode family, the *ordinal* method in which the eigenvalues are compared in sorted order.
However, the magnetorotational instability has *multiple wave families*.
By increasing the resolution, the number of resolved modes for each family increases; because of this, one must compare the drift ratios of the *nearest* eigenvalue [for details, see @boyd2001chebyshev].

![Magnetorotational instability. From left to right: growth rates in the $\mathrm{Rm}-Q$, Black line and circles show zero-growth contour. The MRI spectrum at the critical parameters. Inverse drift ratios for modes shown in the spectrum. Those below $10^6$ are rejected according to nearest (blue) and ordinal (orange) criteria.\label{fig:mri}](mri.png)

# Output and creation of initial conditions

![Rayleigh-Benard convection. From left to right: buoyancy (colormap) and velocities (arrows) for the most unstable eigenmode at $\mathrm{Ra} = 10^5$, buoyancy and velocites for the non-linear steady state for that eigenmode after evolution via an initial value problem in Dedalus, time evolution of RMS buoyancy.\label{fig:rbc}](rbc_evp_ivp.png)

\autoref{fig:rbc} highlights `eigentools` output capability. We solve the EVP at $\mathrm{Ra} = 10^5$ for Rayleigh-Benard convection between two no-slip plates using `eigentools` at a resolution of $n_z = 16$. 
We then output it on a 2-D domain of $(n_x, n_z) = (16,64)$ and load that into a Dedalus initial value problem (IVP) solver using the full, non-linear equations for Rayleigh-Benard convection.
Using Dedalus's ability to change parameters and resolutions on the fly, we then run IVP at with a resolution of $(512, 64)$ until it reaches a non-linear steady state.
In the left panel of \autoref{fig:rbc}, we see excellent agreement between the growth rate from the non-linear IVP and the initial eigenvalue until non-linearity begins to become important around $t\approx 0.02$.

# Pseudospectra
One of the most important features of `eigentools` is the ability to calculate $\epsilon-$pseudospectra, a generalization of the spectrum that reveals non-normality of the underlying matrix [@trefethen2005spectra].
These can be used for non-modal stability analysis in fluid dynamics, and may also play a role in the unusual properties of $\mathcal{PT}-$symmetric quantum mechanics [@doi:10.1063/1.4934378]. The $\epsilon-$pseudospectra was originally developed for standard eigenvalue problems, 

$$\mathbf{L} \mathbf{x} = \lambda \mathbf{x}.$$

We have implemented a newly developed algorithm [@doi:10.1137/15M1055012] to allow the computation of pseudospectra for differential-algebraic equation (DAE) systems.
To our knowledge, this is the first publicly available system for computing $\epsilon-$pseudospectra for arbitrary DAEs.

![Spectrum, pseudospectrum, and four representative eigenmodes for the Orr-Sommerfeld problem, expressed in primitive variables $(u,v)$. The eigenmodes correspond to the eigenvalues highlighted in orange in the middle panel. Pseudospectrum contours are labeled by n, representing $10^{n}$.\label{fig:os_pseudo}](pseudospectra.png)

\autoref{fig:os_pseudo} shows an example pseudospectrum, its corresponding spectrum, and four representative eigenvectors for the classic Orr-Sommerfeld problem in hydrodynamic stability theory. 
As a twist on the standard problem, we demonstrate Dedalus and `eigentools` ability to solve the problem using the standard Navier-Stokes equations linearized about a background velocity, rather than in the traditional, single fourth-order equation for wall-normal velocity. This is not possible without using the generalized eigenvalue pseudospectra algorithm implemented above.
Note that for the four eigenvectors, we plot $u$ and $w$, the streamwise and wall-normal directions, respectively, rather than $w$ and $\eta$, the vorticity as would be the case in the reduced Orr-Sommerfeld form. The solid and dashed lines represent the real and imaginary parts of the eigenvectors, respectively.

# Example
Here we present an script that computes the spectra and pseudospectra for the classic Orr-Sommerfeld problem. 
The script produces a simplified version of the center plot in \autoref{fig:os_pseudo}. 
The first block of code sets up the Navier-Stokes equations in Dedalus, making use of its expressive substitution mechanism.

```python
import matplotlib.pyplot as plt
from eigentools import Eigenproblem
import dedalus.public as de
import numpy as np

# problem parameters
alpha = 1.
Re = 10000

# define Navier-Stokes equations in Dedalus
z = de.Chebyshev('z', 128)
d = de.Domain([z])
os = de.EVP(d,['u','w','uz','wz', 'p'],'c')

os.parameters['alpha'] = alpha
os.parameters['Re'] = Re
os.substitutions['umean'] = '1 - z**2'
os.substitutions['umeanz'] = '-2*z'
os.substitutions['dx(A)'] = '1j*alpha*A' 
os.substitutions['dt(A)'] = '-1j*alpha*c*A'
os.substitutions['Lap(A,Az)'] = 'dx(dx(A)) + dz(Az)'
os.add_equation('dt(u) + umean*dx(u) + w*umeanz + dx(p) - Lap(u, uz)/Re = 0')
os.add_equation('dt(w) + umean*dx(w) + dz(p) - Lap(w, wz)/Re = 0')
os.add_equation('dx(u) + wz = 0')
os.add_equation('uz - dz(u) = 0')
os.add_equation('wz - dz(w) = 0')
os.add_bc('left(w) = 0')
os.add_bc('right(w) = 0')
os.add_bc('left(u) = 0')
os.add_bc('right(u) = 0')

os_EP = Eigenproblem(os) 

# define the energy norm 
def energy_norm(Q1, Q2):
    u1 = Q1['u']
    w1 = Q1['w']
    u2 = Q2['u']
    w2 = Q2['w']
    
    field = (np.conj(u1)*u2 + np.conj(w1)*w2).evaluate().integrate()
    return field['g'][0]
    
# Calculate pseudospectrum
k = 100 # size of invariant subspace
psize = 100 # number of points in real, imaginary points
real_points = np.linspace(0,1, psize)
imag_points = np.linspace(-1,0.1, psize)
os_EP.calc_ps(k, (real_points, imag_points), inner_product=energy_norm)

# plot
P_CS = plt.contour(os_EP.ps_real,os_EP.ps_imag, np.log10(os_EP.pseudospectrum),levels=np.arange(-8,0),linestyles='solid')
plt.scatter(os_EP.evalues.real, os_EP.evalues.imag,color='blue',marker='x')
plt.xlim(0,1)
plt.ylim(-1,0.1)
plt.xlabel(r"$c_r$")
plt.ylabel(r"$c_i$")
plt.tight_layout()
plt.savefig("OS_pseudospectra.png", dpi=300)
```

# Related Work
There are a few other packages dedicated to the automatic construction of eigenvalue problems, including [Chebfun](https://www.chebfun.org/), which can also produce pseudospectra. Chebfun, while itself released under the standard 3-Clause BSD license, is written in the proprietary MATLAB langauge.
For computing spectra and pseudospectra for existing matrices, the venerable [`EigTool`](https://github.com/eigtool/eigtool) package is another open-source option again writtein in MATLAB.
It does not feature parallelism nor the ability to construct eigenvalue problems.
`EigTool` has also been ported to the open-source Julia language in the [`Pseudospectra.jl`](https://github.com/RalphAS/Pseudospectra.jl) package.


<!-- # Performace -->

<!-- Using Orr-Somerfeld test problem  -->

<!-- OMP parallelization leavitt -->
<!-- real	1m9.802s -->
<!-- user	10m13.894s -->
<!-- sys	19m24.646s -->

<!-- Single core no OMP leavitt -->
<!-- real	0m48.659s -->
<!-- user	0m43.606s -->
<!-- sys	0m0.947s -->

<!-- 4 core MPI no OMP leavitt -->
<!-- real	0m28.251s -->
<!-- user	0m52.939s -->
<!-- sys	0m4.548s -->

<!-- openblas OMP 8 threads (default OMP_NUM_THREADS) -->
<!-- 2020-12-07 14:43:18,278 __main__ 0/1 INFO :: grid generation time:   68.02126 sec -->

<!-- openblas OMP 4 threads (default OMP_NUM_THREADS) -->
<!-- 2020-12-07 14:45:15,796 __main__ 0/1 INFO :: grid generation time:   34.12971 sec -->

<!-- openblas OMP 1 threads (default OMP_NUM_THREADS) -->
<!-- 2020-12-07 14:46:19,026 __main__ 0/1 INFO :: grid generation time:   29.15382 sec -->

<!-- MKL 1 threads (default OMP_NUM_THREADS) -->
<!-- 2020-12-07 14:41:32,328 __main__ 0/1 INFO :: grid generation time:   30.47119 sec -->

<!-- MKL dense -->
<!-- 2020-12-07 14:50:12,748 __main__ 0/1 INFO :: grid generation time:  148.12831 sec -->



<!-- # Mathematics -->

<!-- Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$ -->

<!-- Double dollars make self-standing equations: -->

<!-- $$\Theta(x) = \left\{\begin{array}{l} -->
<!-- 0\textrm{ if } x < 0\cr -->
<!-- 1\textrm{ else} -->
<!-- \end{array}\right.$$ -->

<!-- You can also use plain \LaTeX for equations -->
<!-- \begin{equation}\label{eq:fourier} -->
<!-- \hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx -->
<!-- \end{equation} -->
<!-- and refer to \autoref{eq:fourier} from text. -->

`eigentools` has been used in several papers including @2017ApJ.841.1C; @2017ApJ.841.2C; @2020RSPSA.47690622O; @PhysRevResearch.2.023068; and @2020arXiv201112300L.

# Acknowledgements
Eigentools was developed with support from the Research Corporation under award Scialog Collaborative Award (TDA) ID# 24231.

# References
