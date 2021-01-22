# Eigentools

Eigentools is a set of tools for studying linear eigenvalue problems. The underlying eigenproblems are solved using [Dedalus](http://dedalus-project.org), which provides a domain-specific language for partial differential equations. Eigentools extends Dedalus's `EigenvalueProblem` object and provides

* automatic rejection of unresolved eigenvalues
* simple plotting of specified eigenmodes
* simple plotting of spectra
* computation of $\epsilon$-pseudospectra for any Differential-Algebraic Equations with **user-specifiable norms**
* tools to find critical parameters for linear stability analysis
* ability to project eigenmode onto 2- or 3-D domain for visualization
* ability to output projected eigenmodes as Dedalus-formatted HDF5 file to be used as initial conditions for Initial Value Problems
* simple plotting of drift ratios (both ordinal and nearest) to evaluate tolerance for eigenvalue rejection

## Installation

Eigentools requires [Dedalus](http://dedalus-project.org), which can be installed along with all of its dependencies using conda via [its install script](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html#conda-installation-recommended). 

Once Dedalus is installed, eigentools can be pip installed in the same conda environment via

```
pip install eigentools
```

## Documentation

API Documentation can be found at [Read the Docs](https://eigentools.readthedocs.io/).

## Developers
The core development team consists of 

* Jeff Oishi (<jsoishi@gmail.com>)
* Keaton Burns (<keaton.burns@gmail.com>)
* Susan Clark (<susanclark19@gmail.com>)
* Evan Anders (<evan.anders@northwestern.edu>)
* Ben Brown (<bpbrown@gmail.com>)
* Geoff Vasil (<geoffrey.m.vasil@gmail.com>)
* Daniel Lecoanet (<daniel.lecoanet@northwestern.edu>)

## Support 
Eigentools was developed with support from the Research Corporation under award Scialog Collaborative Award (TDA) ID# 24231.


<!--  LocalWords:  Eigentools eigenproblems Dedalus EigenvalueProblem
 -->
<!--  LocalWords:  eigenmodes pseudospectra eigenmode HDF conda Oishi
 -->
<!--  LocalWords:  eigentools Anders Geoff Vasil Lecoanet Scialog TDA
 -->
