Eigentools
**********

Eigentools is a set of tools for studying linear eigenvalue problems. The underlying eigenproblems are solved using `Dedalus <http://dedalus-project.org>`_, which provides a domain-specific language for partial differential equations. Each entry in the following list of features links to a Jupyter notebook giving an example of its use.

* :ref:`automatic rejection of unresolved eigenvalues </notebooks/Orr Somerfeld pseudospectra.ipynb>`
* :ref:`simple plotting of specified eigenmodes </notebooks/Mixed Layer Instability.ipynb>`
* :ref:`simple plotting of spectra </notebooks/Orr Somerfeld pseudospectra.ipynb>`
* :ref:`computation of pseudospectra for any Differential-Algebraic Equations with user-specifiable norms </notebooks/Orr Somerfeld pseudospectra.ipynb>`
* :ref:`tools to find critical parameters for linear stability analysis </notebooks/Mixed Layer Instability.ipynb>`
* :ref:`ability to project eigenmode onto 2- or 3-D domain for visualization </notebooks/Mixed Layer Instability.ipynb>`
* :ref:`ability to output projected eigenmodes as Dedalus-formatted HDF5 file to be used as initial conditions for Initial Value Problems </notebooks/Mixed Layer Instability.ipynb>`
* :ref:`simple plotting of drift ratios (both ordinal and nearest) to evaluate tolerance for eigenvalue rejection </pages/getting_started.rst>`

Contents
========

.. toctree::
   :maxdepth: 2

   pages/installation
   pages/getting_started
   Eigentools API reference <autoapi/eigentools/index>

Example notebooks
-----------------

.. toctree::
   :maxdepth: 1

   Example 1: Orr-Somerfield pseudospectra </notebooks/Orr Somerfeld pseudospectra.ipynb>
   Example 2: Mixed Layer instability </notebooks/Mixed Layer Instability.ipynb>

Developers
==========
The core development team consists of 

* Jeff Oishi (<jsoishi@gmail.com>)
* Keaton Burns (<keaton.burns@gmail.com>)
* Susan Clark (<susanclark19@gmail.com>)
* Evan Anders (<evan.anders@northwestern.edu>)
* Ben Brown (<bpbrown@gmail.com>)
* Geoff Vasil (<geoffrey.m.vasil@gmail.com>)
* Daniel Lecoanet (<daniel.lecoanet@northwestern.edu>)

Support
=======
Eigentools was developed with support from the Research Corporation under award Scialog Collaborative Award (TDA) ID# 24231.

