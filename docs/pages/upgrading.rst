Upgrading eigentools scripts
****************************

Version 2 of eigentools has made significant changes to the API and will necessitate some changes (for the better, we hope) to the user experience. The guiding principle behind the new API is that one should no longer need to touch the Dedalus :code:`EVP` object that defines the eigenvalue problem at hand. 

**Most importantly**, no changes need to be made to the underlying Dedalus :code:`EVP` object.

Basic :code:`eigenproblem` usage
--------------------------------

Choosing a sparse or dense solve is no longer done when instantiating :code:`Eigenproblem` objects. Instead, this is a choice at *solve* time:

.. code-block:: python
                
    EP = Eigenproblem(string, reject=True)
    EP.solve(sparse=False)

Also, notice that rejection of spurious modes is now done automatically with :code:`EP.solve` if :code:`reject=True` is selected at instantiation time. Note that although in the above code, we explicitly set :code:`reject=True`, this is **unnecessary**, as it is the default. The :code:`EP.reject_spurious()` function has been removed

In addition, solving again with different parameters has been greatly simplified from the previous version. You now simply *pass a dictionary* with the parameters you wish to change to solve itself. Let's revisit the simple waves-on-a-string problem from :ref:`the getting started page </pages/getting_started.rst>`,  but add a parameter, :code:`c2`, the wave speed squared.

Here, we solve twice, once with :code:`c1 = 1` and once with :code:`c2 = 2`. Given the dispersion relation for this problem is :math:`\omega^2 = c^2 k` and our eigenvalue :code:`omega` is really :math:`\omega^2`, we expect the eigenvalues for the second solve to be twice those for the first.

.. code-block:: python
                
    import numpy as np
    from eigentools import Eigenproblem
    import dedalus.public as de
    
    Nx = 128
    x = de.Chebyshev('x',Nx, interval=(-1, 1))
    d = de.Domain([x])
    
    string = de.EVP(d, ['u','u_x'], eigenvalue='omega')
    string.parameters['c2'] = 1
    string.add_equation("omega*u + c2*dx(u_x) = 0")
    string.add_equation("u_x - dx(u) = 0")
    string.add_bc("left(u) = 0")
    string.add_bc("right(u) = 0")
    EP = Eigenproblem(string)
    EP.solve(sparse=False)
    evals_c1 = EP.evalues
    EP.solve(sparse=False, parameters={'c2':2})
    evals_c2 = EP.evalues
    
    print(np.allclose(evals_c2, 2*evals_c1))

Getting eigenmodes
==================

Getting eigenmodes has also been simplified and significantly extended. Previously, getting an eigenmode corresponding to an eigenvalue required using the :code:`set_state()` method on the underlying :code:`EVP` object. In keeping with the principle of not needing to manipulate the :code:`EVP`, we provide a new :code:`.eigenmode(index)`, where :code:`index` is the mode number corresponding to the eigenvalue index in :code:`EP.evalues`. By default, with mode rejection on, these are the "good" eigenmodes.
`.eigenmode(index)` returns a Dedalus :code:`FieldSystem` object, with a Dedalus :code:`Field` for each field in the eigenmode:

.. code-block:: python
                
    emode = EP.eigenmode(0)
    print([f.name for f in emode.fields])
    u = emode.fields[0]
    u_x = emode.fields[1]


Finding critical parameters
---------------------------

This has been considerably cleaned up. The two major things to note are that

1. one no longer needs to create a shim function to translate between an x-y grid and the parameter names within the :code:`EVP`.
2. The parameter grid is no longer defined inside :code:`CriticalFinder`, but is instead created by the user and passed in

For example, here are the relevant changes necessary for the `MRI test problem <https://github.com/DedalusProject/eigentools/tree/master/examples/mri.py>`_.

First, replace

.. code-block:: python
                
    EP = Eigenproblem(mri, sparse=True)
    
    # create a shim function to translate (x, y) to the parameters for the eigenvalue problem:
    def shim(x,y):
        iRm = 1/x
        iRe = (iRm*Pm)
        print("Rm = {}; Re = {}; Pm = {}".format(1/iRm, 1/iRe, Pm))
        gr, indx, freq = EP.growth_rate({"Q":y,"iRm":iRm,"iR":iRe})
        ret = gr+1j*freq
        return ret
     
    cf = CriticalFinder(shim, comm)

with

.. code-block:: python
   
    EP = Eigenproblem(mri)

    cf = CriticalFinder(EP, ("Q", "Rm"), comm, find_freq=False)

**Important:** note that :code:`find_freq` is specified at instantiation rather than when calling :code:`cf.crit_finder` later.

Once this is done, the grid generation changes from

.. code-block:: python
                
    mins = np.array((4.6, 0.5))
    maxs = np.array((5.5, 1.5))
    ns   = np.array((10,10))
    logs = np.array((False, False))
    
    cf.grid_generator(mins, maxs, ns, logs=logs)

to

.. code-block:: python
                
    nx = 20
    ny = 20
    xpoints = np.linspace(0.5, 1.5, nx)
    ypoints = np.linspace(4.6, 5.5, ny)
    
    cf.grid_generator((xpoints, ypoints), sparse=True)


