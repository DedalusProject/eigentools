Installing eigentools
*********************

eigentools itself is `pip` installable::

  pip install eigentools

If you would like the development version, you can clone the repository and install locally::

  git clone https://github.com/DedalusProject/eigentools.git
  pip install -e eigentools

**Caution**: eigentools requires Dedalus, which can (and will) also be pip installed, but Dedalus relies on several non-python dependencies that *cannot* be pip installed.
These must be installed before pip installing eigentools or the process will fail.
You can install those dependencies following the procedures in `the Dedalus documentation <https://dedalus-project.readthedocs.io/en/latest/>`_.


