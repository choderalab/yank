.. _installation:

############
Installation
############

***********************
Installing via `conda`
***********************

The simplest way to install YANK is via the `conda <http://www.continuum.io/blog/conda>`_  package manager.
Packages are provided on the `omnia binstar channel <http://binstar.org/omnia>`_ for Linux, OS X, and Win platforms.
The `yank binstar page <https://binstar.org/omnia/yank>`_ has useful instructions and download statistics.

If you are using a version of the `anaconda <https://store.continuum.io/cshop/anaconda/>` scientific Python distribution, you already have the `conda` package manager installed.
If not, but you have the `pip` package manager installed (to access packages from PyPI), you can easily install `conda` with

  $ pip install conda

Release build
=============

You can easily install the stable release build of YANK via the `conda` package with

  $ conda config --add channels http://conda.binstar.org/omnia
  $ conda install yank

This version is recommended for all users not actively developing new algorithms for alchemical free energy calculations.

.. note:: ``conda`` will automatically dependencies from binary packages automatically, including difficult-to-install packages such as numpy and scipy.
This is really the easiest way to get started.

Development build
=================

The bleeding-edge, absolute latest, very likely unstable development build of YANK is pushed to `binstar <https://binstar.org/omnia/yank>`_ with each GitHub commit, and can be obtained by

  $ conda config --add channels http://conda.binstar.org/omnia
  $ conda install yank-dev

Again, this build may very likely be unstable, so use at your own risk!

Testing the installation
========================

Test your YANK installation to make sure everything is behaving properly on your machine:

  $ yank selftest

This will not only check that installation paths are correct, but also run a battery of tests that ensure any automatically detected GPU hardware is behaving as expected.

************************************
Supported platforms and environments
************************************

Software
========

YANK runs on Python 2.7.
The developers generally use Python 2.7, on both Mac and Linux platforms.
Automated tests on Linux are performed on every incremental update to the code, and release tests are performed on Mac and Linux platforms.

Dependencies
------------

YANK uses a number of tools in order to allow the developers to focus on the algorithms involved in alchemical free energy calculations.
Installation of these prerequisites by hand is not recommended---all required dependencies can be installed via the `conda <http://www.continuum.io/blog/conda>`_  package manager.

Required
^^^^^^^^

* OpenMM with Python wrappers compiled:
  http://openmm.org

* Python 2.7 or later:
  http://www.python.org

* NetCDF (compiled with netcdf4 support):
  http://www.unidata.ucar.edu/software/netcdf/

* HDF5 (required by NetCDF4):
  http://www.hdfgroup.org/HDF5/

* netcdf4-python (a Python interface for netcdf4):
  http://code.google.com/p/netcdf4-python/

* numpy and scipy:
  http://www.scipy.org/

  Some components of YANK can exploit `scipy.weave <http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/weave.html>`_ to accelerate stages of the calculation, though this functionality will be deprecated in future versions.

Optional
^^^^^^^^

* AmberTools (for setting up protein-ligand systems):
  http://ambermd.org/#AmberTools

* OpenEye toolkit and Python wrappers (if mol2 and PDB reading features are used ;requires academic or commercial license):
  http://www.eyesopen.com

* mpi4py (if MPI support is desired):
  http://mpi4py.scipy.org/
  (Note that the mpi4py installation must be compiled against the appropriate MPI implementation.)

Hardware
========

Supported hardware
------------------

YANK makes use of `openmm <http://www.openmm.org>`_, a GPU-accelerated framework for molecular simulation.
This allows the calculations to take advantage of hardware that supports CUDA (such as NVIDIA GPUs) and OpenCL (NVIDIA and ATI GPUs, as well as some processors).
OpenMM also supports a multithreaded CPU platform which can be used if no CUDA or OpenCL resources are available.

Recommended hardware
--------------------

We have found the best price/performance results are currently obtained with NVIDIA GTX-class consumer-grade cards, such as the GTX-680, GTX-780, and GTX-Titan cards.

**********************
Installing from source
**********************

Installing from the GitHub source repository
============================================

Installing from source is only recommended for developers that wish to modify YANK or the algorithms it uses.
Installation via `conda` is preferred for all other users.

The source code for YANK resides on `GitHub <http://github.com/choderalab/yank>`_.

Clone the source code repository from github::

  $ git clone git://github.com/choderalab/yank.git
  $ cd yank/
  $ python setup.py install

If you wish to install into a different path (often preferred for development), use

  $ python setup.py install --prefix=$PREFIX

where `$PREFIX` is the desired installation path.
Note that `$PREFIX/lib/python2.7/site-packages/` must be on your `$PYTHONPATH`.

`setup.py` will try to install some of the dependencies, or at least check that you have them installed and throw an error.

Testing your installation
=========================

Test your YANK installation to make sure everything is behaving properly on your machine:

  $ yank selftest

This will not only check that installation paths are correct, but also run a battery of tests that ensure any automatically detected GPU hardware is behaving as expected.

