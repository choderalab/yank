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

If you are using the `anaconda <https://store.continuum.io/cshop/anaconda/>`_ scientific Python distribution, you already have the ``conda`` package manager installed.
If not, but you have the ``pip`` package manager installed (to access packages from `PyPI <http://pypi.org>`_), you can easily install ``conda`` with

.. code-block:: none

   $ pip install conda

Release build
=============

You can install the latest stable release build of YANK via the ``conda`` package with

.. code-block:: none

   $ conda config --add channels http://conda.binstar.org/omnia
   $ conda install yank

This version is recommended for all users not actively developing new algorithms for alchemical free energy calculations.

.. note:: ``conda`` will automatically dependencies from binary packages automatically, including difficult-to-install packages such as numpy and scipy. This is really the easiest way to get started.

Upgrading your instalation
==========================

To update an earlier ``conda`` installation of YANK to the latest release version, you can use ``conda update``:

.. code-block:: none

   $ conda update yank

Development build
=================

The bleeding-edge, absolute latest, very likely unstable development build of YANK is pushed to `binstar <https://binstar.org/omnia/yank>`_ with each GitHub commit, and can be obtained by

.. code-block:: none

   $ conda config --add channels http://conda.binstar.org/omnia
   $ conda install yank-dev

Again, this build may very likely be unstable, so use at your own risk!

Testing the installation
========================

Test your YANK installation to make sure everything is behaving properly on your machine:

.. code-block:: none

   $ yank selftest

This will not only check that installation paths are correct, but also run a battery of tests that ensure any automatically detected GPU hardware is behaving as expected.

************************************
Supported platforms and environments
************************************

Software
========

YANK runs on Python 2.7 and Python 3.3 or 3.4.
The developers generally use Python 2.7, on both Mac and Linux platforms.
Automated tests on Linux are performed on every GitHub commit using `Travis CI <http://travis-ci.org>`_, and release tests are performed on Mac and Linux platforms using `Jenkins <http://jenkins.choderalab.org>`_..

Dependencies
------------

YANK uses a number of tools in order to allow the developers to focus on developing efficient algorithms involved in alchemical free energy calculations, rather than reinventing basic software, numerical, and molecular simulation infrastructure.
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

Optional
^^^^^^^^

* mpi4py is needed if  MPI support is desired:
  http://mpi4py.scipy.org/
  (Note that the mpi4py installation must be compiled against the appropriate MPI implementation.)

* The OpenEye toolkit and Python wrappers can be used to enable free energy calculations to be set up directly from any supported OpenEye format, including mol2, PDB, ChemDraw, and many more (requires academic or commercial license):
  http://www.eyesopen.com

* `scipy.weave <http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/weave.html>`_ is an optional dependency for the replica-exchange code, though this functionality will be migrated to `cython <http://cython.org>`_ in future revisions

* AmberTools can be used for setting up protein-ligand systems using LEaP:
  http://ambermd.org/#AmberTools

Hardware
========

Supported hardware
------------------

YANK makes use of `openmm <http://www.openmm.org>`_, a GPU-accelerated framework for molecular simulation.
This allows the calculations to take advantage of hardware that supports CUDA (such as NVIDIA GPUs) or OpenCL (NVIDIA and ATI GPUs, as well as some processors).
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

Clone the source code repository from `GitHub <http://github.com/choderalab/yank>`_.

.. code-block:: none

   $ git clone git://github.com/choderalab/yank.git
   $ cd yank/
   $ python setup.py install

If you wish to install into a different path (often preferred for development), use

.. code-block:: none

   $ python setup.py install --prefix=$PREFIX

where ``$PREFIX`` is the desired installation path.
Note that ``$PREFIX/lib/python2.7/site-packages/`` must be on your ``$PYTHONPATH``.

``setup.py`` will try to install some of the dependencies, or at least check that you have them installed and throw an error.

Testing your installation
=========================

Test your YANK installation to make sure everything is behaving properly on your machine:

.. code-block:: none

   $ yank selftest

This will not only check that installation paths are correct, but also run a battery of tests that ensure any automatically detected GPU hardware is behaving as expected.

