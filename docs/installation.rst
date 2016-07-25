.. _installation:

Installation
************

Installing via `conda`
======================

The simplest way to install YANK is via the `conda <http://www.continuum.io/blog/conda>`_  package manager.
Packages are provided on the `omnia Anaconda Cloud channel <http://anaconda.org/omnia>`_ for Linux, OS X, and Win platforms.
The `yank Anaconda Cloud page <https://anaconda.org/omnia/yank>`_ has useful instructions and `download statistics <https://anaconda.org/omnia/yank/files>`_`.

If you are using the `anaconda <https://www.continuum.io/downloads/>`_ scientific Python distribution, you already have the ``conda`` package manager installed.
If not, the quickest way to get started is to install the `miniconda <http://conda.pydata.org/miniconda.html>`_ distribution, a lightweight minimal installation of Anaconda Python.

On `linux`, you can install the Python 2.7 version into `$HOME/miniconda2` with (on `bash` systems):

.. code-block:: bash

   $ wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
   $ bash ./Miniconda2-latest-Linux-x86_64.sh -b -p $HOME/miniconda2
   $ export PATH="$HOME/miniconda2/bin:$PATH"

On `osx`, you want to use the `osx` binary

.. code-block:: bash

   $ wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
   $ bash ./Miniconda2-latest-Linux-x86_64.sh -b -p $HOME/miniconda2
   $ export PATH="$HOME/miniconda2/bin:$PATH"

You may want to add the new `$PATH` extension to your `~/.bashrc` file to ensure Anaconda Python is used by default.
Note that YANK will be installed into this local Python installation, so that you will not need to worry about disrupting existing Python installations.

|

Release build
-------------

You can install the latest stable release build of YANK via the ``conda`` package with

.. code-block:: none

   $ conda config --add channels omnia
   $ conda install yank

This version is recommended for all users not actively developing new algorithms for alchemical free energy calculations.

.. note:: ``conda`` will automatically dependencies from binary packages automatically, including difficult-to-install packages such as OpenMM, numpy, and scipy. This is really the easiest way to get started.

|

Development build
-----------------

The bleeding-edge, absolute latest, very likely unstable development build of YANK is pushed to `binstar <https://binstar.org/omnia/yank>`_ with each GitHub commit, and can be obtained by

.. code-block:: bash

   $ conda config --add channels omnia
   $ conda install yank-dev

.. warning:: Development builds may be unstable and are generally subjected to less testing than releases.  Use at your own risk!


Upgrading your installation
---------------------------

To update an earlier ``conda`` installation of YANK to the latest release version, you can use ``conda update``:

.. code-block:: bash

   $ conda update yank

|

.. _yank-dev-conda-package:

Testing your installation
-------------------------

Test your YANK installation to make sure everything is behaving properly on your machine:

.. code-block:: bash

   $ yank selftest

This will not only check that installation paths are correct, but also run a battery of tests that ensure any automatically detected GPU hardware is behaving as expected.

Supported platforms and environments
====================================

Software
--------

YANK runs on Python 2.7.
The developers generally use Python 2.7 on both Mac and Linux platforms.
Automated tests on Linux are performed on every GitHub commit using `Travis CI <http://travis-ci.org>`_, and release tests are performed on Mac and Linux platforms using `Jenkins <http://jenkins.choderalab.org>`_..

Dependencies
++++++++++++

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

* `AmberTools <http://ambermd.org/#AmberTools>`_ is helpful for setting up protein-ligand systems using LEaP.

* `mpi4py <http://mpi4py.scipy.org/>`_ is needed if  MPI support is desired.

.. note:: The ``mpi4py`` installation must be compiled against the system-installed MPI implementation used to launch jobs.

* The `OpenEye toolkit and Python wrappers <http://www.eyesopen.com/toolkits>`_ can be used to enable free energy calculations to be set up directly from any supported OpenEye format, including mol2, PDB, ChemDraw, and many more (requires academic or commercial license).

* `scipy.weave <http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/weave.html>`_ is an optional dependency for the replica-exchange code, though this functionality will be migrated to `cython <http://cython.org>`_ in future revisions.

Hardware
--------

Supported hardware
++++++++++++++++++

YANK makes use of `openmm <http://www.openmm.org>`_, a GPU-accelerated framework for molecular simulation.
This allows the calculations to take advantage of hardware that supports CUDA (such as NVIDIA GPUs) or OpenCL (NVIDIA and ATI GPUs, as well as some processors).
OpenMM also supports a multithreaded CPU platform which can be used if no CUDA or OpenCL resources are available.

Recommended hardware
++++++++++++++++++++

We have found the best price/performance results are currently obtained with NVIDIA GTX-class consumer-grade cards, such as the GTX-680, GTX-780, and GTX-Titan cards.

Ross Walker and the Amber GPU developers maintain a set of `excellent pages with good inexpensive GPU hardware recommendations <http://ambermd.org/gpus/recommended_hardware.htm>`_.

Amazon EC2 now provides `Linux GPU instances <http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using_cluster_computing.html>`_ with high-performance GPUs and inexpensive on-demand and `spot pricing <http://aws.amazon.com/ec2/purchasing-options/spot-instances/>`_ (g2.2xlarge).  We will soon provide ready-to-use images to let you quickly get started on EC2.

Installing from source
======================

.. note:: We recommend only developers wanting to modify the YANK code should install from source. Users who want to use the latest development version are advised to install the :ref:`Development build conda package <yank-dev-conda-package>` instead.

Installing from the GitHub source repository
--------------------------------------------

Installing from source is only recommended for developers that wish to modify YANK or the algorithms it uses.
Installation via `conda` is preferred for all other users.

Clone the source code repository from `GitHub <http://github.com/choderalab/yank>`_.

.. code-block:: bash

   $ git clone git://github.com/choderalab/yank.git
   $ cd yank/
   $ python setup.py install

If you wish to install into a different path (often preferred for development), use

.. code-block:: bash

   $ python setup.py install

``setup.py`` will try to install some of the dependencies, or at least check that you have them installed and throw an error.
Note that not all dependencies can be installed via ``pip``, so you will have to install dependencies if installation fails due to unmet dependencies.

Testing your installation
-------------------------

Test your YANK installation to make sure everything is behaving properly on your machine:

.. code-block:: bash

   $ yank selftest

This will not only check that installation paths are correct, but also run a battery of tests that ensure any automatically detected GPU hardware is behaving as expected.
