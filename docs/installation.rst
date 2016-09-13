.. _installation:

Installation
************

Installing via `conda`
======================

The simplest way to install YANK is via the `conda <http://www.continuum.io/blog/conda>`_  package manager.
Packages are provided on the `omnia Anaconda Cloud channel <http://anaconda.org/omnia>`_ for Linux, OS X, and Win platforms.
The `yank Anaconda Cloud page <https://anaconda.org/omnia/yank>`_ has useful instructions and `download statistics <https://anaconda.org/omnia/yank/files>`_.

If you are using the `anaconda <https://www.continuum.io/downloads/>`_ scientific Python distribution, you already have the ``conda`` package manager installed.
If not, the quickest way to get started is to install the `miniconda <http://conda.pydata.org/miniconda.html>`_ distribution, a lightweight minimal installation of Anaconda Python.

On ``linux``, you can install the Python 2.7 version into ``$HOME/miniconda2`` with (on ``bash`` systems):

.. code-block:: bash

   $ wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
   $ bash ./Miniconda2-latest-Linux-x86_64.sh -b -p $HOME/miniconda2
   $ export PATH="$HOME/miniconda2/bin:$PATH"

On ``osx``, you want to use the ```osx`` binary

.. code-block:: bash

   $ wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
   $ bash ./Miniconda2-latest-Linux-x86_64.sh -b -p $HOME/miniconda2
   $ export PATH="$HOME/miniconda2/bin:$PATH"

You may want to add the new ```$PATH`` extension to your ``~/.bashrc`` file to ensure Anaconda Python is used by default.
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

|

.. _yank-platforms:

Testing Available Platforms
---------------------------

You will want to make sure that all GPU accelerated platforms available on your hardware are accessible to YANK. The simulation library that YANK runs on, OpenMM, can run on CPU, CUDA, and OpenCL platforms. The following command will check which platforms are available:

.. code-block:: bash

   $ yank platforms

You should see an output that looks like the following:

.. code-block:: bash

   Available OpenMM platforms:
    0 Reference
    1 CPU
    2 CUDA
    3 OpenCL

If your output is missing on option you expect, such as CUDA on Nvidia GPUs, then please check that you have correct drivers for your GPU installed. Non-standard CUDA installations require setting specific environment variables; please see the :ref:`appropriate section <non-standard-cuda>` for setting these variables.

|

.. _non-standard-cuda:

Configuring Non-Standard CUDA Install Locations
-----------------------------------------------

Multiple versions of CUDA can be installed on a single machine, such as on shared clusters. If this is the case, it may be necessary to set environment variables to make sure that the right version of CUDA is being used for YANK. You will need to know the full ``<path_to_cuda_install>`` and the location of that installation's ``nvcc`` program is (by default it is at ``<path_to_cuda_install>/bin/nvcc``). Then run the following lines to set the correct variables:

.. code-block:: bash

   export OPENMM_CUDA_COMPILER=<path_to_cuda_install>/bin/nvcc
   export LD_LIBRARY_PATH=<path_to_cuda_install>/lib64:$LD_LIBRARY_PATH

You may want to add the new ``$OPENMM_CUDA_COMPILER`` variable and ``$LD_LIBRARY_PATH`` extension to you ``~/.bashrc`` file to avoid setting this every time. If ``nvcc`` is installed in a different folder than the example, please use the correct path for your system.

|

Optional Tools
--------------

The `OpenEye toolkit and Python wrappers <http://www.eyesopen.com/toolkits>`_ can be installed to enable free energy calculations to be set up directly from `multiple supported small molecule formats <https://docs.eyesopen.com/toolkits/python/oechemtk/molreadwrite.html#file-formats>`_, including

* SDF
* SMILES
* IUPAC names
* Tripos mol2
* PDB

Note that PDB and mol2 are supported through the pure AmberTools pipeline as well, though this does not provide access to the OpenEye AM1-BCC charging pipeline.

Use of the OpenEye toolkit requires an `academic or commercial license <http://www.eyesopen.com/licensing-philosophy>`_.

To install these tools into your conda environment, use `pip`:

.. code-block:: bash

   $ pip install -i https://pypi.anaconda.org/OpenEye/simple OpenEye-toolkits

Note that you will need to configure your ``$OE_LICENSE`` environment variable to point to a valid license file.

You can test your OpenEye installation with

.. code-block:: bash

   $ python -m openeye.examples.openeye_tests

|

Supported platforms and environments
====================================

Software
--------

YANK runs on Python 2.7.
Support for Python 3.x is planned for a future release.

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

* ``netcdf4-python`` (a Python interface for netcdf4):
  http://code.google.com/p/netcdf4-python/

* ``numpy`` and ``scipy``:
  http://www.scipy.org/

* ``docopt``:
  http://docopt.org/

* ``alchemy``
  https://github.com/choderalab/alchemy

* ``pymbar``
  https://github.com/choderalab/pymbar

* ``schema``
  https://pypi.python.org/pypi/schema

* `AmberTools <http://ambermd.org/#AmberTools>`_ is needed for setting up protein-ligand systems using LEaP.
  https://github.com/choderalab/ambertools

Optional
^^^^^^^^

* `mpi4py <http://mpi4py.scipy.org/>`_ is needed if `MPI support <https://de.wikipedia.org/wiki/Message_Passing_Interface>`_ is desired.

.. note:: The ``mpi4py`` installation must be compiled against the system-installed MPI implementation used to launch jobs. Using the ``conda`` version of ``mpi4py`` together with the ``conda``-provided ``mpirun`` is the simplest way to avoid any issues.

* The `OpenEye toolkit and Python wrappers <http://www.eyesopen.com/toolkits>`_ can be used to enable free energy calculations to be set up directly from multiple supported OpenEye formats, including Tripos mol2, PDB, SMILES, and IUPAC names (requires academic or commercial license).
Note that PDB and mol2 are supported through the pure AmberTools pipeline as well, though this does not provide access to the OpenEye AM1-BCC charging pipeline.

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
You can find some benchmarks for OpenMM on several classes of recent GPUs at `openmm.org <http://openmm.org/about.html#benchmarks>`_.

Ross Walker and the Amber GPU developers maintain a set of `excellent pages with good inexpensive GPU hardware recommendations <http://ambermd.org/gpus/recommended_hardware.htm>`_ that will also work well with OpenMM and YANK.

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

This will not only check that installation paths are correct, but also run a battery of tests that ensure any automatically detected GPU hardware is behaving as expected. Please also check that YANK has access to the :ref:`expected platforms <yank-platforms>` and the :ref:`correct CUDA version <non-standard-cuda>` if CUDA is installed in a non-standard location.

Running on the cloud
--------------------

Amazon EC2 now provides `Linux GPU instances <http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using_cluster_computing.html>`_ with high-performance GPUs and inexpensive on-demand and `spot pricing <http://aws.amazon.com/ec2/purchasing-options/spot-instances/>`_ (g2.2xlarge).
We will soon provide ready-to-use images to let you quickly get started on EC2.

We are also exploring building `Docker containers <https://hub.docker.com/>`_ for rapid, reproducible, portable deployment of YANK to new compute environments.
Stay tuned!
