.. _running:

Running YANK
************

YANK can operate in two different parallelization modes:

Serial mode
-----------

In :ref:`serial mode <serial-mode>`, a single simulation is run at a time, either on the GPU or CPU.  YANK uses `OpenMM <http://openmm.org>`_ as its simulation engine, which runs fastest on modern GPUs using either the ``CUDA` or ``OpenCL`` platforms, but if you don't have a GPU, the OpenMM ``CPU`` platform will run in multithreaded mode by default.  While not as fast as `gromacs <http://www.gromacs.org>`_, this can still let you explore the features of YANK without needing a GPU.

MPI mode
--------

In :ref:`mpi mode <mpi-mode>`, multiple simulations can be run at once, either on multiple GPUs or multiple CPUs using `MPI <http://www.mcs.anl.gov/research/projects/mpi/standard.html>`_. All simulations are run using the same OpenMM ``Platform`` choice (one of ``CUDA``, ``OpenCL``, ``CPU``, or ``Reference``); running simulations on a mixture of platforms is not supported at this time.

Simulations may be started in one mode and then can be resumed using another parallelization mode or OpenMM platform.
The NetCDF files in each ``store`` directory are platform-portable and hardware agnostic, so they can be moved from system to system if you want to start a simulation on one system and resume it elsewhere.

.. _getting-help

Getting help
============

To get a list of all command-like options, simply use the ``--help`` flag:

.. code-block:: none

   $ yank --help

|

.. _serial-mode:

Running in serial mode
======================

To run the simulation in serial mode, simply use ``yank run``, specifying a store directory by ``--store=dirname``:

.. code-block:: none

   $ yank run --store=output

The optional ``--verbose`` flag will show additional output during execution.

.. _mpi-mode:

Running in MPI mode
===================

Alternatively, to run the simulation in MPI mode:

.. code-block:: none

   $ yank run --store=output --mpi

On Torque/Moab systems with multiple NVIDIA GPUs per node, it is necessary to perform masking using ``CUDA_VISIBLE_DEVICES``.

On systems using the conda-installed ``mpi4py`` package, the `MPICH2 hydra mpirun <https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager>`_ will be automatically installed for you, and the cluster utility script `build-mpirun-configfile.py <https://github.com/choderalab/clusterutils/blob/master/scripts/build-mpirun-configfile.py>`_ available in our `clusterutils <https://github.com/choderalab/clusterutils>`_ repo will be of use:

.. code-block:: none

  $ python build-mpirun-configfile.py yank run --store=output --mpi --verbose
  $ mpirun -configfile configfile

|

Selecting a platform
====================

OpenMM supports running simulations on a number of platforms, though not all platforms are available on all hardware.
To see which platforms your current installation supports, you can query the list of available platforms with

.. code-block:: none

  $ yank platforms
  Available OpenMM platforms:
      0 Reference
      1 CUDA
      2 CPU
      3 OpenCL

You can either leave the choice of platform up to YANK---in which case it will choose the fastest available platform---or specify
the desired platform via the ``--platform`` argument to ``yank run``.  For example, to force YANK to use the ``OpenCL`` platform:

.. code-block:: none

   $ yank run --store=output --platform=OpenCL

.. note:: The ``CPU`` platform will automatically use all available cores/hyperthreads in serial mode, but in MPI mode, will use a single thread to avoid causing problems in queue-regulated parallel systems.  To control the number of threads yourself, set the ``OPENMM_NUM_THREADS`` environment variable to the desired number of threads.

