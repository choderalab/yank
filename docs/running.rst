.. _running:

Running YANK
************

Hardware
========

Running on GPUs
"""""""""""""""

YANK uses `OpenMM <http://openmm.org>`_ as its simulation engine, which runs fastest on modern GPUs using either the
``CUDA`` or ``OpenCL`` platforms.
Modern GTX-class hardware, such as the `GTX-1080 <http://www.geforce.com/hardware/10series/geforce-gtx-1080>`_ or
`GTX-TITAN-X (Maxwell) <http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x>`_, should work very well.
See :ref:`Supported hardware <supported_hardware>` for more information about supported and recommended hardware.

Running on the CPU
""""""""""""""""""

If you don't have a GPU, the OpenMM ``CPU`` platform will run in multithreaded mode by default (see `Selecting a platform`_).
While not as fast as `gromacs <http://www.gromacs.org>`_, running on the CPU can still let you explore the features of YANK without needing a GPU.
You can also use CPUs acting as ``OpenCL`` devices utilizing the `AMD OpenCL APP SDK <http://developer.amd.com/tools-and-sdks/opencl-zone/>`_ or the `Intel OpenCL SDK <https://software.intel.com/en-us/intel-opencl>`_, though this has not been extensively tested.

Parallelization
===============

YANK can operate in two different modes:

Serial mode
"""""""""""

In :ref:`serial mode <serial-mode>`, a single simulation is run at a time, either on the GPU or CPU.
For replica-exchange calculations, this means that each replica is run serially, and then exchanges between replicas are allowed to occur.
This is useful if you are testing YANK, or running a large number of free energy calculations where speed is not a major concern.

.. _mpi_notes:

MPI mode
""""""""

In :ref:`MPI mode <mpi-mode>`, multiple simulations can be run at once, either on multiple GPUs or multiple CPUs, though
running simulations on a mixture of platforms is not supported at this time.
We utilize the widely-supported
`Message Passing Interface (MPI) standard <http://www.mcs.anl.gov/research/projects/mpi/standard.html>`_ for parallelization.
All simulations are run using the same OpenMM ``Platform`` choice (``CUDA``, ``OpenCL``, ``CPU``, or ``Reference``)

No GPU management is needed on single GPU/node systems.

Multi-GPU and multi-node systems require masking the GPUs so YANK only sees the one its suppose to.
You will need to set the ``CUDA_VISIBLE_DEVICES`` environment variables on each process to mask all but the card you intend to use.
We cannot provide a universal solution as systems will differ, but we can provide some general rules of thumb.

.. note:: |mpichNotes|

.. |mpichNotes| replace::
    All our documentation assumes MPICH 3 formatting which has the package name ``mpich`` and version formatting ``3.X``.
    This is distinct from MPICH2 which has the package name ``mpich2`` and version formatting ``1.X``.
    Please see `the official MPICH site <https://www.mpich.org/about/overview/>`__ for more information about the
    version naming and numbering.


* |torquepbs|

.. |torquepbs| replace::
    ``torque`` or ``PBS`` cluster - Install the `clusterutils <https://github.com/choderalab/clusterutils>`__ module (automatic when YANK
    is installed from conda). Run the ``build_mpirun_configfile "yank --script=yank.yaml"`` targeting your YAML file.
    The command works by targeting the cluster's ``$PBS_GPUFILE`` to determine device IDs as the cluster sees them.


*  |slurmsub|

.. |slurmsub| replace::
    ``SLURM`` cluster -  SLURM handles some of the GPU masking for you by setting ``CUDA_VISIBLE_DEVICES``, but we have
    found breaks down over multiple nodes. The `clusterutils <https://github.com/choderalab/clusterutils>`__ will also
    help you configure the MPI jobs to run over multiple nodes. The command in this case operates by reading environment
    variables to figure out host and card names.


* |lsfsub|

.. |lsfsub| replace::
    ``LSF`` cluster - Rely on the `clusterutils <https://github.com/choderalab/clusterutils>`__ module which comes with
    YANK when conda installed. This configures hosts and GPU masking based on the batch job environment variables which
    are set at job submission.



We recommend running YANK with an MPI configfile. You can specify all of these sets to ``CUDA_VISIBLE_DEVICES`` by hand,
but it will make for a long command line. Creating a hostfile and configfile to feed into MPI is the preferred option
for this reason. Although `clusterutils <https://github.com/choderalab/clusterutils>`_ will handle file creation for you,
below we show an example of a hostfile and configfile in case you need to make the files by hand. This is a 2 node,
4 GPU/node torque cluster (8 processes distributed over 2 nodes) setup:

Hostfile:

.. code-block:: bash

    gpu-1-4
    gpu-1-4
    gpu-1-4
    gpu-1-4
    gpu-2-17
    gpu-2-17
    gpu-2-17
    gpu-2-17

Configfile:

.. code-block:: bash

    -np 1 -env CUDA_VISIBLE_DEVICES 3 yank script --yaml=freesolv.yaml
    -np 1 -env CUDA_VISIBLE_DEVICES 2 yank script --yaml=freesolv.yaml
    -np 1 -env CUDA_VISIBLE_DEVICES 1 yank script --yaml=freesolv.yaml
    -np 1 -env CUDA_VISIBLE_DEVICES 0 yank script --yaml=freesolv.yaml
    -np 1 -env CUDA_VISIBLE_DEVICES 3 yank script --yaml=freesolv.yaml
    -np 1 -env CUDA_VISIBLE_DEVICES 2 yank script --yaml=freesolv.yaml
    -np 1 -env CUDA_VISIBLE_DEVICES 1 yank script --yaml=freesolv.yaml
    -np 1 -env CUDA_VISIBLE_DEVICES 0 yank script —-yaml=freesolv.yaml

Hopefully this example helps you construct your own configfile if the ``build_mpirun_configfile`` is unavailable or you
are on a custom system.

Simulations may be started in one mode and then can be resumed using another parallelization mode or OpenMM ``Platform``.
The NetCDF files generated by YANK are platform-portable and hardware agnostic, so they can be moved from system to
system if you want to start a simulation on one system and resume it elsewhere.

Parallelization is "pleasantly parallel", where information is exchanged only every few seconds or more.
This does not require high-bandwidth interconnects such as `Infiniband <https://en.wikipedia.org/wiki/InfiniBand>`_;
10Gbit/s ethernet connections should work very well.

We show a more formal "use case" of setting up an MPI run :ref:`below <mpi-mode>`.

.. _getting-help:

Getting help
============

To get a list of all command-like options, simply use the ``--help`` flag:

.. code-block:: bash

   $ yank --help

Each YANK command supports its own full command-line support. Type in any of these commands without options or
``yank help COMMAND`` to see what they do.

.. tabularcolumns:: |l|L|

===================  ============================================
Command              Description
===================  ============================================
``yank help``        The basic help message and this list
``yank selftest``    Check YANK's install status and hardware
``yank script``      Primary tool for running yank from command line from options file
``yank platforms``   List platforms available on current hardware
``yank status``      Deprecated - Check the current status of a store directory
``yank analyze``     Analyze a simulation or make an analysis Jupyter Notebook
``yank cleanup``     Remove simulation files
===================  ============================================

|

.. _serial-mode:

Running in serial mode
======================

To run the simulation in serial mode, simply use ``yank script``, specifying a yaml file by ``--yaml=filename.yaml``:

.. code-block:: bash

   $ yank script --yaml=yank.yaml

The optional :ref:`yaml_options_verbose` option will show additional output during execution.

.. _mpi-mode:

Running in MPI mode
===================

Alternatively, to run the simulation in MPI mode:

.. code-block:: none

   $ mpirun yank script --yaml=yank.yaml

Keep in mind your system may have a different MPI executable name and/or syntax.

On systems with multiple NVIDIA GPUs per node, it is necessary to perform masking using ``CUDA_VISIBLE_DEVICES``.

On systems using the conda-installed ``mpi4py`` package, the `MPICH hydra mpirun <https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager>`_ will be automatically installed for you.
You can use the cluster utility command
``build-mpirun-configfile``
available in our `clusterutils <https://github.com/choderalab/clusterutils>`_ tools to generate an appropriate ``configfile``:

.. code-block:: none

  $ build-mpirun-configfile "yank script --yaml=yank.yaml"
  $ mpirun -f hostfile -configfile configfile

``build-mpirun-configfile`` is automatically installed with YANK when you use the ``conda`` installation route. Please
see our :ref:`notes from above <mpi_notes>` about this script's applicability on torque, PBS, SLURM and LSF clusters.

.. note::

   The name of your ``mpi`` binary may be different than what is shown in the example. Make sure you are using the
   correct binary, especially on systems which already had an MPI installed before the ``conda`` one was installed.

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
the desired platform via the :ref:`platform argument <yaml_options_platform>` in a YAML file.

You can also (*although not recommended*) override the platform selection through the ``yank script -o`` flag.
For example, to force YANK to use the ``OpenCL`` platform:

.. code-block:: bash

   $ yank script --yaml=yank.yaml -o options:platform:OpenCL

See the ``yank script`` command line docs for more information on the ``-o`` flag.

.. note:: The ``CPU`` platform will automatically use all available cores/hyperthreads in serial mode, but in MPI mode, will use a single thread to avoid causing problems in queue-regulated parallel systems.  To control the number of threads yourself, set the ``OPENMM_NUM_THREADS`` environment variable to the desired number of threads.


Breaking up Simulations
=======================

One limitation simulations run to compute free energy is that you need sufficient data to estimate all legs of the
:ref:`thermodynamic cycle <yank_cycle>` before you can estimate many properties. YANK by default simulates each phase of
the thermodynamic cycle sequentially, meaning you have to wait for simulation completion before you can do any kind of
analysis.

YANK allows its simulations to switch between phases of the thermodynamic cycle mid-simulation. Setting the
:ref:`yaml_options_switch_phase_interval` option in the YAML file allows you to control how many iterations the
simulation will run of one phase before switching to the other to allow collection between each phase on-the-fly, and
therefore allowing analysis at any point. This feature is helpful for those who want to run for an arbitrarily long
length of time, then wish to end the simulation after some convergence criteria is met, e.g. error in free energy
estimate.

YANK also provides a way flip between different experiments when setting the
:ref:`\!Combinatorial <yaml_combinatorial_head>` flag. The :ref:`yaml_options_switch_experiment_interval` allows
simulations to cycle through all the experiments created when users invoke the ``!Combinatorial`` flag. This option
can be set in tandem with the :ref:`yaml_options_switch_phase_interval`. Please see the respective documentation on
these options for more information.


Specifying Simulation Stop Conditions
=====================================

YANK simulations will run until one of two stop conditions are met, if specified: either the
:ref:`maximum number of iterations <yaml_options_default_number_of_iterations>` is reached, or the
:ref:`error in free energy difference of the phase <yaml_options_online_analysis_parameters>` reaches a target value
through online analysis.
These options can be combined to change when YANK stops a simulation.

Specifying a :ref:`maximum number of iterations <yaml_options_default_number_of_iterations>` will tell YANK to run each phase
up to the target number of iterations. This options accepts any positive integer, ``0`` (zero), or infinity
(``.inf`` in the .yaml, ``float('inf')`` or ``numpy.inf`` in the API). Setting this option to ``0`` will tell YANK to only handle file
initialization and input preparation, without running any production simulation. Setting this to ``.inf`` will
run an unlimited number of simulations until another stop condition is met, or is stopped by the user. This option
can be increased after a simulation has completed to extend the number of iterations run for each phase.

Setting a target :ref:`free energy difference error <yaml_options_online_analysis_parameters>` tells
YANK to run each phase until the error in the free energy difference is below some threshold. The free energy difference
of the phase is estimated during the simulation through online analysis every
:ref:`specified interval <yaml_options_online_analysis_interval>`.
This process will slow down the simulation, so it is recommended the interval be at least 100 iterations, if not more.

It is recommended you also set a :ref:`switch phase interval <yaml_options_switch_phase_interval>` for a large number
of iterations, especially if an unlimited number of iterations are specified, otherwise only one phase will be
simulated.

..
    Extending Simulations
    =====================

    One common operation when running simulations is to collect additional samples from an already run simulation to get
    better statistics. Alternately, when running on shared resources, you may need to break up long simulations into smaller
    simulations run in series. YANK provides a way to run its simulations in this manner by extending its simulations.

    YANK's :doc:`YAML <yamlpages/index>` files have two main options that work together to extend simulations.
    In order to extend simulations, set the following options in a YAML fie:

    .. code-block:: yaml

      number_of_iterations: <Integer>
      extend_simulation: True

    First you set :ref:`yaml_options_number_of_iterations` to an integer number of iterations you wish to extend the
    simulation. If no simulation has been run yet, then one will be run for the number of iterations.
    Setting :ref:`yaml_options_extend_simulation` to ``True`` modifies the behavior of
    :ref:`yaml_options_number_of_iterations` to extend the simulation by the specified number, adding on to what is already
    on the file.

    One could optionally just increase :ref:`yaml_options_number_of_iterations`, but then you have to change
    the YAML file every time you want to extend the run. Setting :ref:`yaml_options_extend_simulation` allows you to run
    the same YAML file without modification to do the same thing.


    You should also set the following two options as well as :ref:`yaml_options_number_of_iterations` and
    :ref:`yaml_options_extend_simulation`:

    .. code-block:: yaml

      resume_setup: yes
      resume_simulation: yes

    :ref:`resume_setup <yaml_options_resume_setup>` and :ref:`resume_simulation <yaml_options_resume_simulation>` allow
    YANK to resume simulations if it detects existing setup file or simulation output respectively. YANK will raise an error
    if these are not set and files exist to protect against overwrite. The only reason these are not mandatory is that if
    no files exist (i.e. fresh simulation), then the simulation will run without error once.


    Extending Previous Simulations from Command Line
    """"""""""""""""""""""""""""""""""""""""""""""""

    You may already have a simulation that you previously ran, but do not want to modify the YAML to extend the simulation.
    In this case, your YAML file has ``extend_simulation: False`` or is not set, and you only want to interact with the
    simulation through the command line. You can override individual settings from the command line; the settings for
    extending simulation would look like:

    .. code-block:: bash

       $ yank script --yaml=yank.yaml -o options:extend_simulation:True -o options:number_of_iterations:X

    where ``X`` is the integer number you wish to extend the simulation by. The second option to override
    ``number_of_iterations`` is optional if you are happy the existing option in the YAML file.

    See the ``yank script`` command line docs for more information on the ``-o`` flag.
