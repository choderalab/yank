.. _running:

************
Running YANK
************

Serial mode
===========

To run the simulation in serial mode, simply use ``yank run``, specifying a store directory by ``--store=dirname``:

.. code-block:: none

   $ yank run --store=output

The optional ``--verbose`` flag will show additional output during execution.

MPI mode
========

Alternatively, to run the simulation in MPI mode:

.. code-block:: none

   $ yank run --store=output --mpi

On Torque/Moab systems with multiple NVIDIA GPUs per node, it is necessary to perform masking using ``CUDA_VISIBLE_DEVICES``.
This cluster utility script `build-mpirun-configfile.py <https://github.com/choderalab/cluster-utils/blob/master/scripts/build-mpirun-configfile.py>`_ will be of use.
This assumes you are using the `MPICH2 hydra mpirun <https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager>`_ installed via the ``mpi4py`` conda package.

.. code-block:: none

  $ python build-mpirun-configfile.py yank run --store=output --mpi --verbose
  $ mpirun -configfile configfile


