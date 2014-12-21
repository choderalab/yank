.. _quickstart:

##############################
Quickstart (for the impatient)
##############################

First, install the `anaconda <https://store.continuum.io/cshop/anaconda/>`_ scientific Python distribution.

Install the release version of YANK from the `omnia binstar channel <https://binstar.org/omnia/yank>`_:

.. code-block:: none

   $ conda install -c https://conda.binstar.org/omnia yank

Go to the ``examples/p-xylene-implicit`` directory to find an example of computing the binding affinity of p-xylene to T4 lysozyme L99A:

.. code-block:: none

   $ cd ~/anaconda/share/yank/examples/p-xylene-implicit

Set up the free energy calculation using the ``OBC2`` implicit solvent model, specifying the residue named ``BEN`` as the ligand, selecting the temperature of 300 Kelvin, and using the diretory ``output/`` to write data to, using:

.. code-block:: none

   $ yank setup binding amber --setupdir=setup --ligname=BEN --store=output --iterations=1000 \
     --restraints=harmonic --gbsa=OBC2 --temperature=300*kelvin --verbose

To run the simulation in serial mode, use:

.. code-block:: none

   $ yank run --store=output --verbose

Alternatively, to run the simulation in MPI mode, use:

.. code-block:: none

   $ yank run --store=output --mpi --verbose

Note that, in MPI mode, the default device id for each GPU is used.
See the section `Running YANK <running-yank>`_ for more information on using MPI mode, including strategies for dealing with machines containing multiple GPUs.
