.. _quickstart:

Quickstart (for the impatient)
******************************

First, install the `anaconda <https://store.continuum.io/cshop/anaconda/>`_ scientific Python distribution.

Install the release version of YANK from the `omnia binstar channel <https://binstar.org/omnia/yank>`_:

.. code-block:: none

   $ conda install -c https://conda.binstar.org/omnia yank

Go to the ``examples/p-xylene-implicit`` directory to find an example of computing the binding affinity of p-xylene to T4 lysozyme L99A:

.. code-block:: none

   $ cd ~/anaconda/share/yank/examples/p-xylene-implicit

Use the `yank prepare` command to set up an alchemical free energy calculation using the ``OBC2`` implicit solvent model, specifying the residue named ``BEN`` as the ligand, selecting the temperature of 300 Kelvin, and using the diretory ``output/`` to write out data:

.. code-block:: none

   $ yank prepare binding amber --setupdir=setup --ligand="resname BEN" --store=output --iterations=1000 \
     --restraints=harmonic --gbsa=OBC2 --temperature=300*kelvin --verbose

Run the simulation in serial mode with:

.. code-block:: none

   $ yank run --store=output --verbose

Alternatively, run the simulation in MPI mode:

.. code-block:: none

   $ yank run --store=output --mpi --verbose

Note that, in MPI mode, the default device id for each GPU is used.
See the section `Running YANK <running-yank>`_ for more information on using MPI mode, including strategies for dealing with machines containing multiple GPUs.
