.. _quickstart:

Quickstart (for the impatient)
******************************

First, install the `anaconda <https://store.continuum.io/cshop/anaconda/>`_ scientific Python distribution.

Install the release version of YANK from the `omnia Anaconda Cloud channel <https://anaconda.org/omnia/yank>`_ (check out our detailed `installation <installation>`_ section):

.. code-block:: bash

   $ conda install -c conda-forge -c omnia yank

Go to the ``examples/p-xylene-implicit`` directory to find an example of computing the binding affinity of p-xylene to T4 lysozyme L99A:

.. code-block:: bash

   $ cd ~/anaconda/share/yank/examples/p-xylene-implicit

Run an alchemical free energy calculation (in serial mode) using the parameters specified in the ``yank.yaml`` file:

.. code-block:: bash

   $ yank script --yaml=yank.yaml

Alternatively, the simulation can be :doc:`run in MPI mode <running>` if you have multiple GPUs available.

Analyze the simulation data:

.. code-block:: bash

   $ yank analyze --store=experiments
