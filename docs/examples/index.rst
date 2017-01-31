########
Examples
########

We make several examples available to YANK through `a standalone repository
<http://www.github.com/choderalab/yank-examples>`_ that may be helpful in constructing new applications. Each example is
broken down into the different types of calculations that YANK can do. The examples are also ordered to help users
learn the options and nuances of YANK simulations. If you are new to YANK, recommend going through the examples in order.

Installing the Examples
-----------------------
Since YANK 0.12.0, the examples exist in their own repository. You can get them from the link above, or through the
``omnia`` conda channel by

.. code-block:: bash

  conda install -c omnia yank-examples

The ``-c omnia`` is optional if you have already the omnia channel added to your conda channels


Install Location
----------------
The examples are installed in the following path:

.. code-block:: bash

  {PYTHON SOURCE DIR}/share/yank/examples

Where ``{PYTHON SOURCE DIR}`` is where your python install is located. E.g. for anaaconda/miniconda, it would be in those
program's install location, then ``./share/yank/examples``.

Uninstalling this program will also remove the files.

|

List of Examples
----------------

.. toctree::
   :maxdepth: 1

   Example 1: Absolute Binding of para-Xylene to T4-Lysozyme in Explicit Solvent <p-xylene-explicit>
   Example 2: Absolute Binding of Host cucurbit[7]uril with Guest molecule B2 in Implicit Solvent <host-guest-implicit>
   Example 3: Hydration Free energy of Phenol in Explicit Solvent <hydration-phenol-explicit>
   Example 4: Binding Free Energy of a large set of ligands for T4-Lysozyme, split by "!Combinatorial" <all-ligand-explicit>
   Example 5: Hydration Free Energy of the FreeSolv database (subset) <freesolv-imp-exp>
