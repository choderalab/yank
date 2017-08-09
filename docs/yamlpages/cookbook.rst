.. _yaml_cookbook_head:

Cookbook for YAML Options
*************************

Having all the options laid out in front of you is good for advanced users, but sometimes practical examples are much more helpful.
Below is the YANK YAML Cookbook:
A series of examples that may help you understanding how to put together the options found on the other pages.

----

Best Practices Shown by These Examples
======================================

The following examples exemplify some of the `best practices <http://www.alchemistry.org/wiki/Best_Practices>`_ laid out on
`alchemistry.org <http://www.alchemistry.org/>`_. Specifically:

* There are no partial atomic charges on an atom while its Lennard-Jones interactions are being removed.
* The whole free energy setup and run is automated through YANK's YAML format, meaning its easily transferable and repeatable
* A statistically efficient alchemical path is selected because the ``softcore_X`` parameters were not set, so the default,
  efficient, soft core parameters were chosen automatically.



.. _yaml_ex_implicit:

Example: Implicit Solvent Binding Simulation:
=============================================

This example sets up para-xylene binding to T4-Lysozyme in implicit solvent.
We recommend these settings as a good baseline simulation setup.
Although you do not have to set all these options (e.g. the ligand may be a pdb file, not just a SMILES string), many 
of these options are good stock settings to remember.

In this example:

* Targeting folders for input files
* Configuring Output files
* Setting good stock options for implicit simulations
* Setting up simple protein/ligand binding simulation
* NVT ensemble


.. code-block:: yaml

  options:
    verbose: yes
    setup_dir: setup
    output_dir: output
    experiments_dir: experiments
    randomize_ligand: yes
    minimize: yes
    number_of_iterations: 2000
    temperature: 300*kelvin
    pressure: null
  
  molecules:
    T4_lysozyme:
      filepath: T4.pdb
    p-xylene:
      smiles: CC1=CC=C(C=C1)C
      antechamber:
        charge_method: bcc
    
  solvents:
    GBSA:
      nonbonded_method: NoCutoff
      implicit_solvent: OBC2
      solvent_dielectric: 78.5

  systems:
    T4-xylene-complex:
      receptor: T4_lysozyme
      ligand: p-xylene
      solvent: GBSA
      pack: yes
      leap:
        parameters: [leaprc.ff14SB, leaprc.gaff]

  protocols:
    absolute-binding:
      complex:
        alchemical_path:
          lambda_electrostatics: [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
          lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.00]
      solvent:
        alchemical_path:
          lambda_electrostatics: [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
          lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.00]

  experiments:
    system: T4-xylene-complex
    protocol: absolute-binding
    restraint:
      type: FlatBottom
                
After this is all setup, simply run: ``yank script --yaml={ThisScriptName.yaml}`` and YANK will take care of the rest for you.


----

.. _yaml_ex_explicit:

Example: Absolute Binding free energy in explicit solvent
=========================================================

This example takes the same para-xylene binding to T4-Lysozyme system as before, but now uses an explicit solvent setup, 
minimal options, and automatic water addition (TIP3P).

This example also shows how to make YANK run with MPI; assumes 4 nodes are available.
It should be noted there is nothing you set in the YAML file or with YANK itself to run with MPI.
YANK automatically detects if MPI was called to run YANK and interacts with it accordingly.

In this Example:

* Automatic solvent addition
* Setting good stock options for explicit simulations
* Call MPI
* NPT ensemble

.. code-block:: yaml

   options:
     minimize: yes
     verbose: yes
     output_dir: .
     number_of_iterations: 2000
     temperature: 300*kelvin
     pressure: 1*atmosphere

    molecules:
      t4-lysozyme:
        filepath: setup/receptor.pdbfixer.pdb
        parameters: leaprc.ff14SB
      p-xylene:
        filepath: setup/ligand.tripos.mol2
        antechamber:
          charge_method: bcc

  solvents:
    PME:
      nonbonded_method: PME
      nonbonded_cutoff: 0.9*nanometer
      switch_distance: 0.8*nanometer
      clearance: 12*angstroms
      positive_ion: Na+
      negative_ion: Cl-

  systems:
    t4-xylene-explicit:
      receptor: t4-lysozyne
      ligand: p-xylene
      solvent: PME
      leap:
        parameters: [leaprc.ff12, leaprc.gaff]

  protocols:
    absolute-binding:
      complex:
        alchemical_path:
          lambda_electrostatics: [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
          lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.00]
      solvent:
        alchemical_path:
          lambda_electrostatics: [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
          lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.00]

  experiments:
    system: t4-xylene-explicit
    protocol: absolute-binding
    restraint:
      type: Harmonic

Now run: 

.. code-block:: bash

  $ build_mpirun_configfile "yank script --yaml=yank.yaml"
  $ mpiexec -f hostfile -configfile configfile

The ``build_mpirun_configfile`` is a command available if you have installed YANK through conda, and though the ``clusterutils``
repo.


.. .. _yaml_raw_examples:

    Raw YAML File Examples
    ======================

    .. literalinclude:: ../yank-yaml-cookbook/all-options.yaml
          :language: yaml

    |

    .. literalinclude:: ../yank-yaml-cookbook/combinatorial-experiment.yaml
          :language: yaml
