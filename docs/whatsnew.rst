***************
Release history
***************

This section features and improvements of note in each release.

The full release history can be viewed `at the GitHub yank releases page <https://github.com/choderalab/yank/releases>`_.

0.20.0 Mutations and PDBFixer
-----------------------------
- Protein Mutations can be specified through PDBFixer calls
- Generalized PDBFixer options through the YAML molecules with `pdbfixer` directive

0.19.4 Schema and Parallel Setup Fixes
--------------------------------------
- Fixed bug in parallel molecule setup which caused the same molecule to be setup multiple times.
- Fixed bug in Cerberus schema for LEaP where molecule parameters accumulated.
- Fixed bug where options in experiment section were not coerced.
- Fixed status command to print information about all combinatorial experiments.
- Faster restart with combinatorial experiments.

0.19.3 Support for Amber restart files
--------------------------------------
- Added support for Amber ``rst7`` files in ``phase1_path``/``phase2_path``.
- The CLI option ``jobid`` now uses 1-based numbering like Torque and LSF do for array jobs.

0.19.2 Include ions in solute-only trajectory
---------------------------------------------
- Ions are now included in the solute-only trajectories.

0.19.1 Trailblaze fix and restart stability from OpenMMTools
------------------------------------------------------------
- OpenMMTools 0.13.4 now required to fix issues listed below
- Restrained atoms to absolute coordinates caused issue in Trailblaze
  with a Barostat
- Last restart attempt uses a slower, but more robust restart method

0.19.0 Regions, Cerberus, and Errors
------------------------------------
- Added custom region selection to Topography
- Custom regions can now be defined through YAML
- Compound custom Topography regions can now be selected
- Restraints atom selection can now use Topography Regions
- Topography now can select from arbitrary string, either complex regions, DSL strings, and in the future SMARTS strings
- Changed to Cerberus for data validation (was Schema), public facing validation schemas in the future
- Added better error handling of known LEaP Errors
- Fixed issue for ``start_frame`` and ``end_frame`` were ignored for trajectory extraction
- OpenMMTools 0.13.3 now required to fix bug in ``SamplerState``

0.18.0 Python 2 Dropped, Solute Only Trajectories, and Trailblaze Bugfixes
--------------------------------------------------------------------------
- Python 2.X Support officially *removed*
- Additional doc cleanups
- Added restraint selection flowchart to documentation
- Implement #772: Use infinity instead of None to specify unlimited number of iterations.
- Implemented #557: Parallelized setup of molecules and systems with MPI.
- Generalized restrained atoms selection during trailblaze scheme to include non-protein receptors (see also choderalab/openmmtools#290).
- Fix loading of leap parameters from a local .dat files (allow us to use local versions of gaff parameters for validation).
- Fix #762: Trailblaze protocol crashes with MPI.
- Fixed bug when computing reduced potentials of simulated energies during trailblaze scheme.
- Fix #763: Automatic path is saved in YAML as a mix of python and numpy floats.
- Fixed the number of neutralizing counterions when receptor and ligand have opposite charges (we were adding too many in this case).
- Fixed the log file name with lists of experiments that ended up being just .log.
- Implemented workaround for fixing the net charge of cyclic multi-residue mol2 files.
- Added GAFF2 Torsion support based on YAML input files
- Solute-only trajectories can now be stored every iteration, regardless of checkpoint interval

0.17.0 Auto Alchemical Path and Split Langevin Integrators
----------------------------------------------------------
- Added Langevin Splitting Integrator which allows time-substep operation order
- Automatic Alchemical Path selection feature added.
- Many Website additions and cleanups
- Online analysis allowing simulations to be run until they reach a target free energy uncertainty
- Renamed and refactored ``YAMLBuilder`` to more general ``ExperimentBuilder``
- Remove ligand rotation and displacement with Boresch restraints to improve acceptance rates
- Analyze module fully tested now
- Fully updated API docstrings. API auto-generated on website
- Parallelize multiple experiments over MPI by splitting MPI Communicator
- Anisotropic dispersion options in YAML reduced to single option
- Ionic Strength ability added to setup pipeline
- Centroids for restraints now selectable through DSL string instead of whole molecule
- Added MDTraj, Matplotlib, and Jupyter as requirements
- Analyze Jupyter Notebooks can now be exported as pre-rendered static HTML or PDF pages (LaTeX required for PDF)
- Refactor some API function names and keywords

0.16.2 Startup Speed and Reduced File Sizes
-------------------------------------------
- Automatic Expanded Cutoff Distance Selection
- Compressed stored systems drastically reduce initial file sizes
- Use C Yaml Dumper and Loaders to speed up YAML object processing
- Requires OpenMMTools 0.11.2 at minimum

0.16.1 Auto Expanded Cutoffs and bug fixes for Transition Matrix and Reporter
-----------------------------------------------------------------------------
- Expanded cutoff now able to be chosen automatically instead of just hard coded number
- Fixes bug causing transition matrix to be computed incorrectly, uses empirical to estimate
- Allows user to drop samples equilibration report to avoid plot scale being dominated by initial fast equilibration

0.16.0 Full API and Python 3.6
------------------------------
- Full feature API for setting up, running, and analyzing experiments
- Supports new generalized MCMC moves, ThermodynamicStates, and other features from improved OpenMMTools
- Checkpoint feature added to reduce file size, add portability to data analysis files.
- Simulations can now alternate between phases to allow analysis even before simulations are done
- OpenEye features compartmentalized so you don't need every OpenEye feature YANK could use to use any of them
- Major under the hood speed ups to base code and MPI behavior, includes a full code refactor.
- Mol2 files can now read in multi-molecule files
- No longer uses standalone Alchemy module, uses the one built into OpenMMTools
- Added Python 3.6 support.
- Retired Python 3.4 support

0.15.2 Health Report and Anisotropic Dispersion Control
-------------------------------------------------------
- Added simulation Health Report through a Jupyter Notebook with CLI support
- Added ability to control Anisotropic Dispersion Correction through YAML files

0.15.0 Backend and Helpful Debugging Build
------------------------------------------
- Added support for ``solvent_dsl`` in user defined systems of YAML pages
- Removed Command Line Interface ability to do ``yank prepare`` and ``yank run``
- Added ability to overwrite individual YAML commands from command line
- Added YAML feature to ``extend_simulation`` without modifying YAML files or command line every iteration
- NaN's generated during simulations serialize system, state, and integrator which can be passed off for debugging to others
- Backend website updating and pushes improved
- Improved GROMACS extension file handling

0.14.1 Early Access of 1.0 Release
----------------------------------
- YAML Syntax Structure Frozen. YANK YAML Version 1.0. All YAML scripts from this version will be compatible with future versions until YAML 2.0
  New features may appear in the time meantime, but scripts will be forwards compatible.
- Initial support for OpenMM XML systems and PDB files
- Support for separate solvent configurations for the two phases when defined from amber/gromacs/openmm files
- ``clearance`` in YAML now mandatory parameter of explicit solvent, but only when molecule setup goes through pipeline
- Boresch Orientational Restraints fully implemented and documented.
- Long range anisotropic dispersion correction improved to work on both ends of thermodynamic cycle leg
- Documentation updated with better algorithms and theory sections.
- Full walkthroughs of ``yank-examples`` added to online documentation
- Various other documentation improvements
- Support for upcoming OpenMM 7.1 Release and features (still works with 7.0.1)
- YANK now on MIT License
- Many bugfixes

0.12.0 (development)
--------------------
- Examples split into their own repository
- Old CLI commands staring depreciation

0.11.2 (development)
--------------------
- Better long range dispersion and electrostatics corrections
- Best practices and guidelines for the YAML documentation published

0.11.0 (development)
--------------------
- Full YAML documentation available online with all possible options specified
- Developer documentation

0.10.0 (development)
--------------------
- Python 3.X support
- Online documentation has been updated to include the YAML input files
- Selftests now provide more helpful output


0.9.0 (development)
-------------------
- Changed YAML Syntax
- New Command ``yank analyze extrat-trajectory`` to extract data from NetCDF4 file in a common trajectory format.
- Support for solvation free energy calculations.
- Automatic detection of MPI.
- Various bug fixes.

0.8.0 (development)
-------------------
- ``alchemy`` split to a standalone repository
- YAML based input files for setting up and running simulations. Uses an AmberTools-based pipeline

0.7.0 (development)
-------------------
- Convert to single ``Context`` Hamiltonian Replica Exchange

v0.6.1 (development)
--------------------
- mpi4py automatically installed via conda

v0.6.0 (development)
--------------------
- New command-line interface
- Sphinx-based documentation

v0.5.0 (development)
--------------------
- Release for deployment to collaborators

