yank
====

YANK: GPU-accelerated calculation of ligand binding affinities in implicit and explicit solvent using alchemical free energy methodologies.

Description
-----------

YANK uses a sophisticated set of algorithms to rigorously compute (within a classical statistical mechanical framework) biomolecular ligand binding free energies.  This is accomplished by performing an *alchemical free energy calculation* in either implicit or explicit solvent, in which the interactions between a ligand (usually a small molecule or peptide) are decoupled in a number of *alchemical intermediates* whose interactions with the environment are modified, creating an alternative thermodynamic cycle to the direct dissociation of ligand and target biomolecule.  The free energy of decoupling the ligand from the environment is computed both in the presence and absence of the biomolecular target, yielding the overall binding free energy (in the strong single-binding-mode framework of Gilson et al.) once a standard state correction is applied to correct for the restraint added between ligand and biomolecule in the complex.

Computation of the free energy for each leg of the thermodynamic cycle utilized a modified replica-exchange simulations in which exchanges between alchemical states are permitted (Hamiltonian exchange) and faster mixing is achieved by using a Gibbs sampling framework.

Authors
-------

* John D. Chodera | jchodera@gmail.com
* Kim Branson | kim.branson@gmail.com
* Imran Haque | ihaque@gmail.com
* Michael Shirts | mrshirts@gmail.com

Copyright
---------

Portions of this code copyright (c) 2009-2011 University of California, Berkeley, Vertex Pharmaceuticals, Stanford University, University of Virginia, Memorial Sloan-Kettering Cancer Center, and the Authors.

Prerequisites
-------------

Use of this module requires the following

* AmberTools (for setting up protein-ligand systems):
  http://ambermd.org/#AmberTools

* OpenMM with Python wrappers: 
  http://simtk.org/home/openmm

* Python 2.6 or later: 
  http://www.python.org

* NetCDF (compiled with netcdf4 support):
  http://www.unidata.ucar.edu/software/netcdf/

* HDF5 (required by NetCDF4): 
  http://www.hdfgroup.org/HDF5/

* netcdf4-python (a Python interface for netcdf4):
  http://code.google.com/p/netcdf4-python/

* numpy and scipy:
  http://www.scipy.org/

* mpi4py (if MPI support is desired):
  http://mpi4py.scipy.org/
  (Note that the mpi4py installation must be compiled against the appropriate MPI implementation.)

* OpenEye toolkit and Python wrappers (if mol2 and PDB reading features are used ;requires academic or commercial license):
  http://www.eyesopen.com

Simplified Python prerequisite installation
-------------------------------------------

The Enthought Python Distribution (EPD) provides many of these prerequisites (including Python, NetCDF 4, HDF5, netcdf4-python, numpy, and scipy): http://www.enthought.com/products/epd.php

*Note that using EPD with OpenEye requires some care, as OpenEye tools are very selective about which Python and library versions are compatible.*

For example, to use EPD 7.1-2 on OS X with OpenEye's latest toolkit, install OpenEye's toolkit and Python wrappers, then:

    # Change to OpenEye libs directory
    cd /path/to/openeye/python/openeye/libs

    # Create a symlink for your EPD platform to trick OpenEye into thinking it is supported.
    ln -s osx-10.7-g++4.2-x64-python2.7 osx-10.7-g++4.0-x64-python2.7

Running YANK from the command line
----------------------------------

    python yank.py --ligand_prmtop PRMTOP --receptor_prmtop PRMTOP --complex_prmtop PRMTOP { {--ligand_crd CRD | --ligand_mol2 MOL2} {--receptor_crd CRD | --receptor_pdb PDB} | {--complex_crd CRD | --complex_pdb PDB} } [-v | --verbose] [-i | --iterations ITERATIONS] [-o | --online] [-m | --mpi] [--restraints restraint-type]

EXAMPLES:

Serial execution:

    # Specify AMBER prmtop/crd files for ligand and receptor.
    python yank.py --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --complex_prmtop complex.prmtop --ligand_crd ligand.crd --receptor_crd receptor.crd --iterations 1000

    # Specify (potentially multi-conformer) mol2 file for ligand and (potentially multi-model) PDB file for receptor.
    python yank.py --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --complex_prmtop complex.prmtop --ligand_mol2 ligand.mol2 --receptor_pdb receptor.pdb --iterations 1000

    # Specify (potentially multi-model) PDB file for complex, along with flat-bottom restraints (instead of harmonic).
    python yank.py --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --complex_prmtop complex.prmtop --complex_pdb complex.pdb --iterations 1000 --restraints flat-bottom

MPI execution:

    See example script mvapich2.pbs for an example using MVAPICH2.

Notes
-----

Currently, YANK only accepts AMBER "new style" `prmtop` topology files to define molecular systems.
For examples of how to set up your own systems using the free AmberTools suite, see the examples/ directory.

In these AMBER `prmtop` and `inpcrd` files, receptor atoms must come before ligand atoms.
Atom orderings must be the same in all files (AMBER prmtop/crd, PDB, mol2).
mol2 files must contain only copies of the same molecule in different geometries.

Free energy calculations in both implicit and explicit solvent are supported.  The presence of water is automatically detected.

Use the testrun.sh script as an example for serial execution, and the mvapich2.pbs script as an example of MPI execution (can be run with batch or interactive queues).

Using the YANK module from Python
---------------------------------

YANK can also be used as a module from another Python program.  See the command-line driver in '__main__' in yank.py as an example:

    # Import the YANK module:
    from yank import Yank

    # Initialize YANK object using OpenMM System objects and coordinates.
    yank = Yank(receptor=receptor_system, ligand=ligand_system, complex=complex_system, complex_coordinates=complex_coordinates, output_directory=output_directory, verbose=verbose)

    # Run the YANK simulation using serial execution; use run_mpi() method for MPI execution.
    yank.run()

    # Analyze results.
    results = yank.analyze()

Note that the analysis routines can also be run asynchronously as the YANK object is running. 

Testing
-------

Three levels of testing frameworks are provided:

* doctests

Doctests ensure that each of the individual functions that compose YANK run on valid data without throwing exceptions.  These are implemented in the '__main__' part of each module in YANK (e.g. 'repex.py'), and are regularly run to ensure that there is no invalid code in YANK.

* module tests

Module tests test that the code contained in the corresponding module (e.g. 'test_repex.py' for 'repex.py') generates the correct results for analytically-tractable test cases.  This code ensures the correctness of individual components of YANK.  Though it is impossible to test every conceivable input combination, some care is taken to ensure overall correctness of recommended codepaths.

* integration tests

Integration tests ensure that the whole of YANK run on certain test problems produce reliable free energy differences for well-characterized systems (such as harmonic oscillators, Lennard-Jones particles, etc.).  Integration tests are run from the provided 'integration_tests.py' script.

Known Issues
------------

* Running the MPI version of YANK with certain MPI implementations (mpich2 with hydra in particular) appears to hang in 'nvcc' if the CUDA platform is used.  The OpenCL platform seems to be unaffected.
* The CUDA 5.0 runtime on Linux platforms appears to allocate an enormous amount (tens of GB) of virtual memory that is not used.  This is a known bug in the CUDA runtime and is expected to be rectified in CUDA 5.5.

TODO
----

- [ ] Remove dependence on deprecated pyopenmm pure-Python wrapper; require System objects for complex, ligand, and protein instead.
- [ ] Add support for asynchronous execution of Yank.run() 
- [ ] Add support for on-the-fly analysis thread(s)
- [ ] Speed up initialization and resuming runs
- [ ] Change atom ordering so ligand is first, protein second, and solvent third.

Roadmap
-------

Support for the following is planned:

* Online analysis and automatic convergence detection/termination [in progress]
* Explicit solvent support with NPT simulations [almost ready; only waiting on analytical dispersion correction additions]
* General Markov chain Monte Carlo (MCMC) move sets in between Hamiltonian exchanges [refactoring of repex.py in progress]
* Expanded ensemble simulations (as an alternative to Hamiltonian exchange)
* Support for relative free energy calculations
* Support for sampling over protein mutations
* Generative factories, to allow searching over combinatorially large chemical spaces (both for ligand substituents and protein mutations)
* Constant-pH and ligand tautomer sampling

License
-------

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

Acknowledgents
--------------

The authors are extremely grateful to the OpenMM development team for their help
in the development of YANK, especially (but not limited to):

* Peter Eastman, Stanford University <peastman@stanford.edu>
* Mark Friedrichs <friedrim@stanford.edu>
* Vijay Pande, Stanford University <pande@stanford.edu>
* Randy Radmer
* Christopher Bruns

The developers are very grateful to the following contributors for suggesting patches, 
bugfixes, or changes that have improved YANK:

* Kai Wang, University of Virginia <kw8cr@virginia.edu>
* Christoph Klein, University of Virginia <ctk3b@virginia.edu>
* Levi Naden, University of Virginia <lnaden@virginia.edu>

Citations
---------

Please cite the following papers if you use YANK for a publication:

* YANK 

  Parton DL, Shirts MR, Wang K, Eastman P, Friedrichs M, Pande VS, Branson K, Mobley DL, Chodera JD. YANK: A GPU-accelerated platform for alchemical free energy calculations. 
  In preparation.

* OpenMM GPU-accelerated molecular mechanics library

  Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing units. 
  J. Comput. Chem. 30:864, 2009.
  http://dx.doi.org/10.1002/jcc.21209
  
  Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. 
  Comput. Sci. Eng. 12:34, 2010.
  http://dx.doi.org/10.1109/MCSE.2010.27
  
  Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. 
  J. Comput. Chem. 31:1268, 2010. 
  http://dx.doi.org/10.1002/jcc.21413
  
  Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. 
  J. Chem. Theor. Comput. 6:434, 2010.
  http://dx.doi.org/10.1021/ct900463w

  Eastman P, Friedrichs M, Chodera JD, Radmer RJ, Bruns CM, Ku JP, Beauchamp KA, Lane TJ, Wang LP, Shukla D, Tye T, Houston M, Stich T, Klein C, Shirts M, and Pande VS.  OpenMM 4: A Reusable, Extensible, 
  Hardware Independent Library for High Performance Molecular Simulation. J. Chem. Theor. Comput. 2012.  
  http://dx.doi.org/10.1021/ct300857j

* Replica-exchange with Gibbs sampling
  
  Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs sampling: Simple improvements for enhanced mixing. 
  J. Chem. Phys. 135:19410, 2011.
  http://dx.doi.org/10.1063/1.3660669

* MBAR for estimation of free energies from simulation data

  Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states. 
  J. Chem. Phys. 129:124105, 2008.
  http://dx.doi.org/10.1063/1.2978177

* Long-range dispsersion corrections for explicit solvent free energy calculations

  Shirts MR, Mobley DL, Chodera JD, and Pande VS. Accurate and efficient corrections or missing dispersion interactions in molecular simulations.
  J. Phys. Chem. 111:13052, 2007.
  http://dx.doi.org/10.1021/jp0735987

s
