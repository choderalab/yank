#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Tools for constructing OpenMM System objects from various convenient input file formats.

DESCRIPTION

This module contains tools for constructing OpenMM System objects for receptor-ligand systems
from various convenient input formats, and includes the ability to form models of complexes
from individual receptor and ligand files.

"""

#=============================================================================================
# GLOBAL DEFINITIONS
#=============================================================================================

__author__ = 'Patrick Grinaway'

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import abc
import copy
import pdbfixer
import mdtraj
import numpy as np
import simtk.openmm.app as app
import simtk.unit as units
import gaff2xml

#=============================================================================================
# ABSTRACT BASE CLASS
#=============================================================================================

class SystemBuilder():
    """
    Abstract base class for SystemBuilder classes.

    This is a base class for other classes that will create systems for use with Yank
    Only its children should use it. Several unimplemented methods must be overridden

    Properties
    ----------
    positions : simtk.openmm.Quantity with units compatible with nanometers, wrapping a numpy.array of natoms x 3
       Atomic positions associated with the constructed System object.
    topology : simtk.openmm.app.Topology
       OpenMM Topology specifying system.
    system : simtk.openmm.System
       The OpenMM System object created by this SystemBuilder.
    natoms : int
       The number of atoms in the system.
    ffxmls : list of str
       List of OpenMM ForceField ffxml file contents used to parameterize this System.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, ffxml_filenames=None, ffxmls=None):
        """
        Abstract base class for SystemBuilder classes.

        Parameters
        ----------
        ffxml_filenames : list of str, optional, default=None
           List of OpenMM ForceField ffxml filenames used to parameterize the System.
        ffxmls : list of str, optional, default=None
            List of ffxml file contents used to parameterize the System.

        """
        
        # Set private class properties.
        self._ffxmls = ffxmls
        self._append_ffxmls(ffxml_filenames)

        self._topology = None # OpenMM Topology object
        self._positions = None # OpenMM positions as simtk.unit.Quantity with units compatible with nanometers
        self._system = None # OpenMM System object created by ForceField

        return

    def _append_ffxmls(self, ffxml_filenames):
        """
        Read specified ffxml files and append to internal _ffxml structure.

        Parameters
        ----------
        ffxml_filenames : list of str
           List of filenames for ffxml files to read and append to self._ffxmls.

        """
        if ffxml_filenames:
            for ffxml_filename in ffxml_filenames:
                ffxml = self._read_ffxml(ffxml_filename)
                self._ffxmls.append(ffxml)
        return

    def _create_system(self):
        """
        Create the OpenMM System object.

        """
        forcefield = app.ForceField(*self._ffxmls)
        self._system = forcefield.createSystem(self._topology)
        return

    @property
    def system(self):
        """
        Return the SystemBuilder's simtk.openmm.System object.

        A deep copy is returned to avoid accidentally modifying these objects.

        """
        if self._system is None:

            self._create_system()
        return self._system

    @property
    def natoms(self):
        """
        Return the number of particles in this sytem.

        Returns
        -------
        natoms : int
           The number of particles in this system.

        """
        return len(self._positions)

    @property
    def topology(self):
        """
        Return the SystemBuilder's OpenMM topology object.
        
        Returns
        -------
        topology : simtk.openmm.app.Toplogy
           The topology object containing all atoms in the system.

        A deep copy is returned to avoid accidentally modifying these objects.

        """
        return copy.deepcopy(self._topology)

    @property
    def positions(self):
        """
        Return atomic positions in OpenMM format.

        Returns
        -------
        positions : simtk.unit.Quantity with units compatible with nanometers, wrapping natoms x 3 numpy array
           Atomic positions.

        A deep copy is returned to avoid accidentally modifying these objects.

        """
        return copy.deepcopy(self._positions)

    @property
    def forcefield(self):
        """
        Return the associated ForceField object.

        Returns
        --------
        forcefield : simtk.openmm.app.ForceField
           The ForceField object associated with this class.  One is created if it does not yet exist.

        A deep copy is returned to avoid accidentally modifying these objects.
           
        """
        return copy.deepcopy(self._forcefield)

    @property
    def ffxmls(self):
        """
        Return the list of OpenMM forcefield definition (ffxml) files associated with this SystemBuilder object.

        Returns
        -------
        ffxmls : list of str
           The list of OpenMM ffxml file contents associated with this object.

        A deep copy is returned to avoid accidentally modifying these objects.

        """
        return copy.deepcopy(self._ffxmls)

#=============================================================================================
# ABSTRACT BASE SMALL MOLECULE BUILDER
#=============================================================================================

class SmallMoleculeBuilder(SystemBuilder):
    """
    class SmallMoleculeBuilder:
        This is a base class for SystemBuilders that will handle small molecules. In general, this means that forcefield
        parameters may have to be generated. It does not necessarily imply that this

    TODO:
    * Allow specification of charge assignment method.
    * Support additional small molecule parameterization methods (e.g. oeante, paramchem, ATB)

    """

    def __init__(self, coordinate_file, molecule_name, forcefield_files=None, **kwargs):
        """

        SmallMoleculeBuilder Constructor. This is a stub.
        It calls the base constructor, then checks to see if there are any forcefield files. If not,
        it will call the build_forcefield() method using the method specified in parameterization_method

        Parameters
        ----------
        coordinate_file : str
           Coordinate file specifying the small molecule (e.g. PDB, mol2, SDF)
        molecule_name : str
           Name of the molecule.
        forcefield_files : list of str
           List of OpenMM ffxml files that contain parameters for the molecule.
           If None, these files will be constructed using the specified parameterization method.

        """
        super(SmallMoleculeBuilder, self).__init__()
        if forcefield_files is None:
            self.build_forcefield(**kwargs)

    @abc.abstractmethod
    def build_forcefield(self, **kwargs):
        """
        Build a forcefield ffxml file from the molecule using the specified method.

        Takes coordinate file and molecule name from member variables
        Must be overridden by child classes

        """
        pass

#=============================================================================================
# ABSTRACT BIOPOLYMER SYSTEM BUILDER 
#=============================================================================================

class BiopolymerSystemBuilder(SystemBuilder):
    """
    Abstract base class for classes that will read biopolymers.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        Abstract base class for classes that will read biopolymers.

        Parameters
        ----------
        coordinate_file : str
           Filename specifying biopolymer.
        molecule_name : str
           Name of the biopolymer.
        forcefield_files : list of str
           ForceField ffxml files.
        chain_ids : list of str, optional, default=None
           List of which chains to include, or None if all chains are to be included.
           e.g. 'ABC'
           
        """
        super(BiopolymerSystemBuilder, self).__init__()

#=============================================================================================
# BIOPOLYMER SYSTEM BUILDER FOR PDB FILES
#=============================================================================================

class BiopolymerPDBSystemBuilder(BiopolymerSystemBuilder):

    """
    BiopolymerPDBSystemBuilder:

    This class is a subclass of BiopolymerSystemBuilder, and uses a PDB file as input. Currently, it accepts proteins and uses PDBFixer to try to repair issues with the PDB file
    This class does not currently perform any parameterization. As such, failure to give it appropriate forcefield ffxml(s) will cause it to fail.

    """

    def __init__(self, pdb_filename, chain_ids=None, ffxml_filenames=['amber99sb_ildn.xml', 'amber99_obc.xml'], pH=7.0):
        """
        Create a biopolymer from a specified PDB file.

        Parameters
        ----------
        pdb_filename : str
           PDB filename
        forcefield_files : list of str, default=None
           List of OpenMM ForceField ffxml files used to parameterize the System.
        chain_ids : list of str, default=None
           List of chain IDs that should be kept.
        pH : float, default 7.0
           pH to be used in determining the protonation state of the biopolymer.

        """
        # Call base constructor.
        super(BiopolymerPDBMoleculeBuilder, self).__init__(ffxml_filenames=ffxml_filenames)

        # Store the desired pH.
        self._pH = pH
        
        # Use PDBFixer to add missing atoms and residues and set protonation states appropriately.
        fixer = pdbfixer.PDBFixer(self._coordinate_file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.removeHeterogens(True)
        fixer.addMissingHydrogens(self._pH)
        self._fixer = fixer

        # Keep only the chains the user wants
        if self._chains_to_use is not None:
            # TODO: Check correctness.
            n_chains = len(list(fixer.topology.chains()))
            chains_to_remove = np.setdiff1d(np.arange(n_chains), self.keep_chains)
            fixer.removeChains(chains_to_remove)
            
        # Store OpenMM topology.
        self._topology = fixer.topology
        
        # Store OpenMM positions.
        self._positions = fixer.positions

        return

    @property
    def pH(self):
        """
        The pH used to determine the protonation state of the biopolymer.

        Returns
        -------
        pH : float
           The pH that was used to determine the protonation state.

        """
        return self._pH

#=============================================================================================
# SMALL MOLECULE SYSTEM BUILDER FOR MOL2 FILES
#=============================================================================================

class Mol2SystemBuilder(SmallMoleculeBuilder):
    """
    Create a system from a small molecule specified in a Tripos mol2 file.

    TODO:
    * Clean up temporary ffxml file on __del__()
    """

    def __init__(self, coordinate_file, molecule_name, forcefield_files=None, parameterization_method="antechamber", gaff_mol2_file=None):
        """
        Create a system from a Tripos mol2 file.

        Parameters
        ----------
        coordinate_file : str
           Small molecule coordinate file in Tripos mol2 format.
        molecule_name : str
           Name of the molecule.
        forcefield_file  : str, optional, default=None
           ffxml file that contains parameters for the molecule. If none, will be generated by the parameterization method specified.
        parameterization_method : str, optional, default='antechamber'
           The method used to parameterize the molecule. One of ['antechamber'].
        gaff_mol2_file : str
           A parsed gaff mol2 filename, if available.

        """
        self._gaff_mol2 = gaff_mol2_file
        super(Mol2SystemBuilder, self).__init__(coordinate_file, molecule_name, forcefield_files=forcefield_files, parameterization_method=parameterization_method)

    def build_forcefield(self, param_method="antechamber", **kwargs):
        """
        This function overrides the parent's method, and builds a forcefield ffxml file for the given mol2 using gaff2xml (which in turn runs antechamber)

        Parameters
        ----------        
        param_method : str
           Method to use to parameterize molecule; currently only antechamber is supported.
        **kwargs
           Used to allow extra parameters to be passed to parameterization scheme.

        """
        
        # Set charge method to 'bcc' if not specified.
        if 'charge_method' not in kwargs:
            kwargs['charge_method'] = 'bcc'

        # Run antechamber via gaff2xml to generate charges.
        (gaff_mol2_filename, gaff_frcmod_filename) = gaff2xml.utils.run_antechamber(self._molecule_name, self._coordinate_file, **kwargs)

        # Write out the ffxml file from gaff2xml.
        ffxml_filename = tempfile.NamedTemporaryFile(delete=False)
        gaff2xml.utils.create_ffxml_file(gaff_mol2_filename, gaff_frcmod_filename, ffxml_filename)

        # Store name of generated gaff format mol2 file.
        self._gaff_mol2 = gaff_mol2_filename
        
        # Append forcefield files.
        if self._forcefield_files is None:
            self._forcefield_files = list()
            self._forcefield_files.append(ffxml_filename)
        pass

    def _create_system(self):
        """
        This function overrides the parent's _create_system in order to create a simtk.openmm.System object representing the molecule of interest.

        TODO
        * Store mdtraj object, openmm positions, mdtraj topology, and openmm topology.

        """
        # Load the mol2 file to create an MDTraj object.
        mol2 = mdtraj.load(self._gaff_mol2)
        positions = mol2.openmm_positions(0)
        topology = mol2.top.to_openmm()
        forcefield = app.ForceField(*self._forcefield_files)
        self._system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)

    def _create_traj(self):
        """
        """
        self._mdtraj = mdtraj.load(self._gaff_mol2)

#=============================================================================================
# SYSTEM BUILDER FOR COMBINING RECEPTOR AND LIGAND INTO A COMPLEX
#=============================================================================================

class ComplexSystemBuilder(SystemBuilder):

    def __init__(self, ligand, receptor, remove_ligand_overlap=False):
        """
        Create a new SystemBuilder for a complex of a given ligand and receptor, keeping track of which atoms belong to which.

        Parameters
        ----------
        ligand : SystemBuilder
           The SystemBuilder representing the ligand
        ligand : SystemBuilder
           The SystemBuilder representing the ligand
        remove_ligand_overlap : bool, optional, default=False
           If True, will translate ligand to not overlap with receptor atoms.

        Properties
        ----------
        ligand_atoms : list of int
           List of atoms representing the ligand.
        receptor_atoms : list of int
           List of atoms representing the receptor.

        """

        # Call base class constructor.
        super(ComplexSystemBuilder, self).__init__()

        # Append ffxml files.
        self._ffxmls.append(receptor.ffxmls)
        self._ffxmls.append(ligand.ffxmls)

        # Concatenate topologies and positions.
        from simtk.openmm import app
        model = app.modeller.Modeller(receptor.topology, receptor.positions)
        model.add(ligand.topology, ligand.positions)
        self._topology = model.topology
        self._positions = model.positions

        # Store indices for receptor and ligand.
        self._ligand_atoms = range(receptor.natoms)
        self._receptor_atoms = range(receptor.natoms, receptor.natoms + ligand.natoms)

        # Modify ligand coordinates to not overlap with receptor.
        if remove_ligand_overlap:
            self._remove_ligand_overlap()

        return

    def _remove_ligand_overlap(self):
        """
        Translate the ligand so that it is not overlapping with the receptor.

        Description
        -----------
        The bounding sphere of the ligand and receptor are computed, and the ligand translated along the x-direction to not overlap with the protein.
        
        TODO:
        * This is not guaranteed to work for periodic systems.

        """

        # Create an mdtraj Topology from OpenMM Topology object.
        mdtraj_topology = mdtraj.Topology.from_openmm(self._topology)

        # Create an mdtraj instance.
        mdtraj = mdtraj.Trajectory(self._positions, mdtraj_topology)

        # Compute centers of receptor and ligand.
        receptor_center = mdtraj.xyz[0][self._receptor_atoms,:].sum(1)
        ligand_center = mdtraj.xyz[0][self._ligand_atoms,:].sum(1)

        # Compute radii of receptor and ligand.
        receptor_radius = ((ligand_traj.xyz[0][self._receptor_atoms,:] - receptor_center) ** 2.).sum(1) ** 0.5).max()
        ligand_radius = ((ligand_traj.xyz[0][self._ligand_atoms,:] - receptor_center) ** 2.).sum(1) ** 0.5).max()

        # Translate ligand along x-axis from receptor center with 5% clearance.
        mdtraj.xyz[0][self._ligand_atoms,:] += np.array([1.0, 0.0, 0.0]) * (receptor_radius + ligand_radius) * 1.05 - ligand_center + receptor_center

        # Extract system positions.
        self._positions = mdtraj.openmm_positions(0)

        return

#=============================================================================================
# TEST CODE
#=============================================================================================

if __name__=="__main__":
    import os
    import simtk.unit as unit
    import simtk.openmm as openmm
    import numpy as np
    import simtk.openmm.app as app
    import alchemy
    os.environ['AMBERHOME']='/Users/grinawap/anaconda/pkgs/ambermini-14-py27_0'
    os.chdir('/Users/grinawap/driver/yank/examples/px-test')
    ligand = Mol2SystemBuilder('ligand.tripos.mol2', 'ligand')
    receptor = BiopolymerPDBSystemBuilder('receptor.pdb','protein')
    complex_system = ComplexSystemBuilder(ligand, receptor, "complex")
    complex_positions = complex_system.positions
    receptor_positions = receptor.positions
    print type(complex_system.coordinates_as_quantity)
    timestep = 2 * unit.femtoseconds # timestep
    temperature = 300.0 * unit.kelvin # simulation temperature
    collision_rate = 20.0 / unit.picoseconds # Langevin collision rate
    minimization_tolerance = 10.0 * unit.kilojoules_per_mole / unit.nanometer
    minimization_steps = 20
    plat = "CUDA"
    i=2
    platform = openmm.Platform.getPlatformByName(plat)
    forcefield = app.ForceField
    systembuilders = [ligand, receptor, complex_system]
    receptor_atoms = range(0,2603)
    ligand_atoms = range(2603,2621)
    factory = alchemy.AbsoluteAlchemicalFactory(systembuilders[i].system, ligand_atoms=ligand_atoms)
    protocol = factory.defaultComplexProtocolImplicit()
    systems = factory.createPerturbedSystems(protocol)
    integrator_interacting = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    #test an alchemical intermediate and an unperturbed system:
    fully_interacting = app.Simulation(systembuilders[i].traj.top.to_openmm(),systems[0], integrator_interacting, platform=plat)
    fully_interacting.context.setPositions(systembuilders[i].openmm_positions)
    fully_interacting.minimizeEnergy(tolerance=10*unit.kilojoule_per_mole)
    fully_interacting.reporters.append(app.PDBReporter('fully_interacting.pdb', 10))
    for j in range(10):
        print str(j)
        fully_interacting.step(100)
    del fully_interacting


    for p in range(1, len(systems)):
        print "now simulating " + str(p)
        integrator_partialinteracting = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        partially_interacting = app.Simulation(systembuilders[i].traj.top.to_openmm(),systems[p], integrator_partialinteracting, platform=plat)
        partially_interacting.context.setPositions(systembuilders[i].openmm_positions)
        partially_interacting.minimizeEnergy(tolerance=10*unit.kilojoule_per_mole)
        partially_interacting.reporters.append(app.PDBReporter('partial_interacting'+str(p)+'.pdb', 10))
        for k in range(10):
            print str(k)
            partially_interacting.step(100)
        del partially_interacting

