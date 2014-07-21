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
import os, os.path
import tempfile

import numpy as np

import simtk.openmm.app as app
import simtk.unit as units

#=============================================================================================
# ABSTRACT BASE CLASS
#=============================================================================================

class SystemBuilder():
    """
    Abstract base class for SystemBuilder classes.

    This is a base class for other classes that will create systems for use with Yank
    Only its children should use it. Several unimplemented methods must be overridden

    To be in a valid and consistent state, each subclass must have defined the following properties
    after initialization: _positions, _topology, _ffxmls

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
    system_creation_parameters : dict of str
       Key-value pairs passed to ForceField.createSystem() when system is created.

    TODO:
    * Make system_creation_parameters private and use setter/getters that only allow changes to dict before self._system has been created.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, ffxml_filenames=None, ffxmls=None, system_creation_parameters=dict(), molecule_name="MOL"):
        """
        Abstract base class for SystemBuilder classes.

        Parameters
        ----------
        ffxml_filenames : list of str, optional, default=None
           List of OpenMM ForceField ffxml filenames (relative or absolute) used to parameterize the System.
        ffxmls : list of str, optional, default=None
            List of ffxml file contents used to parameterize the System.
        system_creation_parameters : dict of str, optional, default=None
            If specified, these key-value parameters are used in the creation of the System object via ForceField.createSystem().
            If None, an empty dict is used.

        """

        if ffxmls is None: ffxmls = set() # empty set

        # Set private class properties.
        self._ffxmls = ffxmls # Start with contents of any specified ffxml files.
        self._append_ffxmls(ffxml_filenames) # Append contents of any ffxml files to be read.
        self._molecule_name = molecule_name # Optional molecule name
        self._topology = None # OpenMM Topology object
        self._positions = None # OpenMM positions as simtk.unit.Quantity with units compatible with nanometers
        self._system = None # OpenMM System object created by ForceField

        self.system_creation_parameters = system_creation_parameters # dictionary of parameters passed to ForceField.createSystem()

        return

    def _read_ffxml(self, ffxml_filename):
        """
        Read the contents of the specified ffxml file, using relative or absolute paths.

        Parameters
        ----------
        ffxml_filename : str
           An XML file defining the force field.  Each entry may be an absolute file path, a path relative to the
           current working directory, or a path relative to this module's data subdirectory (for built in force fields).

        TODO:
        * Also try append .xml or .ffxml extensions.

        """
        try:
            infile = open(ffxml_filename, 'r')
        except IOError:
            from simtk.openmm.app import forcefield
            forcefield_data_dir = os.path.join(os.path.dirname(app.forcefield.__file__), 'data')
            fullpath = os.path.join(forcefield_data_dir, ffxml_filename)
            infile = open(fullpath, 'r')

        ffxml = infile.read()
        infile.close()
        return ffxml

    def _append_ffxmls(self, ffxml_filenames):
        """
        Read specified ffxml files and append to internal _ffxml structure.

        Parameters
        ----------
        ffxml_filenames : list of str
           A list of XML files defining the force field.  Each entry may be an absolute file path, a path relative to the
           current working directory, or a path relative to this module's data subdirectory (for built in force fields).

        """

        if ffxml_filenames:
            for ffxml_filename in ffxml_filenames:
                ffxml = self._read_ffxml(ffxml_filename)
                self._ffxmls.add(ffxml)

        return

    def _create_system(self):
        """
        Create the OpenMM System object.

        """

        # Create file-like objects from ffxml contents because ForceField cannot yet read strings.
        from StringIO import StringIO
        ffxml_streams = list()
        for ffxml in self._ffxmls:
            ffxml_streams.append(StringIO(ffxml))

        # Create ForceField.
        forcefield = app.ForceField(*ffxml_streams)

        # Create System from topology.
        self._system = forcefield.createSystem(self._topology, **self.system_creation_parameters)
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
# ABSTRACT BIOPOLYMER SYSTEM BUILDER
#=============================================================================================

class BiopolymerSystemBuilder(SystemBuilder):
    """
    Abstract base class for classes that will read biopolymers.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,ffxml_filenames=None):
        """
        Abstract base class for classes that will read biopolymers.

        Parameters
        ----------

        """
        super(BiopolymerSystemBuilder, self).__init__(ffxml_filenames=ffxml_filenames)
        return

#=============================================================================================
# BIOPOLYMER SYSTEM BUILDER FOR PDB FILES
#=============================================================================================

class BiopolymerPDBSystemBuilder(BiopolymerSystemBuilder):

    """
    BiopolymerPDBSystemBuilder:

    This class is a subclass of BiopolymerSystemBuilder, and uses a PDB file as input. Currently, it accepts proteins and uses PDBFixer to try to repair issues with the PDB file
    This class does not currently perform any parameterization. As such, failure to give it appropriate forcefield ffxml(s) will cause it to fail.

    """

    def __init__(self, pdb_filename, chain_ids=None, ffxml_filenames=['amber99sbildn.xml'], pH=7.0):
        """
        Create a biopolymer from a specified PDB file.

        Parameters
        ----------
        pdb_filename : str
           PDB filename
        chain_ids : str, default=None
           List of chain IDs that should be kept (e.g. 'ACFG'), or None if all are kept.
        ffxml_filenames : list of str, default=None
           List of OpenMM ForceField ffxml files used to parameterize the System.
        pH : float, default 7.0
           pH to be used in determining the protonation state of the biopolymer.

        Examples
        --------
        Create a SystemBuilder for a PDB file.
        >>> from repex import testsystems
        >>> receptor_pdb_filename = testsystems.get_data_filename("data/T4-lysozyme-L99A-implicit/receptor.pdb")
        >>> receptor = BiopolymerPDBSystemBuilder(receptor_pdb_filename, pH=7.0)
        >>> system = receptor.system
        >>> positions = receptor.positions
        >>> natoms = receptor.natoms

        """
        # Call base constructor.
        super(BiopolymerPDBSystemBuilder, self).__init__(ffxml_filenames=ffxml_filenames)

        # Store the desired pH used to assign protonation states.
        self._pH = pH

        # Use PDBFixer to add missing atoms and residues and set protonation states appropriately.
        from pdbfixer import pdbfixer
        fixer = pdbfixer.PDBFixer(pdb_filename)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.removeHeterogens(True)
        fixer.addMissingHydrogens(self._pH)

        # Keep only the chains the user wants
        if chain_ids is not None:
            # TODO: Check correctness of this.
            n_chains = len(list(fixer.topology.chains()))
            chains_to_remove = np.setdiff1d(np.arange(n_chains), chain_ids) # TODO: Check if this is robust to weird chain orderings.
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
# ABSTRACT BASE SMALL MOLECULE BUILDER
#=============================================================================================

class SmallMoleculeBuilder(SystemBuilder):
    """
    Concrete base class for SystemBuilders that will handle small molecules given OpenMM positions and topology.
    Parameters may be created by an external tool, if specified.

    This version manages an internal OpenEye OEChem object for convenience.
    Other tools may be supported as well in the future.

    TODO:
    * Allow specification of charge assignment method.
    * Support additional small molecule parameterization methods (e.g. oeante, paramchem, ATB)

    """

    # Loaded OpenEye modules
    oechem = None
    oeiupac = None
    oeomega = None
    oequacpac = None

    def __init__(self, molecule, parameterize='gaff2xml', parameterize_arguments=None, charge=None, molecule_name=None, **kwargs):
        """
        SystemBuilder capable of parameterizing small molecules given OpenMM positions and topology.

        Parameters
        ----------
        molecule : openeye.oechem.OEMol
        parameterize : str, optional, default='gaff2xml'
           External tool used to parameterize the molecule. One of [False, 'gaff2xml'].
           If False, tool will not be called.
        parameterize_arguments : dict, optional, default=None
           Dictionary to be passed to parameterization tool.
        charge : int, optional, default=None
           If specified, the appropriate charge state will be selected.

        **kwargs are passed to external parameterization tool

        """
        # Load and license check for all necessary OpenEye libraries
        SmallMoleculeBuilder._load_verify_openeye()

        # Call the constructor.
        super(SmallMoleculeBuilder, self).__init__(**kwargs)

        # Normalize the molecule.
        molecule = self._normalize_molecule(molecule)

        # Select the desired charge state, if one is specified.
        if charge is not None:
            # Enumerate protonation states and select desired state.
            protonation_states = self._enumerate_states(molecule, type_of_states="protonation")

            # Search through the states for desired charge
            for molecule in protonation_states:
                if self._formal_charge(molecule) == charge:
                    break

            # Throw exception if we are unable to find desired charge.
            if self._formal_charge(molecule) != charge:
                print "enumerateStates did not enumerate a molecule with desired formal charge."
                print "Options are:"
                for molecule in protonation_states:
                    print "%s, formal charge %d" % (molecule.GetTitle(), self._formal_charge(molecule))
                raise RuntimeError("Could not find desired formal charge.")

        # Generate a 3D conformation if we don't have a 3-dimensional molecule.
        if molecule.GetDimension() < 3:
            molecule = self._expand_conformations(molecule, maxconfs=1)

        # Store OpenMM positions and topologies.
        [self._positions, self._topology] = self._oemol_to_openmm(molecule)

        # Parameterize if requested.
        if parameterize:
            if parameterize == 'gaff2xml':
                self._parameterize_with_gaff2xml(molecule, parameterize_arguments)

        return

    @classmethod
    def is_supported(cls):
        """
        Check whether this module is supported.

        Returns
        -------
        is_supported : bool
           True if all functionality is present; False otherwise.

        Examples
        --------
        >>> is_supported = SmallMoleculeBuilder.is_supported()

        """

        try:
            # Load and license check for all necessary OpenEye libraries
            cls._load_verify_openeye()
            return True
        except RuntimeError:
            return False

    def _parameterize_with_gaff2xml(self, molecule, charge, parameterize_arguments=dict()):
        """
        Parameterize the molecule using gaff2xml, appending the parameters to the set of loaded parameters.

        Parameters
        ----------
        parameterize_arguments : dict, optional, default=None
           Optional kwargs to be passed to gaff2xml.

        """

        # Attempt to import gaff2xml.
        import gaff2xml

        # Change to a temporary working directory.
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)

        # Write Tripos mol2 file.
        substructure_name = "MOL" # substructure name used in mol2 file
        mol2_filename = self._write_molecule(molecule, filename='tripos.mol2', substructure_name=substructure_name)

        # Run antechamber via gaff2xml to generate parameters.
        # TODO: We need a way to pass the net charge.
        # TODO: Can this structure be simplified?
        if 'charge_method' in parameterize_arguments:
            if 'net_charge' not in parameterize_arguments:
                # Specify formal charge.
                formal_charge = self._formal_charge(molecule)
                parameterize_arguments['net_charge'] = formal_charge

        if parameterize_arguments:
            (gaff_mol2_filename, gaff_frcmod_filename) = gaff2xml.utils.run_antechamber(self._molecule_name, mol2_filename, **parameterize_arguments)
        else:
            (gaff_mol2_filename, gaff_frcmod_filename) = gaff2xml.utils.run_antechamber(self._molecule_name, mol2_filename)

        # Write out the ffxml file from gaff2xml.
        ffxml_filename = "molecule.ffxml"
        gaff2xml.utils.create_ffxml_file(gaff_mol2_filename, gaff_frcmod_filename, ffxml_filename)

        # Append the ffxml file to loaded parameters.
        self._append_ffxmls([ffxml_filename])

        # Restore working directory.
        os.chdir(cwd)

        # Clean up temporary working directory.
        for filename in os.listdir(tmpdir):
            file_path = os.path.join(tmpdir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception, e:
                print e

        return

    def _write_molecule(self, molecule, filename=None, format=None, preserve_atomtypes=False, substructure_name=None):
        """Write the given OpenEye molecule to a file.

        Parameters
        ----------
        molecule : openeye.oechem.OEMol
           The molecule to be written to file (will notbe changed).
        filename : str, optional, default=None
           The name of the file to be written, or None if a temporary file is to be created.
        format : OEFormat, optional, default=None
           The format of the file to be written (
        preserve_atomtypes : bool, optional, default=False
           If True, atom types will not be converted before writing.
        substructure_name : str, optional, default=None
           If specified, mol2 substructure name will be set to specified name.

        Returns
        -------
        filename : str
           The name of the file written.

        """
        if filename == None:
            file = tempfile.NamedTemporaryFile(delete=False)
            filename = file.name
            file.close() # close the file so we can open it again

        # Make a copy of the molecule so it will not be changed.
        molecule = self.oechem.OEMol(molecule)

        # Open the output stream
        ostream = self.oechem.oemolostream(filename)

        # Select the format.
        if format:
            ostream.SetFormat(format)

        # Define internal function for writing multiple conformers to an output stream.
        def write_all_conformers(ostream, molecule):
            # write all conformers of each molecule
            for conformer in molecule.GetConfs():
                if preserve_atomtypes: self.oechem.OEWriteMol2File(ostream, conformer)
                else: self.oechem.OEWriteConstMolecule(ostream, conformer)
            return

        # If 'molecule' is actually a list of molecules, write them all.
        if type(molecule) == type(list()):
            for individual_molecule in molecule:
                write_all_conformers(ostream, individual_molecule)
        else:
            write_all_conformers(ostream, molecule)

        # Close the stream.
        ostream.close()

        # Modify substructure name if requested.
        if substructure_name:
            self._modify_substructure_name(filename, substructure_name)

        return filename

    def _modify_substructure_name(self, mol2file, name):
        """Replace the substructure name (subst_name) in a mol2 file.

        ARGUMENTS
        mol2file (string) - name of the mol2 file to modify
        name (string) - new substructure name

        NOTES
        This is useful becuase the OpenEye tools leave this name set to <0>.
        The transformation is only applied to the first molecule in the mol2 file.

        TODO
        This function is still difficult to read.  It should be rewritten to be comprehensible by humans.
        Check again to see if there is OpenEye functionality to write the substructure name correctly.

        """

        # Read mol2 file.
        file = open(mol2file, 'r')
        text = file.readlines()
        file.close()

        # Find the atom records.
        atomsec = []
        ct = 0
        while text[ct].find('<TRIPOS>ATOM')==-1:
            ct+=1
        ct+=1
        atomstart = ct
        while text[ct].find('<TRIPOS>BOND')==-1:
            ct+=1
        atomend = ct

        atomsec = text[atomstart:atomend]
        outtext=text[0:atomstart]
        repltext = atomsec[0].split()[7] # mol2 file uses space delimited, not fixed-width

        # Replace substructure name.
        for line in atomsec:
            # If we blindly search and replace, we'll tend to clobber stuff, as the subst_name might be "1" or something lame like that that will occur all over. 
            # If it only occurs once, just replace it.
            if line.count(repltext)==1:
                outtext.append( line.replace(repltext, name) )
            else:
                # Otherwise grab the string left and right of the subst_name and sandwich the new subst_name in between. This can probably be done easier in Python 2.5 with partition, but 2.4 is still used someplaces.
                # Loop through the line and tag locations of every non-space entry
                blockstart=[]
                ct=0
                c=' '
                for ct in range(len(line)):
                    lastc = c
                    c = line[ct]
                    if lastc.isspace() and not c.isspace():
                        blockstart.append(ct)
                        line = line[0:blockstart[7]] + line[blockstart[7]:].replace(repltext, name, 1)
                        outtext.append(line)

        # Append rest of file.
        for line in text[atomend:]:
            outtext.append(line)

        # Write out modified mol2 file, overwriting old one.
        file = open(mol2file,'w')
        file.writelines(outtext)
        file.close()

        return

    def _oemol_to_openmm(self, molecule):
        """Extract OpenMM positions and topologies from an OpenEye OEMol molecule.

        Parameters
        ----------
        molecule : openeye.oechem.OEMol
           The molecule from which positions and topology are to be extracted.
           NOTE: This must be a Tripos format mol2 file, not a GAFF format file.

        Returns
        -------
        positions : simtk.unit.Quantity with units compatible with nanometers, natoms x 3
           The atomic positions.
        topology : simtk.openmm.app.Topology
           OpenMM Topology object for the small molecule.

        """

        # Change to a temporary working directory.
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)

        # Write a Tripos mol2 file to a temporary file.
        substructure_name = "MOL" # substructure name used in mol2 file
        mol2_filename = 'molecule.mol2'
        self._write_molecule(molecule, filename=mol2_filename, substructure_name=substructure_name)

        # Read the mol2 file in MDTraj.
        import mdtraj
        mdtraj_molecule = mdtraj.load(mol2_filename)
        positions = mdtraj_molecule.openmm_positions(0)
        topology = mdtraj_molecule.top.to_openmm()

        # Restore working directory.
        os.chdir(cwd)

        # Clean up temporary working directory.
        for filename in os.listdir(tmpdir):
            file_path = os.path.join(tmpdir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception, e:
                print e

        # Return OpenMM format positions and topology.
        return [positions, topology]

    def _normalize_molecule(self, molecule, set_name_to_iupac=True):
        """Normalize the molecule by checking aromaticity, adding explicit hydrogens, and renaming by IUPAC name.

        Parameters
        ----------
        molecule : openeye.oechem.OEMol
           The molecule to be normalized.
        set_name_to_iupac : book, optional, default=True
           If True, the molecule title will be set to the IUPAC name.

        Returns
        -------
        normalized_molecule : openeye.oechem.OEMol
           The normalized molecule.

        """

        # Make a copy of the molecule.
        normalized_molecule = self.oechem.OEMol(molecule)

        # Find ring atoms and bonds
        self.oechem.OEFindRingAtomsAndBonds(normalized_molecule)

        # Assign aromaticity.
        self.oechem.OEAssignAromaticFlags(normalized_molecule, self.oechem.OEAroModelOpenEye)

        # Add hydrogens.
        self.oechem.OEAddExplicitHydrogens(normalized_molecule)

        if set_name_to_iupac:
            # Set title to IUPAC name.
            name = self.oeiupac.OECreateIUPACName(normalized_molecule)
            normalized_molecule.SetTitle(name)

        return normalized_molecule

    def _expand_conformations(self,molecule, maxconfs=None, threshold=None, include_original=False, torsionlib=None, verbose=False, strictTyping=None, strictStereo=True):
        """Enumerate conformations of the molecule with OpenEye's Omega after normalizing molecule.

        Parameters
        ----------
        molecule : openeye.oechem.OEMol
           Molecule to enumerate conformations for.
        include_original (boolean) - if True, original conformation is included (default: False)
        maxconfs (integer) - if set to an integer, limits the maximum number of conformations to generated -- maximum of 120 (default: None)
        threshold (real) - threshold in RMSD (in Angstroms) for retaining conformers -- lower thresholds retain more conformers (default: None)
        torsionlib (string) - if a path to an Omega torsion library is given, this will be used instead (default: None)
        verbose (boolean) - if True, omega will print extra information
        strictTyping (boolean) -- if specified, pass option to SetStrictAtomTypes for Omega to control whether related MMFF types are allowed to be substituted for exact matches.
        strictStereo (boolean) -- if specified, pass option to SetStrictStereo; otherwise use default.

        Returns
        -------
        expanded_molecule : openeye.oechem.OEMol
           Molecule with expanded conformations.

        """

        # Copy molecule.
        expanded_molecule = self.openeye.oechem.OEMol(molecule)

        # Initialize Omega.
        omega = self.oeomega.OEOmega()
        if strictStereo != None: omega.SetStrictStereo(strictStereo)  # Fail if stereochemistry is not specified.
        if strictTyping != None: omega.SetStrictAtomTypes(strictTyping)  # Fail if any atom does not have a valid MMFF atom type.
        if include_original != None: omega.SetIncludeInput(include_original)  # Include input
        if torsionlib != None: omega.SetTorsionLibrary(torsionlib)
        if maxconfs != None: omega.SetMaxConfs(maxconfs)  # Return just one conformation.
        omega(expanded_molecule)  # Generate conformation.

        return expanded_molecule

    def _formal_charge(self, molecule):
        """Find the net charge of a molecule.

        Parameters
        ----------
        molecule : OEMol
            the molecule whose formal charge is to be determined

        Returns
        -------
        int
            The formal charge of the molecule

        """
        mol_copy = self.oechem.OEMol(molecule)
        self.oechem.OEFormalPartialCharges(mol_copy)
        return int(round(self.oechem.OENetCharge(mol_copy)))

    def _enumerate_states(self, molecules, type_of_states="protonation", consider_aromaticity=True, maxstates=200, verbose=False):
        """Enumerate protonation or tautomer states for a list of molecules.

         Parameters
         ----------
         molecules : (OEMol or list of OEMol)
            molecule(s) for which states are to be enumerated
         type_of_states : str, optional
            type of states to expand -- 'protonation' or 'tautomer' (default: 'protonation')
         consider_aromaticity - bool, optional
            if True, aromaticity of the states will be evaluated. (default : True)
         verbose - bool, optional
            if True, will print out debug output (default : False)

         Returns
         -------
         list of OEMol
            molecules in different protonation or tautomeric states

         """

        # If 'molecules' is not a list, promote it to a list.
        if type(molecules) != type(list()):
            molecules = [molecules]

        # Check input arguments.
        if not ((type_of_states == "protonation") or (type_of_states == "tautomer")):
            raise "'enumerate' argument must be either 'protonation' or 'tautomer' -- instead got '%s'" % enumerate

        # Create an internal output stream to expand states into.
        ostream = self.oechem.oemolostream()
        ostream.openstring()
        ostream.SetFormat(self.oechem.OEFormat_SDF)

        # Default parameters.
        only_count_states = False # enumerate states, don't just count them

        # Enumerate states for each molecule in the input list.
        states_enumerated = 0
        for molecule in molecules:
            if verbose:
                print "Enumerating states for molecule %s." % molecule.GetTitle()

            # Dump enumerated states to output stream (ostream).
            if type_of_states == "protonation":
                # Create a functor associated with the output stream.
                functor = self.oequacpac.OETyperMolFunction(ostream, consider_aromaticity, False, maxstates)

                # Enumerate protonation states.
                if verbose:
                    print "Enumerating protonation states..."
                states_enumerated += self.oequacpac.OEEnumerateFormalCharges(molecule, functor, verbose)
            elif type_of_states == "tautomer":
                # Create a functor associated with the output stream.
                functor = self.oequacpac.OETautomerMolFunction(ostream, consider_aromaticity, False, maxstates)  # TODO: deprecated

                # Enumerate tautomeric states.
                if verbose:
                    print "Enumerating tautomer states..."
                states_enumerated += self.oequacpac.OEEnumerateTautomers(molecule, functor, verbose)

        if verbose:
            print "Enumerated a total of %d states." % states_enumerated

        # Collect molecules from output stream into a list.
        states = list()
        if states_enumerated > 0:
            state = self.oechem.OEMol()
            istream = self.oechem.oemolistream()
            istream.openstring(ostream.GetString())
            istream.SetFormat(self.oechem.OEFormat_SDF)
            while self.oechem.OEReadMolecule(istream, state):
                states.append(self.oechem.OEMol(state)) # append a copy

        # Return the list of expanded states as a Python list of OEMol() molecules.
        return states

    @classmethod
    def _load_verify_openeye(cls, oechemlicensepath=None):
        """Loads required OpenEye libraries and checks licenses

        Parameters
        ----------
        oechemlicensepath : str, optional, default=None
            OpenEye license path to use, or None if environment variables are to be used.

        Raises
        ------
        RuntimeError
            If OE_LICENSE is not found as an environment variable
            If A valid license is missing

        Notes
        -----
        Needs to be run before any of the other functions to assure OpenEye libraries are accessible.

        """

        # Don't do anything if we've already imported OpenEye toolkit.
        if cls.oechem: return

        # Import the OpenEye toolkit components.
        from openeye import oechem     # For chemical objects
        from openeye import oeiupac    # For IUPAC conversion
        from openeye import oeomega    # For conformer generation
        from openeye import oequacpac  # For pKa estimations

        import os

        if oechemlicensepath is not None:
            os.environ['OE_LICENSE'] = oechemlicensepath

        try:
            os.environ['OE_LICENSE']  # See if license path is set.
        except KeyError:
            raise RuntimeError("Environment variable OE_LICENSE needs to be set.")

        if not oechem.OEChemIsLicensed():  # Check for OEchem TK license.
            raise RuntimeError("No valid license available for OEChem TK.")

        if not oeiupac.OEIUPACIsLicensed():  # Check for Lexichem TK license.
            raise RuntimeError("No valid license available for Lexichem TK.")

        if not oeomega.OEOmegaIsLicensed():  # Check for Omega TK license.
            raise RuntimeError("No valid license for Omega TK.")

        if not oequacpac.OEQuacPacIsLicensed():  # Check for Quacpac TK license.
            raise RuntimeError("No valid license for Quacpac TK.")

        #Attach libraries to the instance to only load and check them once at initialization.
        cls.oechem = oechem
        cls.oeiupac = oeiupac
        cls.oeomega = oeomega
        cls.oequacpac = oequacpac
        return

#=============================================================================================
# SMALL MOLECULE SYSTEM BUILDER FOR MOL2 FILES
#=============================================================================================

class Mol2SystemBuilder(SmallMoleculeBuilder):
    """
    Create a system from a small molecule specified in a Tripos mol2 file.

    """

    def __init__(self, mol2_filename, **kwargs):
        """
        Create a system from a Tripos mol2 file.

        Parameters
        ----------
        mol2_filename : str
           Small molecule coordinate file in Tripos mol2 format.
        Other arguments are inherited from SmallMoleculeSystemBuilder.

        Examples
        --------
        Create a SystemBuilder from a ligand mol2 file, using default parameterization scheme.
        >>> from repex import testsystems
        >>> ligand_mol2_filename = testsystems.get_data_filename("data/T4-lysozyme-L99A-implicit/ligand.tripos.mol2")
        >>> ligand = Mol2SystemBuilder(ligand_mol2_filename, charge=0)
        >>> system = ligand.system
        >>> positions = ligand.positions
        >>> natoms = ligand.natoms

        """

        # Initialize the OpenEye toolkit.
        SmallMoleculeBuilder._load_verify_openeye()

        # Open an input stream
        istream = self.oechem.oemolistream()
        istream.open(mol2_filename)

        # Prepare a molecule object
        molecule = self.oechem.OEMol()

        # Read the molecule
        self.oechem.OEReadMolecule(istream, molecule)

        # Close stream
        istream.close()

        # Initialize small molecule parameterization engine.
        super(Mol2SystemBuilder, self).__init__(molecule, **kwargs)

        return

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

        Examples
        --------
        Create a ComplexSystemBuilder from a protein PDB file and a ligand mol2 file.
        >>> from repex import testsystems
        >>> receptor_pdb_filename = testsystems.get_data_filename("data/T4-lysozyme-L99A-implicit/receptor.pdb")
        >>> ligand_mol2_filename = testsystems.get_data_filename("data/T4-lysozyme-L99A-implicit/ligand.tripos.mol2")
        >>> receptor = BiopolymerPDBSystemBuilder(receptor_pdb_filename, pH=7.0)
        >>> ligand = Mol2SystemBuilder(ligand_mol2_filename, charge=0)
        >>> complex = ComplexSystemBuilder(ligand, receptor, remove_ligand_overlap=True)
        >>> system = complex.system
        >>> positions = complex.positions
        >>> ligand_atoms = complex.ligand_atoms
        >>> receptor_atoms = complex.receptor_atoms
        >>> natoms = complex.natoms

        """

        # Call base class constructor.
        super(ComplexSystemBuilder, self).__init__()

        # Append ffxml files.
        self._ffxmls = list()
        self._ffxmls += receptor.ffxmls
        self._ffxmls += ligand.ffxmls

        # Concatenate topologies and positions.
        from simtk.openmm import app
        model = app.modeller.Modeller(receptor.topology, receptor.positions)
        model.add(ligand.topology, ligand.positions)
        self._topology = model.topology
        self._positions = model.positions

        # Store indices for receptor and ligand.
        self._receptor_atoms = range(receptor.natoms)
        self._ligand_atoms = range(receptor.natoms, receptor.natoms + ligand.natoms)

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
        import mdtraj as md

        # Create an mdtraj Topology of the complex from OpenMM Topology object.
        mdtraj_complex_topology = md.Topology.from_openmm(self._topology)

        # Create an mdtraj instance of the complex.
        # TODO: Fix this when mdtraj can deal with OpenMM units.
        positions_in_mdtraj_format = np.array(self._positions / units.nanometers)
        mdtraj_complex = md.Trajectory(positions_in_mdtraj_format, mdtraj_complex_topology)

        # Compute centers of receptor and ligand.
        receptor_center = mdtraj_complex.xyz[0][self._receptor_atoms,:].mean(0)
        ligand_center = mdtraj_complex.xyz[0][self._ligand_atoms,:].mean(0)

        # Count number of receptor and ligand atoms.
        nreceptor_atoms = len(self._receptor_atoms)
        nligand_atoms = len(self._ligand_atoms)

        # Compute max radii of receptor and ligand.
        receptor_radius = (((mdtraj_complex.xyz[0][self._receptor_atoms,:] - np.tile(receptor_center, (nreceptor_atoms,1))) ** 2.).sum(1) ** 0.5).max()
        ligand_radius = (((mdtraj_complex.xyz[0][self._ligand_atoms,:] - np.tile(ligand_center, (nligand_atoms,1))) ** 2.).sum(1) ** 0.5).max()

        # Translate ligand along x-axis from receptor center with 5% clearance.
        mdtraj_complex.xyz[0][self._ligand_atoms,:] += np.array([1.0, 0.0, 0.0]) * (receptor_radius + ligand_radius) * 1.05 - ligand_center + receptor_center

        # Extract updated system positions.
        self._positions = mdtraj_complex.openmm_positions(0)

        return

    @property
    def ligand_atoms(self):
        return copy.deepcopy(self._ligand_atoms)

    @property
    def receptor_atoms(self):
        return copy.deepcopy(self._receptor_atoms)

#=============================================================================================
# TEST CODE
#=============================================================================================

def test_alchemy():
    import os
    import simtk.unit as unit
    import simtk.openmm as openmm
    import numpy as np
    import simtk.openmm.app as app
    import alchemy

    # Create SystemBuilder objects.
    from repex import testsystems
    receptor_pdb_filename = testsystems.get_data_filename("data/T4-lysozyme-L99A-implicit/receptor.pdb")
    ligand_mol2_filename = testsystems.get_data_filename("data/T4-lysozyme-L99A-implicit/ligand.tripos.mol2")
    receptor = BiopolymerPDBSystemBuilder(receptor_pdb_filename, pH=7.0)
    ligand = Mol2SystemBuilder(ligand_mol2_filename, charge=0)
    complex = ComplexSystemBuilder(ligand, receptor, remove_ligand_overlap=True)

    timestep = 2 * unit.femtoseconds # timestep
    temperature = 300.0 * unit.kelvin # simulation temperature
    collision_rate = 20.0 / unit.picoseconds # Langevin collision rate
    minimization_tolerance = 10.0 * unit.kilojoules_per_mole / unit.nanometer
    minimization_steps = 20
    plat = "CPU"
    i=2
    platform = openmm.Platform.getPlatformByName(plat)
    forcefield = app.ForceField
    systembuilders = [ligand, receptor, complex]
    receptor_atoms = complex.receptor_atoms
    ligand_atoms = complex.ligand_atoms
    factory = alchemy.AbsoluteAlchemicalFactory(systembuilders[i].system, ligand_atoms=ligand_atoms)
    protocol = factory.defaultComplexProtocolImplicit()
    systems = factory.createPerturbedSystems(protocol)
    integrator_interacting = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    #test an alchemical intermediate and an unperturbed system:
    fully_interacting = app.Simulation(systembuilders[i].topology, systems[0], integrator_interacting, platform=plat)
    fully_interacting.context.setPositions(systembuilders[i].positions)
    fully_interacting.minimizeEnergy(tolerance=10*unit.kilojoule_per_mole)
    fully_interacting.reporters.append(app.PDBReporter('fully_interacting.pdb', 10))
    for j in range(10):
        print str(j)
        fully_interacting.step(100)
    del fully_interacting


    for p in range(1, len(systems)):
        print "now simulating " + str(p)
        integrator_partialinteracting = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        partially_interacting = app.Simulation(systembuilders[i].topology, systems[p], integrator_partialinteracting, platform=plat)
        partially_interacting.context.setPositions(systembuilders[i].positions)
        partially_interacting.minimizeEnergy(tolerance=10*unit.kilojoule_per_mole)
        partially_interacting.reporters.append(app.PDBReporter('partial_interacting'+str(p)+'.pdb', 10))
        for k in range(10):
            print str(k)
            partially_interacting.step(100)
        del partially_interacting

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    # Run doctests.
    import doctest
    doctest.testmod()

    # Run test.
    test_alchemy()
