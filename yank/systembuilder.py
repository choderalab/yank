__author__ = 'Patrick Grinaway'

import abc
#mol2 = Mol2SystemBuilder("ligand.tripos.mol2")
class SystemBuilder(object, metaclass=abc.ABCmeta):
    """
    class SystemBuilder:
        This is a base class for other classes that will create systems for use with Yank
        Only its children should use it. Several unimplemented methods must be overridden

        Properties:
            system: a simtk.openmm.system object, created when child class __init__runs
            openmm_positions: the positions of the first frame, in openmm format
            positions: returns numpy array from MDTraj of atomic positions
            forcefield_files: returns list of filenames of ffxml files
            traj: returns mdtraj.trajectory object, used in manipulating coordinates of molecules
            forcefield: returns simtk.openmm.app.ForceField object for system



    """

    def __init__(self, coordinate_file, molecule_name, forcefield_files=None):
        """
        Generic SystemBuilder constructors

        ARGUMENTS:

        coordinate_file: filename that contains the molecule of interest
        molecule_name: name of molecule
        forcefield_file: ffxml file for the molecule.

        """
        import simtk.openmm.app as app
        self._forcefield_files = forcefield_files
        self._coordinate_file = coordinate_file
        self._molecule_name = molecule_name
        self._system = self._create_system()
        self._traj = self._create_traj()






    @abc.abstractmethod
    def _create_traj(self):
        """
        _create_traj should use the provided coordinate object (or transform to a more suitable format first) then set
        self._traj to the correct mdtraj.Trajectory object
        """
        pass

    def center_coordinates(self):
        self._traj.center_coordinates()

    @abc.abstractmethod
    def _create_system(self):
        """
        _create_system() should use the provided forcefield file and coordinates to create the class's internal _system object,
        of type simtk.openmm.System

        """
        pass


    @property
    def system(self):
        return self._system

    @property
    def openmm_positions(self):
        return self._traj.openmm_positions(0)

    @property
    def positions(self):
        return self._traj.xyz[:,:]

    @property
    def forcefield(self):
        import simtk.openmm.app as app
        return app.ForceField(*self._forcefield_files)

    @property
    def forcefield_file(self):
        return self._forcefield_files





class SmallMoleculeBuilder(SystemBuilder):

    def __init__(self, coordinate_file, molecule_name, forcefield_files=None, parameterization_method="antechamber"):
        """
        SmallMoleculeBuilder Constructor. This is a stub.

        ACCEPTS

        coordinate_file: small molecule coordinate file name
        molecule_name: name of the molecule (string)
        forcefield_file: ffxml file that contains parameters for the molecule. If none, generate.
        """
        if self._forcefield_files is None:
            self.build_forcefield(param_method=parameterization_method)
            super(SmallMoleculeBuilder, self).__init__(coordinate_file, molecule_name, forcefield_files)
        else:
            super(SmallMoleculeBuilder, self).__init__(coordinate_file, molecule_name, forcefield_files)


    @abc.abstractmethod
    def build_forcefield(self, param_method=None, **kwargs):
        """
        Builds a forcefield ffxml file from the molecule using the specified method.
        Takes coordinate file and molecule name from member variables
        Must be overridden by child classes

        ACCEPTS:
        method: string indicating which method to use in parameterization
        """
        pass








class PDBSystemBuilder(SystemBuilder):

    def _create_pdb_system(self, coordinate_file, output_filename='receptor.pdb', pH=7.0, forcefield=None):

        """
        This function creates a system from a PDB file, assuming it is a protein, and uses the amber99sbildn forcefield by default.


        ARGUMENTS:

        coordinate_file (string) : this is the pdb file from which coordinates will be read
        output_filename (string): this is where the pdbfixer fixed output will go. default: receptor.pdb
        pH (float) : this is the pH that is used in assigning protonation states
        forcefield (string): this is the list of names of the forcefield xml file to use. If None (default) amber99sbildn.xml and amber99_obc.xml will be used



        """
        import pdbfixer
        import simtk.openmm.app as app
        import mdtraj
        fixer = pdbfixer.PDBFixer(coordinate_file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.removeHeterogens(True)
        fixer.addMissingHydrogens(pH)
        app.PDBFile.writeFile(fixer.topology,fixer.positions, file=open(output_filename,'w'))
        if forcefield is None:
            forcefields_to_use = ['amber99sbildn.xml', 'amber99_obc.xml']
        else:
            forcefields_to_use = forcefield
        forcefield = app.ForceField(*forcefields_to_use)
        system = forcefield.createSystem(fixer.topology, nonbondedMethod=app.NoCutoff,constraints=None,implicitSolvent=app.OBC2)
        return (system, mdtraj.load_pdb(output_filename), forcefield)

    def __init__(self, coordinate_file, forcefield_file=None):
        self._system, self._traj, self._forcefield = self._create_pdb_system(coordinate_file)

        return





class Mol2SystemBuilder(SmallMoleculeBuilder):


    def __init__(self, coordinate_file, molecule_name, forcefield_files=None, parameterization_method="antechamber", gaff_mol2_file=None):
        """
        Mol2SystemBuilder constructor

        ACCEPTS

        coordinate_file: small molecule coordinate file name
        molecule_name: name of the molecule (string)
        forcefield_file: ffxml file that contains parameters for the molecule. If none, generate.
        parameterization_method: the method used to parameterize the molecule. currently only antechamber
        gaff_mol2_file: a parsed gaff mol2 file, if available

        """
        self._gaff_mol2 = gaff_mol2_file
        super(Mol2SystemBuilder, self).__init__(self, coordinate_file, molecule_name, forcefield_files=None, parameterization_method="antechamber")




    def build_forcefield(self, param_method="antechamber", **kwargs):
        """
        This function overrides the parent's method, and builds a forcefield ffxml file for the given mol2 using gaff2xml (which in turn runs antechamber)

        Accepts:
        param_method (string): method to use to parameterize molecule; currently only antechamber is supported
        **kwargs: will be used to allow extra parameters to be passed to parameterization scheme

        """
        import gaff2xml
        from simtk.openmm import app
        parser = gaff2xml.amber_parser.AmberParser()
        (gaff_mol2_filename, gaff_frcmod_filename) = gaff2xml.utils.run_antechamber(self._molecule_name, self._coordinate_file,charge_method="bcc")
        parser.parse_filenames([gaff_mol2_filename, gaff_frcmod_filename])
        ffxml_stream = parser.generate_xml()
        ffxml_filename = self._molecule_name+ '.ffxml'
        outfile = open(ffxml_filename, 'w')
        outfile.write(ffxml_stream.read())
        outfile.close()
        self._forcefield_files[0] = ffxml_filename




    def _create_system(self):

        """
        This function overrides the parent's _create_system in order to create a simtk.openmm.System object representing the molecule of interest.


        """


        import gaff2xml
        from simtk.openmm import app
        import mdtraj

        mol2 = mdtraj.load_mol2(self._gaff_mol2)
        positions = mol2.openmm_positions()
        topology = mol2.top.to_openmm()
        forcefield = app.ForceField(*self._forcefield_files)
        self._system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)


    def _create_traj(self):
        import mdtraj
        self._traj=mdtraj.load_mol2(self._gaff_mol2)







class ComplexSystemBuilder(SystemBuilder):

    def __init__(self, ligand_system, receptor_system):
        import numpy as np
        self._ligand_system = ligand_system
        self._receptor_system = receptor_system

        #center the coordinate systems
        self.center_coordinates()

        #change ligand coordinates to not overlap with protein
        self._adjust_ligand_coordinates()
        self._system, self._traj = self._combine_systems(receptor_system, ligand_system)
        natoms_total = self._traj.n_atoms
        natoms_receptor = self._receptor_system.traj.n_atoms
        self._complex_positions = np.zeros([natoms_total, 3])
        self._complex_positions[0:natoms_receptor, :] = self._receptor_system.positions
        self._complex_positions[natoms_receptor:natoms_total, :] = self._ligand_system.positions



        return

    def center_coordinates(self):
        self._ligand_system.center_coordinates()
        self._receptor_system.center_coordinates()
        return


    def _adjust_ligand_coordinates(self):
        import numpy as np
        ligand_traj = self._ligand_system.traj
        receptor_traj = self._receptor_system.traj
        min_atom_pair_distance = ((ligand_traj.xyz[0] ** 2.).sum(1) ** 0.5).max() + ((receptor_traj.xyz[0] ** 2.).sum(1) ** 0.5).max() + 0.3
        ligand_traj.xyz += np.array([1.0, 0.0, 0.0]) * min_atom_pair_distance
        self._ligand_system.traj = ligand_traj
        return






    def _combine_systems(self, receptor_system, ligand_system):
        """
        This function will combine the two systems

        ARGUMENTS:

        receptor_system (SystemBuilder) : the system containing the 'receptor' molecule
        ligand_system (SystemBuilder) : the system containing the ligand molecule

        RETURNS:

        complex_system (simtk.openmm.System) : system containing both molecules
        complex_positions(numpy n_atoms_complex*3 array): positions of the atoms in the complex system

        """

        import simtk.openmm
        import simtk.openmm.app as app
        import simtk.unit as units
        import numpy as np
        import mdtraj
        #get ligand ffxml and create forcefield object
        receptor_positions = receptor_system.traj.openmm_positions(0)
        receptor_topology = receptor_system.traj.top.to_openmm()
        ligand_positions = ligand_system.traj.openmm_positions(0)
        ligand_topology = ligand_system.traj.top.to_openmm()


        forcefield = app.ForceField('amber10.xml', ligand_system.forcefield_file)
        model = app.modeller.Modeller(receptor_topology, receptor_positions)
        model.add(ligand_system.traj.topology.to_openmm(), ligand_positions)

        complex_system = forcefield.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None)
        app.PDBFile.writeFile(model.topology, model.getPositions(),open('complex.pdb','w'))



        #intialize complex_positions in the way that yank wants it



        return complex_system, mdtraj.load_pdb('complex.pdb')


    @property
    def complex_positions(self):
        return self._complex_positions
    @property
    def what_yank_wants(self):
        import simtk.unit as units
        return units.Quantity(self._complex_positions,units.nanometers)





if __name__=="__main__":
    import os
    os.environ['AMBERHOME']='/Users/grinawap/anaconda/pkgs/ambermini-14-py27_0'
    os.chdir('/Users/grinawap/driver/yank/examples/px-test')
    ligand = Mol2SystemBuilder('ligand.tripos.mol2')
    receptor = PDBSystemBuilder('receptor.pdb')
    complex_system = ComplexSystemBuilder(ligand, receptor)
    complex_positions = complex_system.positions
    receptor_positions = receptor.positions
    print type(complex_system.what_yank_wants)





