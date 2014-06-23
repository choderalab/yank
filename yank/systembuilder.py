__author__ = 'Patrick Grinaway'

import abc
#mol2 = Mol2SystemBuilder("ligand.tripos.mol2")
class SystemBuilder():
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
    __metaclass__ = abc.ABCMeta



    def __init__(self, coordinate_file, molecule_name, forcefield_files=None):
        """
        Generic SystemBuilder constructors

        ARGUMENTS:

        coordinate_file: filename that contains the molecule of interest
        molecule_name: name of molecule
        forcefield_file: ffxml file for the molecule.

        """
        self._forcefield_files = forcefield_files
        self._coordinate_file = coordinate_file
        self._molecule_name = molecule_name
        self._forcefield = None
        self._traj = None
        self._system = None







    @abc.abstractmethod
    def _create_traj(self):
        """
        _create_traj should use the provided coordinate object (or transform to a more suitable format first) then set
        self._traj to the correct mdtraj.Trajectory object
        """
        pass

    def center_coordinates(self):
        if self._traj is None:
            self._create_traj()
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
        if self._system is None:
            self._create_system()
        return self._system


    @property
    def openmm_positions(self):
        return self._traj.openmm_positions(0)

    @property
    def positions(self):
        if self._traj is None:
            self._create_traj()
            return self._traj.xyz[:,:]
        else:
            return self._traj.xyz[:,:]

    @property
    def forcefield(self):
        if self._forcefield is None:
            import simtk.openmm.app as app
            self._forcefield = app.ForceField(*self._forcefield_files)
            return self._forcefield
        else:
            return self._forcefield


    @property
    def forcefield_files(self):
        return self._forcefield_files

    @property
    def traj(self):
        return self._traj
    @traj.setter
    def traj(self, new_traj):
        if (new_traj.top!=self._traj.top):
            raise Exception
        else:
            self._traj = new_traj



class SmallMoleculeBuilder(SystemBuilder):


    def __init__(self, coordinate_file, molecule_name, forcefield_files=None, parameterization_method="antechamber"):
        """
        SmallMoleculeBuilder Constructor. This is a stub.

        ACCEPTS

        coordinate_file: small molecule coordinate file name
        molecule_name: name of the molecule (string)
        forcefield_file: ffxml file that contains parameters for the molecule. If none, generate.
        """
        super(SmallMoleculeBuilder, self).__init__(coordinate_file, molecule_name, forcefield_files)
        if forcefield_files is None:
            self.build_forcefield(param_method=parameterization_method)




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





class BiomoleculeSystemBuilder(SystemBuilder):
    """
    This is an abstract base class for classes that will read biomolecules (large molecules) from files. This class
    will not perform parameterization (at least for now)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,coordinate_file, molecule_name, forcefield_files, chain_ids):
        #self._chains_to_use = chain_ids
        super(BiomoleculeSystemBuilder, self).__init__(coordinate_file, molecule_name, forcefield_files)
        self._chains_to_use = chain_ids

    #set default to amber10 in case forcefield files is none
    @property
    def forcefield_files(self):
        if self._forcefield_files is None:
            self._forcefield_files = ["amber10.xml"]
            return self._forcefield_files
        else:
            return self._forcefield_files
    @property
    def chains_to_use(self):
        return self._chains_to_use




class BiomoleculePDBSystemBuilder(BiomoleculeSystemBuilder):



    def __init__(self, coordinate_file, molecule_name, forcefield_files=None, chain_ids=None, pH=7.0 ):
        self._pH = pH
        super(BiomoleculePDBSystemBuilder, self).__init__(coordinate_file, molecule_name, forcefield_files, chain_ids)
        return

    def _create_traj(self):
        import mdtraj
        self._traj = mdtraj.load_pdb(self._coordinate_file)




    def _create_system(self):
        import pdbfixer
        import simtk.openmm.app as app
        import mdtraj
        fixer = pdbfixer.PDBFixer(self._coordinate_file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.removeHeterogens(True)
        fixer.addMissingHydrogens(self._pH)
        #keep only the chains the user wants
        if self._chains_to_use is not None:
            molecule_chains = list(fixer.topology.chains())
            #figure this out
            raise NotImplementedError
        if self._forcefield_files is None:
            self._forcefield_files = ["amber10.xml"]
        forcefield = app.ForceField(*self._forcefield_files)
        #change this later to not have these hardcoded
        self._system = forcefield.createSystem(fixer.topology, implicitSolvent=app.OBC2)
        output_filename = "fixed." + self._coordinate_file
        app.PDBFile.writeFile(fixer.topology,fixer.positions, file=open(output_filename,'w'))
        self._coordinate_file = output_filename

    @property
    def pH(self):
        return self._pH



class Mol2SystemBuilder(SmallMoleculeBuilder):


    def __init__(self, coordinate_file, molecule_name, forcefield_files=None, parameterization_method="antechamber", gaff_mol2_file=None):
        """
        Mol2SystemBuilder constructor

        ACCEPTS


        objectcoordinate_file: small molecule coordinate file name
        molecule_name: name of the molecule (string)
        forcefield_file: ffxml file that contains parameters for the molecule. If none, generate.
        parameterization_method: the method used to parameterize the molecule. currently only antechamber
        gaff_mol2_file: a parsed gaff mol2 file, if available

        """
        self._gaff_mol2 = gaff_mol2_file
        super(Mol2SystemBuilder, self).__init__(coordinate_file, molecule_name, forcefield_files=forcefield_files, parameterization_method=parameterization_method)




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
        self._gaff_mol2 = gaff_mol2_filename
        parser.parse_filenames([gaff_mol2_filename, gaff_frcmod_filename])
        ffxml_stream = parser.generate_xml()
        ffxml_filename = self._molecule_name+ '.ffxml'
        outfile = open(ffxml_filename, 'w')
        outfile.write(ffxml_stream.read())
        outfile.close()
        if self._forcefield_files is None:
            self._forcefield_files = list()
            self._forcefield_files.append(ffxml_filename)




    def _create_system(self):

        """
        This function overrides the parent's _create_system in order to create a simtk.openmm.System object representing the molecule of interest.


        """


        import gaff2xml
        from simtk.openmm import app
        import mdtraj

        mol2 = mdtraj.load(self._gaff_mol2)
        positions = mol2.openmm_positions(0)
        topology = mol2.top.to_openmm()
        forcefield = app.ForceField(*self._forcefield_files)
        self._system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)


    def _create_traj(self):
        import mdtraj as md
        print self._gaff_mol2
        self._traj=md.load(self._gaff_mol2)







class ComplexSystemBuilder(SystemBuilder):

    def __init__(self, ligand_system, receptor_system, complex_name):
        import numpy as np

        #fill in appropriate member variables
        self._ligand_system = ligand_system
        self._receptor_system = receptor_system
        print type(self._receptor_system.forcefield_files)

        self._forcefield_files = self._ligand_system.forcefield_files + self._receptor_system.forcefield_files
        #center the coordinate systems
        self.center_coordinates()
        self._complex_name = complex_name


        #change ligand coordinates to not overlap with protein
        self._adjust_ligand_coordinates()
        self._combine_systems()
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

    def _create_traj(self):
        import mdtraj
        self._traj = mdtraj.load_pdb(self._coordinate_file)




    def _create_system(self):
        self._combine_systems()

    def _combine_systems(self):
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
        receptor_positions = self._receptor_system.traj.openmm_positions(0)
        receptor_topology = self._receptor_system.traj.top.to_openmm()
        ligand_positions = self._ligand_system.traj.openmm_positions(0)
        ligand_topology = self._ligand_system.traj.top.to_openmm()
        self._forcefield_files = self._receptor_system.forcefield_files + self._ligand_system.forcefield_files


        forcefield = app.ForceField(*self._forcefield_files)
        model = app.modeller.Modeller(receptor_topology, receptor_positions)
        model.add(self._ligand_system.traj.topology.to_openmm(), ligand_positions)

        complex_system = forcefield.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None)
        app.PDBFile.writeFile(model.topology, model.getPositions(),open(self._complex_name+'.pdb','w'))
        self._coordinate_file = self._complex_name + '.pdb'
        self._system = complex_system
        self._create_traj()


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
    ligand = Mol2SystemBuilder('ligand.tripos.mol2', 'ligand')
    receptor = BiomoleculePDBSystemBuilder('receptor.pdb','protein')
    complex_system = ComplexSystemBuilder(ligand, receptor, "complex")
    complex_positions = complex_system.positions
    receptor_positions = receptor.positions
    print type(complex_system.what_yank_wants)





