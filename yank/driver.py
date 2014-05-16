#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
YANK command-line driver

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import yank
import sys
import numpy
import simtk.openmm as openmm
import simtk.unit as units
import os

#=============================================================================================
# DRIVER
#=============================================================================================


#helper functions to allow driver to operate
def read_amber_crd(filename, natoms_expected, verbose=False):
    """
    Read AMBER coordinate file.

    ARGUMENTS

    filename (string) - AMBER crd file to read
    natoms_expected (int) - number of atoms expected

    RETURNS

    positions (numpy-wrapped simtk.unit.Quantity with units of distance) - a single read coordinate set

    TODO

    * Automatically handle box vectors for systems in explicit solvent
    * Merge this function back into main?

    """

    if verbose: print "Reading cooordinate sets from '%s'..." % filename

    # Read positions.
    import simtk.openmm.app as app
    inpcrd = app.AmberInpcrdFile(filename)
    positions = inpcrd.getPositions(asNumpy=True)

    # Check to make sure number of atoms match expectation.
    natoms = positions.shape[0]
    if natoms != natoms_expected:
        raise Exception("Read coordinate set from '%s' that had %d atoms (expected %d)." % (filename, natoms, natoms_expected))

    return positions

def read_openeye_crd(filename, natoms_expected, verbose=False):
    """
    Read one or more coordinate sets from a file that OpenEye supports.

    ARGUMENTS

    filename (string) - the coordinate filename to be read
    natoms_expected (int) - number of atoms expected

    RETURNS

    positions_list (list of numpy array of simtk.unit.Quantity) - list of coordinate sets read
    """

    if verbose: print "Reading cooordinate sets from '%s'..." % filename

    import openeye.oechem as oe
    imolstream = oe.oemolistream()
    imolstream.open(filename)
    positions_list = list()
    for molecule in imolstream.GetOEGraphMols():
        oecoords = molecule.GetCoords() # oecoords[atom_index] is tuple of atom positions, in angstroms
        natoms = len(oecoords) # number of atoms
        if natoms != natoms_expected:
            raise Exception("Read coordinate set from '%s' that had %d atoms (expected %d)." % (filename, natoms, natoms_expected))
        positions = units.Quantity(numpy.zeros([natoms,3], numpy.float32), units.angstroms) # positions[atom_index,dim_index] is positions of dim_index dimension of atom atom_index
        for atom_index in range(natoms):
            positions[atom_index,:] = units.Quantity(numpy.array(oecoords[atom_index]), units.angstroms)
        positions_list.append(positions)

    if verbose: print "%d coordinate sets read." % len(positions_list)

    return positions_list

def read_pdb_crd(filename, natoms_expected, verbose=False):
    """
    Read one or more coordinate sets from a PDB file.
    Multiple coordinate sets (in the form of multiple MODELs) can be read.

    ARGUMENTS

    filename (string) - name of the file to be read
    natoms_expected (int) - number of atoms expected

    RETURNS

    positions_list (list of numpy array of simtk.unit.Quantity) - list of coordinate sets read

    """
    import simtk.openmm.app as app
    pdb = app.PDBFile(filename)
    positions_list = pdb.getPositions(asNumpy=True)
    natoms = positions_list.shape[0]
    if natoms != natoms_expected:
        raise Exception("Read coordinate set from '%s' that had %d atoms (expected %d)." % (filename, natoms, natoms_expected))

    # Append if we haven't dumped positions yet.
 #   if (atom_index == natoms_expected):
  #       positions_list.append(copy.deepcopy(positions))

    # Return positions.
    return positions_list
def create_ligand_system(mol2_filename, molecule_name="ligand"):
    """
    Read in a tripos mol2 file, call antechamber via gaff2xml to parametrize ligand

    ARGUMENTS

    mol2_filename (string) - name of the file in tripos mol2 format
    molecule_name (string) - name of output ffxml. default "ligand"


    RETURNS
    simtk.openmm.System object of ligand
    mdtraj trajectory object of ligand

    ffxml filename for ligand to use in creating complex


    """
    import gaff2xml
    from simtk.openmm import app




    parser =gaff2xml.amber_parser.AmberParser()
    (gaff_mol2_filename, gaff_frcmod_filename) = gaff2xml.utils.run_antechamber(molecule_name,mol2_filename,charge_method="bcc")
    print gaff_mol2_filename
    parser.parse_filenames([gaff_mol2_filename, gaff_frcmod_filename])
    ffxml_stream = parser.generate_xml()
    ffxml_filename = molecule_name + '.ffxml'
    outfile = open(ffxml_filename, 'w')
    outfile.write(ffxml_stream.read())
    outfile.close()
    mol2 = gaff2xml.gafftools.Mol2Parser(gaff_mol2_filename)
    (topology, positions) = mol2.to_openmm()
    forcefield = app.ForceField(ffxml_filename)
    system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None, implicitSolvent=app.OBC2)
    return (system, mol2.to_mdtraj(), ffxml_filename)


def create_receptor_system(receptor_pdb, pH=7.0, output_filename='receptor.pdbfixer.pdb'):
    """
    Read in a PDB file, run it through pdbfixer, and return an OpenMM system with the receptor

    ARGUMENTS
    receptor_pdb (string): filename of receptor pdb

    RETURNS
    simtk.openmm.System object containing receptor
    mdtraj object of fixed pdb
    """
    import pdbfixer
    import simtk.openmm.app as app
    from simtk.openmm.app.internal.pdbstructure import PdbStructure
    import mdtraj
    fixer = pdbfixer.PDBFixer(PdbStructure(open(receptor_pdb)))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.removeHeterogens(True)
    fixer.addMissingHydrogens(pH)
    app.PDBFile.writeFile(fixer.topology,fixer.positions, file=output_filename)
    forcefields_to_use = ['amber99sbildn.xml', 'amber99_obc.xml']
    forcefield = app.ForceField(forcefields_to_use)
    system = forcefield.createSystem(fixer.topology, nonbondedMethod=app.NoCutoff,constraints=None,implicitSolvent=app.OBC2)
    return (system, mdtraj.load_pdb(output_filename))




def create_complex_system(ligand_traj, ligand_ffxml_name, receptor_traj):
    """

    Take in mdtraj objects of ligand and receptor, as well as ligand ffxml, to create complex system

    ARGUMENTS
    ligand_traj (mdjtraj trajectory): an mdtraj trajectory containing the ligand topology and positions
    ligand_ffxml_name (string) : the name of the file containing parameters for the ligand, in ffxml format
    receptor_traj (mdtraj trajectory): an mdtraj trajectory containing the receptor topology and positions

    RETURNS
    simtk.openmm.System containing ligand and receptor
    """

    import simtk.openmm.app as app
    import numpy as np
    ligand_traj.center_coordinates()
    ligand_top = ligand_traj.topology.to_openmm()
    receptor_traj.center_coordinates()
    receptor_top = receptor_traj.topology.to_openmm()
    receptor_xyz = receptor_traj.openmm_positions(0)
    min_atom_pair_distance = ((ligand_traj.xyz[0] ** 2.).sum(1) ** 0.5).max() + ((receptor_traj.xyz[0] ** 2.).sum(1) ** 0.5).max() + 0.3
    ligand_traj.xyz += np.array([1.0, 0.0, 0.0]) * min_atom_pair_distance
    ligand_xyz = ligand_traj.openmm_positions(0)
    forcefield = app.ForceField("amber10.xml", ligand_ffxml_name)
    model = app.modeller.Modeller(receptor_top, receptor_xyz)
    model.add(ligand_top, ligand_xyz)
    system = forcefield.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=None, implicitSolvent=app.OBC2)
    return system, model.getPositions()


def driver():
    # Initialize command-line argument parser.

    usage = """
    USAGE

    %prog --ligand_prmtop PRMTOP --receptor_prmtop PRMTOP { {--ligand_crd CRD | --ligand_mol2 MOL2} {--receptor_crd CRD | --receptor_pdb PDB} | {--complex_crd CRD | --complex_pdb PDB} } [-v | --verbose] [-i | --iterations ITERATIONS] [-o | --online] [-m | --mpi] [--restraints restraint-type] [--doctests] [--randomize_ligand]

    EXAMPLES

    # Specify AMBER prmtop/crd files for ligand and receptor.
    %prog --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --ligand_crd ligand.crd --receptor_crd receptor.crd --iterations 1000

    # Specify (potentially multi-conformer) mol2 file for ligand and (potentially multi-model) PDB file for receptor.
    %prog --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --ligand_mol2 ligand.mol2 --receptor_pdb receptor.pdb --iterations 1000

    # Specify (potentially multi-model) PDB file for complex, along with flat-bottom restraints (instead of harmonic).
    %prog --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --complex_pdb complex.pdb --iterations 1000 --restraints flat-bottom

    # Specify (potentially multi-model) PDB file for complex, along with flat-bottom restraints (instead of harmonic); randomize ligand positions/orientations at start.
    %prog --ligand_prmtop ligand.prmtop --receptor_prmtop receptor.prmtop --complex_pdb complex.pdb --iterations 1000 --restraints flat-bottom --randomize_ligand

    NOTES

    In atom ordering, receptor comes before ligand atoms.

    """

    # Parse command-line arguments.
    from optparse import OptionParser
    parser = OptionParser(usage=usage)
    parser.add_option("--ligand_mol2", dest="ligand_mol2_filename", default=None, help="ligand mol2 file (can contain multiple conformations)", metavar="LIGAND_MOL2")
    parser.add_option("--receptor_pdb", dest="receptor_pdb_filename", default=None, help="receptor PDB file (can contain multiple MODELs)", metavar="RECEPTOR_PDB")
    parser.add_option("--complex_pdb", dest="complex_pdb_filename", default=None, help="complex PDB file (can contain multiple MODELs)", metavar="COMPLEX_PDB")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False, help="verbosity flag")
    parser.add_option("-i", "--iterations", dest="niterations", default=None, help="number of iterations", metavar="ITERATIONS")
    parser.add_option("-o", "--online", dest="online_analysis", default=False, help="perform online analysis")
    parser.add_option("-m", "--mpi", action="store_true", dest="mpi", default=False, help="use mpi if possible")
    parser.add_option("--restraints", dest="restraint_type", default=None, help="specify ligand restraint type: 'harmonic' or 'flat-bottom' (default: 'harmonic')")
    parser.add_option("--output", dest="output_directory", default=None, help="specify output directory---must be unique for each calculation (default: current directory)")
    parser.add_option("--doctests", action="store_true", dest="doctests", default=False, help="run doctests first (default: False)")
    parser.add_option("--randomize_ligand", action="store_true", dest="randomize_ligand", default=False, help="randomize ligand positions and orientations (default: False)")
    parser.add_option("--ignore_signal", action="append", dest="ignore_signals", default=[], help="signals to trap and ignore (default: None)")
    parser.add_option("--platform", dest="platform", default="CPU",help="The platform to use when running simulations")
    parser.add_option("--gpus_per_node", dest="gpus_per_node", type='int', default=None, help="number of GPUs per node to use for complex simulations during MPI calculations")

    # Parse command-line arguments.
    (options, args) = parser.parse_args()
    
    if options.doctests:
        print "Running doctests for all modules..."
        import doctest
        # TODO: Test all modules
        import yank, oldrepex, alchemy, analyze, utils
        (failure_count, test_count) = doctest.testmod(verbose=options.verbose)
        if failure_count == 0:
            print "All doctests pass."
            sys.exit(0)
        else:
            print "WARNING: There were %d doctest failures." % failure_count
            sys.exit(1)

    # Check arguments for validity.
    if not (options.ligand_mol2_filename):
        parser.error("Please supply a ligand in mol2 format")
    if not (options.receptor_pdb_filename):
        parser.error("Please supply the receptor in pdb format")
    if not (options.complex_pdb_filename):
        print("Will combine ligand and receptor")



    # Initialize MPI if requested.
    if options.mpi:
        # Initialize MPI. 
        try:
            from mpi4py import MPI # MPI wrapper
            hostname = os.uname()[1]
            options.mpi = MPI.COMM_WORLD
            if not MPI.COMM_WORLD.rank == 0: 
                options.verbose = False
            MPI.COMM_WORLD.barrier()
            if MPI.COMM_WORLD.rank == 0: print "Initialized MPI on %d processes." % (MPI.COMM_WORLD.size)
        except Exception as e:
            print e
            parser.error("Could not initialize MPI.")


    #create systems
    (ligand_system, ligand_traj, ffxml_name) = create_ligand_system(options.ligand_mol2_filename)
    (receptor_system, receptor_traj) = create_receptor_system(options.receptor_pdb_filename)
    (complex_system, complex_positions) = create_complex_system(ligand_traj,ffxml_name,receptor_traj)



    # Select simulation parameters.
    # TODO: Allow user selection or intelligent automated selection of simulation parameters.
    # NOTE: Simulation paramters are hard-coded for now.
    # NOTE: Simulation parameters will be different for explicit solvent.
    import simtk.openmm.app as app
    nonbondedMethod = app.NoCutoff
    implicitSolvent = app.OBC2
    constraints = app.HBonds
    removeCMMotion = False


    # Determine number of atoms for each system.
    natoms_receptor = receptor_system.getNumParticles()
    natoms_ligand = ligand_system.getNumParticles()
    natoms_complex = complex_system.getNumParticles()
    if (natoms_complex != natoms_ligand + natoms_receptor):
        raise Exception("Number of complex atoms must equal sum of ligand and receptor atoms.")



    # Initialize YANK object.
    from yank import Yank
    yank = Yank(receptor=receptor_system, ligand=ligand_system, complex=complex_system, complex_positions=complex_positions, output_directory=options.output_directory, verbose=options.verbose)

    # Configure YANK object with command-line parameter overrides.
    if options.niterations is not None:
        yank.niterations = int(options.niterations)
    if options.verbose:
        yank.verbose = options.verbose
    if options.online_analysis:
        yank.online_analysis = options.online_analysis
    if options.restraint_type is not None:
        yank.restraint_type = options.restraint_type
    if options.randomize_ligand:
        yank.randomize_ligand = True 
    if options.platform:
        yank.platform = openmm.Platform.getPlatformByName(options.platform)


    # Hard-coded cpuid:gpuid for Exxact 4xGPU nodes
    # TODO: Replace this with something automated
   # cpuid_gpuid_mapping = { 0:0, 1:1, 2:2, 3:3 }
    #ncpus_per_node = None

    # Run calculation.
    if options.mpi:
        # Run MPI version.
        yank.run_mpi(options.mpi, options.gpus_per_node)
    else:
        # Run serial version.
        yank.run()

    # Run analysis.
    results = yank.analyze()

    # Print/write results.
    print results

#=============================================================================================
# MAIN
#=============================================================================================

if __name__ == '__main__':    
    driver()
