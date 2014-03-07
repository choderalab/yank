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

#=============================================================================================
# DRIVER
#=============================================================================================

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
    parser.add_option("--ligand_prmtop", dest="ligand_prmtop_filename", default=None, help="ligand Amber parameter file", metavar="LIGAND_PRMTOP")
    parser.add_option("--receptor_prmtop", dest="receptor_prmtop_filename", default=None, help="receptor Amber parameter file", metavar="RECEPTOR_PRMTOP")    
    parser.add_option("--ligand_crd", dest="ligand_crd_filename", default=None, help="ligand Amber crd file", metavar="LIGAND_CRD")
    parser.add_option("--receptor_crd", dest="receptor_crd_filename", default=None, help="receptor Amber crd file", metavar="RECEPTOR_CRD")
    parser.add_option("--ligand_mol2", dest="ligand_mol2_filename", default=None, help="ligand mol2 file (can contain multiple conformations)", metavar="LIGAND_MOL2")
    parser.add_option("--receptor_pdb", dest="receptor_pdb_filename", default=None, help="receptor PDB file (can contain multiple MODELs)", metavar="RECEPTOR_PDB")
    parser.add_option("--complex_prmtop", dest="complex_prmtop_filename", default=None, help="complex Amber parameter file", metavar="COMPLEX_PRMTOP")
    parser.add_option("--complex_crd", dest="complex_crd_filename", default=None, help="complex Amber crd file", metavar="COMPLEX_CRD")
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
    if not (options.ligand_prmtop_filename and options.receptor_prmtop_filename):
        parser.error("ligand and receptor prmtop files must be specified")        
    if not (bool(options.ligand_mol2_filename) ^ bool(options.ligand_crd_filename) ^ bool(options.complex_pdb_filename) ^ bool(options.complex_crd_filename)):
        parser.error("Ligand coordinates must be specified through only one of --ligand_crd, --ligand_mol2, --complex_crd, or --complex_pdb.")
    if not (bool(options.receptor_pdb_filename) ^ bool(options.receptor_crd_filename) ^ bool(options.complex_pdb_filename) ^ bool(options.complex_crd_filename)):
        parser.error("Receptor coordinates must be specified through only one of --receptor_crd, --receptor_pdb, --complex_crd, or --complex_pdb.")    
    if not (options.complex_prmtop_filename):
        parser.error("Please specify --complex_prmtop [complex_prmtop_filename] argument.")

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

    # Select simulation parameters.
    # TODO: Allow user selection or intelligent automated selection of simulation parameters.
    # NOTE: Simulation paramters are hard-coded for now.
    # NOTE: Simulation parameters will be different for explicit solvent.
    import simtk.openmm.app as app
    nonbondedMethod = app.NoCutoff
    implicitSolvent = app.OBC2
    constraints = app.HBonds
    removeCMMotion = False

    # Create System objects for ligand and receptor.
    ligand_system = app.AmberPrmtopFile(options.ligand_prmtop_filename).createSystem(nonbondedMethod=nonbondedMethod, implicitSolvent=implicitSolvent, constraints=constraints, removeCMMotion=removeCMMotion)
    receptor_system = app.AmberPrmtopFile(options.receptor_prmtop_filename).createSystem(nonbondedMethod=nonbondedMethod, implicitSolvent=implicitSolvent, constraints=constraints, removeCMMotion=removeCMMotion)
    complex_system = app.AmberPrmtopFile(options.complex_prmtop_filename).createSystem(nonbondedMethod=nonbondedMethod, implicitSolvent=implicitSolvent, constraints=constraints, removeCMMotion=removeCMMotion)

    # Determine number of atoms for each system.
    natoms_receptor = receptor_system.getNumParticles()
    natoms_ligand = ligand_system.getNumParticles()
    natoms_complex = complex_system.getNumParticles()
    if (natoms_complex != natoms_ligand + natoms_receptor):
        raise Exception("Number of complex atoms must equal sum of ligand and receptor atoms.")

    # Read ligand and receptor coordinates.
    ligand_coordinates = list()
    receptor_coordinates = list()
    complex_coordinates = list()
    if (options.complex_crd_filename or options.complex_pdb_filename):
        # Read coordinates for whole complex.
        if options.complex_crd_filename:
            coordinates = read_amber_crd(options.complex_crd_filename, natoms_complex, options.verbose)
            complex_coordinates.append(coordinates)
        else:
            try:
                coordinates_list = read_openeye_crd(options.complex_pdb_filename, natoms_complex, options.verbose)
            except:
                coordinates_list = read_pdb_crd(options.complex_pdb_filename, natoms_complex, options.verbose)
            complex_coordinates += coordinates_list
    elif options.ligand_crd_filename:
        coordinates = read_amber_crd(options.ligand_crd_filename, natoms_ligand, options.verbose)
        coordinates = units.Quantity(numpy.array(coordinates / coordinates.unit), coordinates.unit)
        ligand_coordinates.append(coordinates)
    elif options.ligand_mol2_filename:
        coordinates_list = read_openeye_crd(options.ligand_mol2_filename, natoms_ligand, options.verbose)
        ligand_coordinates += coordinates_list
    elif options.receptor_crd_filename:
        coordinates = read_amber_crd(options.receptor_crd_filename, natoms_receptor, options.verbose)
        coordinates = units.Quantity(numpy.array(coordinates / coordinates.unit), coordinates.unit)
        receptor_coordinates.append(coordinates)
    elif options.receptor_pdb_filename:
        try:
            coordinates_list = read_openeye_crd(options.receptor_pdb_filename, natoms_receptor, options.verbose)
        except:
            coordinates_list = read_pdb_crd(options.receptor_pdb_filename, natoms_receptor, options.verbose)
        receptor_coordinates += coordinates_list

    # Assemble complex coordinates if we haven't read any.
    if len(complex_coordinates)==0:
        for x in receptor_coordinates:
            for y in ligand_coordinates:
                z = units.Quantity(numpy.zeros([natoms_complex,3]), units.angstroms)
                z[0:natoms_receptor,:] = x[:,:]
                z[natoms_receptor:natoms_complex,:] = y[:,:]
                complex_coordinates.append(z)

    # Initialize YANK object.
    from yank import Yank
    yank = Yank(receptor=receptor_system, ligand=ligand_system, complex=complex_system, complex_coordinates=complex_coordinates, output_directory=options.output_directory, verbose=options.verbose)

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

    # Hard-coded cpuid:gpuid for Exxact 4xGPU nodes
    # TODO: Replace this with something automated
    cpuid_gpuid_mapping = { 0:0, 1:1, 2:2, 3:3 } 
    ncpus_per_node = None

    # Run calculation.
    if options.mpi:
        # Run MPI version.
        yank.run_mpi(options.mpi, cpuid_gpuid_mapping=cpuid_gpuid_mapping, ncpus_per_node=ncpus_per_node)
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
