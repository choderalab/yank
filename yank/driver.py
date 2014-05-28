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
from systembuilder import *
import os


def driver():
    # Initialize command-line argument parser.
    verbose = True
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





    # Select simulation parameters.
    # TODO: Allow user selection or intelligent automated selection of simulation parameters.
    # NOTE: Simulation paramters are hard-coded for now.
    # NOTE: Simulation parameters will be different for explicit solvent.
    import simtk.openmm.app as app
    nonbondedMethod = app.NoCutoff
    implicitSolvent = app.OBC2
    constraints = app.HBonds
    removeCMMotion = False


    #make ligand, receptor, complex objects
    ligand = Mol2SystemBuilder(options.ligand_mol2_filename)
    receptor = PDBSystemBuilder(options.receptor_pdb_filename)
    complex_system = ComplexSystemBuilder(ligand,receptor)
    print len(complex_system.positions)
    # Initialize YANK object.
    print complex_system.what_yank_wants[0:3,:]
    from yank import Yank
    yank = Yank(receptor=receptor.system, ligand=ligand.system, complex=complex_system.system, complex_positions=[complex_system.what_yank_wants], output_directory=options.output_directory, verbose=options.verbose)

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

    # Run calculation.brbz
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
