#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
YANK command-line interface (cli)

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

import sys
import docopt

#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================

usage = """\
YANK

Usage:
  yank [-h | --help] [-c | --cite]
  yank selftest [-v | --verbose]
  yank platforms
  yank setup binding amber --setupdir=DIRECTORY --ligname=RESNAME (-s=STORE | --store=STORE) [-i=ITERATIONS | --iterations=ITERATIONS] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--platform=PLATFORM] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [--minimize] [-v | --verbose]
  yank setup binding systembuilder --ligand=FILENAME --receptor=FILENAME [-i=ITERATIONS | --iterations=ITERATIONS] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--platform=PLATFORM] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [-v | --verbose]
  yank run (-s=STORE | --store=STORE) [-m | --mpi] [-i=ITERATIONS | --iterations ITERATIONS] [--platform=PLATFORM] [--phase=PHASE] [-o | --online-analysis] [-v | --verbose]
  yank status (-s=STORE | --store=STORE) [-v | --verbose]
  yank analyze (-s STORE | --store=STORE) [-v | --verbose]
  yank cleanup (-s=STORE | --store=STORE) [-v | --verbose]

Commands:
  selftest                      Run selftests.
  platforms                     List available OpenMM platforms.
  setup binding amber           Set up binding free energy calculation using AMBER.
  run                           Run the calculation that has been set up
  status                        Get the current status
  analyze                       Analyze data
  cleanup                       Clean up (delete) run files.

General options:
  -h, --help                    Print command line help
  -c, --cite                    Print relevant citations
  -i NITER, --iterations NITER  Number of iterations to run [default: 1000]
  --randomize-ligand            Randomize initial ligand positions if specified
  -v, --verbose                 Print verbose output
  -s=STORE, --store=STORE       Storage directory for NetCDF data files.
  -o, --online-analysis         Enable on-the-fly analysis
  --platform=PLATFORM           OpenMM Platform to use (Reference, CPU, OpenCL, CUDA) [default: None]
  --minimize                    Minimize configurations before running simulation.

Simulation options:
  --restraints=TYPE             Restraint type to add between protein and ligand in implicit solvent ('harmonic', 'flat-bottom') [default: flat-bottom]
  --gbsa=GBSA                   OpenMM GBSA model (HCT, OBC1, OBC2, GBn, GBn2) [default: OBC2]
  --nbmethod=METHOD             OpenMM nonbonded method (NoCutoff, CutoffPeriodic, PME, Ewald) [default: NoCutoff]
  --constraints=CONSTRAINTS     OpenMM constraints (None, HBonds, AllBonds, HAngles) [default: HBonds]
  --phase=PHASE                 Resume only specified phase of calculation ('solvent', 'complex')
  --temperature=TEMPERATURE     Temperature for simulation (in K, or simtk.unit readable string) [default: 298*kelvin]
  --pressure=PRESSURE           Pressure for simulation (in atm, or simtk.unit readable string) [default: 1*atmospheres]

Amber options:
  --setupdir=DIRECTORY          Setup directory to look for AMBER {receptor|ligand|complex}.{prmtop|inpcrd} files.
  --ligname=RESNAME             Residue name of ligand [default: MOL]

Systembuilder options:
  --ligand=FILENAME             Ligand filename (formats: mol2, sdf, pdb, smi, iupac, cdx)
  --receptor=FILENAME           Receptor filename (formats: mol2, sdf, pdb, smi, iupac, cdx)

"""

# TODO: Add optional arguments that we can use to override sys.argv for testing purposes.
def main(argv=None):
    # Parse command-line arguments.
    from docopt import docopt
    import version
    args = docopt(usage, version=version.version, argv=argv)

    dispatched = False # Flag set to True if we have correctly dispatched a command.
    from . import commands # TODO: This would be clearer if we could do 'from yank import commands', but can't figure out how

    # Handle simple arguments.
    if args['--help']:
        print usage
        dispatched = True
    if args['--cite']:
        dispatched = commands.cite.dispatch(args)

    # Handle commands.
    command_list = ['selftest', 'platforms', 'setup', 'run', 'status', 'analyze', 'cleanup'] # TODO: Build this list automagically by introspection of commands submodule.
    for command in command_list:
        if args[command]:
            dispatched = getattr(commands, command).dispatch(args)

    # If unsuccessful, print usage and exit with an error.
    if not dispatched:
        print usage
        return True

    # Indicate success.
    return False
