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
  yank setup binding amber --ligand_prmtop=PRMTOP --ligand_inpcrd=INPCRD --receptor_prmtop=PRMTOP --receptor_inpcrd=INPCRD --complex_prmtop=PRMTOP --complex_inpcrd=INPCRD --ligname=RESNAME (-s=STORE | --store=STORE) [-i=ITERATIONS | --iterations=ITERATIONS] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--platform=PLATFORM] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [-v | --verbose]
  yank setup binding systembuilder --ligand=FILENAME --receptor=FILENAME [-i=ITERATIONS | --iterations=ITERATIONS] [--restraints <restraint_type>] [--randomize-ligand] [--nbmethod=METHOD] [--gbsa=GBSA] [--constraints=CONSTRAINTS] [--platform=PLATFORM] [--temperature=TEMPERATURE] [--pressure=PRESSURE] [-v | --verbose]
  yank run (-s=STORE | --store=STORE) [-m | --mpi] [-i=ITERATIONS | --iterations ITERATIONS] [--platform=PLATFORM] [--phase=PHASE] [-o | --online-analysis] [-v | --verbose]
  yank status (-s=STORE | --store=STORE) [-v | --verbose]
  yank analyze (-s STORE | --store=STORE) [-v | --verbose]
  yank cleanup (-s=STORE | --store=STORE) [-v | --verbose]

Commands:
  selftest                      Run selftests.
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

Simulation options:
  --restraints=TYPE             Restraint type to add between protein and ligand in implicit solvent ('harmonic', 'flat-bottom') [default: flat-bottom]
  --gbsa=GBSA                   OpenMM GBSA model (HCT, OBC1, OBC2, GBn, GBn2) [default: OBC2]
  --nbmethod=METHOD             OpenMM nonbonded method (NoCutoff, CutoffPeriodic, PME, Ewald) [default: NoCutoff]
  --constraints=CONSTRAINTS     OpenMM constraints (None, HBonds, AllBonds, HAngles) [default: HBonds]
  --phase=PHASE                 Resume only specified phase of calculation ('solvent', 'complex')
  --temperature=TEMPERATURE     Temperature for simulation (in K, or simtk.unit readable string) [default: 298*kelvin]
  --pressure=PRESSURE           Pressure for simulation (in atm, or simtk.unit readable string) [default: 1*atmospheres]

Amber options:
  --ligand_prmtop=PRMTOP        AMBER prmtop file for ligand [default: ligand.prmtop]
  --ligand_inpcrd=INPCRD        AMBER inpcrd file for ligand [default: ligand.inpcrd]
  --receptor_prmtop=PRMTOP      AMBER prmtop file for receptor [default: receptor.prmtop]
  --receptor_inpcrd=INPCRD      AMBER inpcrd file for receptor [default: receptor.inpcrd]
  --complex_prmtop=PRMTOP       AMBER prmtop file for complex [default: complex.prmtop]
  --complex_inpcrd=INPCRD       AMBER inpcrd file for complex [default: complex.inpcrd]
  --ligname=RESNAME             Residue name of ligand [default: MOL]

Systembuilder options:
  --ligand=FILENAME             Ligand filename (formats: mol2, sdf, pdb, smi, iupac)
  --receptor=FILENAME           Receptor filename (formats: mol2, sdf, pdb, smi, iupac)

"""

def main():
    # Parse command-line arguments.
    from docopt import docopt
    import version
    args = docopt(usage, version=version.version)

    dispatched = False # Flag set to True if we have correctly dispatched a command.
    from . import commands # TODO: This would be clearer if we could do 'from yank import commands', but can't figure out how

    # Handle simple arguments.
    if args['--help']:
        print usage
        dispatched = True
    if args['--cite']:
        dispatched = True
        success = commands.cite.dispatch(args)

    # Handle commands.
    command_list = ['selftest', 'setup', 'run', 'status', 'analyze', 'cleanup'] # TODO: Build this list automagically by introspection of commands submodule.
    for command in command_list:
        if args[command]:
            dispatched = True
            success = getattr(commands, command).dispatch(args)

    # If unsuccessful, print usage and exit with an error.
    if not dispatched:
        print usage
        sys.exit(1)
