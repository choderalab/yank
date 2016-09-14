#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Run YANK self tests after installation.

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

import doctest
import pkgutil
import sys

from yank import utils


#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================
usage = """
YANK selftest

Usage:
  yank selftest [-v | --verbose] [-d | --doctests] [-o | --openeye]

Description:
  Run the YANK selftests to check that functions are behaving as expected and if external licenses are available

General Options:
  -v, --verbose                 Print verbose output
  -d, --doctests                Run module doctests
  -o, --openeye                 Check if the optional OpenEye modules and licenses are working corretly

"""
#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    # TODO: Figure out how to run nosetests instead.
    verbose = args['--verbose']

    # Keeping these as comments for if/when we switch to nosetests instead of doctests
    # import nose
    # result = nose.run(argv=['yank', '--nocapture', '--verbosity=2', '--with-timer', '-a', '!slow'] )
    # result = nose.run(argv=['yank', '--nocapture', '--verbosity=2', '-a', '!slow'] )

    if args['--doctests']:
        print("Running doctests for all modules...")

        # Run tests on main module.
        import yank
        (failure_count, test_count) = doctest.testmod(yank, verbose=verbose)
        # Run tests on all submodules.
        package = yank
        prefix = package.__name__ + "."
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
            module = __import__(modname, fromlist="dummy")
            (module_failure_count, module_test_count) = doctest.testmod(module, verbose=verbose)
            failure_count += module_failure_count
            test_count += module_test_count

        # Report results.
        if failure_count == 0:
            print("All doctests pass.")
        else:
            print("WARNING: There were %d doctest failures." % failure_count)
            sys.exit(1)

    if args['--openeye']:
        print("Testing OpenEye Installation")
        try:
            import openeye.examples.openeye_tests as OETests
            OETests.run_test_suite()
        except:
            print("WARNING: OpenEye Tests Failed.")

    return True
