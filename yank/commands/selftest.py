#!/usr/local/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Run YANK self tests after installation.

"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

# Module imports handled in individual functions since CLI should be faster to boot up

# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================
usage = """
YANK selftest

Usage:
  yank selftest  [-d | --doctests] [-n | --nosetests] [--verbosity=#] [(--skip (platforms | openeye))]...

Description:
  Run the YANK selftests to check that functions are behaving as expected and if external licenses are available

General Options:
  -d, --doctests                Run module doctests
  -n, --nosetests               Run the nosetests (slow) on YANK
  --verbosity=#                 Optional verbosity level of nosetests OR verbose doctests [default: 1]
                                Does nothing without -n or -d
  --skip                        Skips a named selftests (platforms OR openeye) for speed
                                May be specified multiple times, once per TEST

"""
# =============================================================================================
# COMMAND DISPATCH
# =============================================================================================


class LicenseError(Exception):
    """Error raised by a missing License."""
    pass


def dispatch(args):

    import re
    import doctest
    import pkgutil
    import subprocess
    import simtk.openmm as mm
    from .. import version
    from . import platforms

    # Determine verbosity in advance
    # TODO: Figure out how to get -v back in to command and allow -vv and -vvv
    # nosetests: -v == --verbosity=2
    # Assuming that verbosity = 1 (or no -v) is no verbosity for doctests
    # verbosity = max(args['-v'] + 1, int(args['--verbosity']))
    verbosity = int(args['--verbosity'])
    # Header
    print("\n")
    print("YANK Selftest")
    print("-------------")

    # yank Version
    print("yank Version %s \n" % version.version)

    # OpenMM Platforms
    if not (args['platforms'] > 0):  # Don't need to check for --skip since invalid without argument
        platforms.dispatch(None)
        # Errors
        platform_errors = mm.Platform.getPluginLoadFailures()
        if len(platform_errors) > 0:  # This check only required to make header
            print("************************************************")
            print("\nWarning! There were OpenMM Platform Load Errors!")
            print("************************************************")
            for e in platform_errors:
                print(e)
            print("************************************************")
            print("************************************************")
    else:
        print("Skipped OpenMM Platform Test")

    # Space out tests
    print("\n")

    # OpenEye checks
    if not (args['openeye'] > 0):
        try:
            import openeye
            import openeye.examples.openeye_tests as OETests
            print("OpenEye version {} Found! Checking install...".format(openeye.__version__))
            # OETests.run_test_suite([])  # Disabled for now as its slow
            # Check that the required tools work
            from ..utils import is_openeye_installed
            complete_tool_set = set(['oechem', 'oequacpac', 'oeiupac', 'oeomega'])
            unlicensed_tools = complete_tool_set.copy()
            for tool in complete_tool_set:
                if is_openeye_installed(oetools=(tool,)):
                    unlicensed_tools.remove(tool)
            # If not the empty set, raise a unique error
            if unlicensed_tools != set():
                raise LicenseError
        except LicenseError:
            if unlicensed_tools == complete_tool_set:
                message = "No valid OpenEye licenses were found for the tools YANK uses, despite an OpenEye install " \
                          "being found.\nPlease check you have at least one " \
                          "of the following tools to use some of YANK's OpenEye features:\n    " \
                          "{}"
            else:
                message = "OpenEye is missing licenses to some of the tools YANK can use.\n" \
                          "Some features of the OpenEye features YANK can use will be available, but not all.\n" \
                          "    Missing Licenses: {}"
            print(message.format(unlicensed_tools))
        except:
            # Broad except to trap any thing else that went wrong in the OpenEye test
            print("Valid OpenEye install not found")
            print("Not required, but please check install if you expected it")
    else:
        print("Skipped OpenEye Tests")
    print("\n")

    # NVIDIA-SMI calls
    print("Checking GPU Computed Mode (if present)...")
    try:
        nvidia_output = subprocess.check_output('nvidia-smi -q -d COMPUTE', shell=True)
    except subprocess.CalledProcessError as e:
        print("nvidia-smi had an issue, could not find CUDA cards, however this may be expected on your system.")
    else:
        n_cards = 0
        card_modes = []
        # Cast subprocess output to str
        try:
            # Try decoding byte string
            nvidia_output = nvidia_output.decode()
        except AttributeError:
            # Already a string
            pass
        finally:
            # Ensure its a string in case decode didn't work
            # Only a problem in case of odd encoding but all it should do is add odd character to the start of string
            # which is not really a problem since its not what we are searching for here
            nvidia_output = str(nvidia_output)

        split_nvidia_output = nvidia_output.split('\n')
        for line in split_nvidia_output:
            match = re.search('(?:Compute[^:]*:\s+)(\w+)', line)
            if match:
                n_cards += 1
                card_modes.append(match.group(1))
        if n_cards == 0:
            print("nvidia-smi returned 'Compute' search, but no cards matched query pattern.\n"
                  "Please run `nvidia-smi` yourself to confirm the Compute Mode is in shared/Default")
        else:
            print("Found {} NVIDIA GPUs in the following modes: [".format(n_cards) + ', '.join(card_modes) + "]\n"
                  "These should all be in shared/Default mode for YANK to use them")

    # Run nosetests
    # Note: These will not run during standard nosetests because they must be explicitly called
    # i.e. no infinite nosetests loop
    if args['--nosetests']:
        # Clear some lines
        print("\n")
        # Alert User
        print("******************************************")
        print("Nosetests invoked! This will take a while!")
        print("******************************************")
        import nose
        try:  # Check for timer install
            result = nose.run(argv=['yank', '--nocapture', '--verbosity={}'.format(verbosity),
                                    '--with-timer', '-a', '!slow'])
        except:
            result = nose.run(argv=['yank', '--nocapture', '--verbosity={}'.format(verbosity), '-a', '!slow'])
        print("\n")

    # Doctests
    if args['--doctests']:
        # Alert User
        print("*****************************************")
        print("Doctests invoked! This will take a while!")
        print("*****************************************")
        # Run tests on main module.
        import yank  # NOT "from .. import yank" since we want to run on the whole module
        if verbosity > 1:
            verbose = True
        else:
            verbose = False
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
            print("WARNING: There were {} doctest failures.".format(failure_count))
        print("\n")

    # Helpful end test
    print("YANK Selftest complete.\nThank you for using YANK!\n")
    return True
