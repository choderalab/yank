#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test command-line interface.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import textwrap
import tempfile
import commands

from yank import utils

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def run_cli(arguments, expected_output=None):
    #cli.main(argv=arguments.split())
    [status, output] = commands.getstatusoutput('yank ' + arguments)

    if status:
        message  = "An error return value (%s) was obtained:\n" % str(status)
        message += "\n"
        message += output
        message += "\n"
        raise Exception(message)

    if expected_output:
        if output != expected_output:
            message  = "Output differs from expected output.\n"
            message += "\n"
            message += "Expected output:\n"
            message += expected_output
            message += "\n"
            message += "Actual output:\n"
            message += output
            message += "\n"
            raise Exception(message)

def test_help():
    run_cli('--help')

def test_selftest():
    run_cli('selftest')

def test_prepare_binding():
    # NOTE: switched to yank p-xylene from openmmtools T4-lysozyme because of yank bugs.
    dirname = utils.get_data_filename("../examples/p-xylene-implicit/setup/")  # Could only figure out how to install things like yank.egg/examples/, rather than yank.egg/yank/examples/
    with utils.temporary_directory() as store_dir:
        run_cli('prepare binding amber --setupdir=%(dirname)s --ligand="resname MOL" --store %(store_dir)s --gbsa OBC1' % vars())

def test_script_yaml():
    """Check that yank script --yamlscript command works."""
    setup_dir = utils.get_data_filename(os.path.join('..', 'examples', 'p-xylene-implicit', 'setup'))
    pxylene_path = os.path.join(setup_dir, 'ligand.tripos.mol2')
    lysozyme_path = os.path.join(setup_dir, 'receptor.pdbfixer.pdb')
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            number_of_iterations: 1
            output_dir: {}
        molecules:
            T4lysozyme:
                filepath: {}
                parameters: oldff/leaprc.ff99SBildn
            p-xylene:
                filepath: {}
                parameters: antechamber
        solvents:
            vacuum:
                nonbonded_method: NoCutoff
        experiment:
            components:
                receptor: T4lysozyme
                ligand: p-xylene
                solvent: vacuum
        """.format(tmp_dir, lysozyme_path, pxylene_path)
        yaml_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            # Check that handles no options
            yaml_file.write(textwrap.dedent(yaml_content))
            yaml_file.close()
            run_cli('script --yaml={}'.format(yaml_file.name))
        finally:
            os.remove(yaml_file.name)
