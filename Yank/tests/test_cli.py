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
import subprocess

import openmoltools as omt

from yank import utils

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def run_cli(arguments, expected_output=None):
    #cli.main(argv=arguments.split())
    command = 'yank ' + arguments
    [stoutdata, sterrdata] = subprocess.Popen(command.split()).communicate()

    # TODO: Interprety suprocess data better
    if sterrdata:
        message  = "An error return value (%s) was obtained:\n" % str(sterrdata)
        message += "\n"
        message += stoutdata
        message += "\n"
        raise Exception(message)

    if expected_output:
        if stoutdata != expected_output:
            message  = "Output differs from expected output.\n"
            message += "\n"
            message += "Expected output:\n"
            message += expected_output
            message += "\n"
            message += "Actual output:\n"
            message += stoutdata
            message += "\n"
            raise Exception(message)

def test_help():
    run_cli('--help')

def test_selftest():
    try:
        run_cli('selftest')
    except ImportError as e:
        # Trap the libOpenCl error
        if "libOpenCL.so" in e.message:
            print("Failed to load OpenCL. If this is an expected reult, carry on, if not, please debug!")
        else:
            raise(e)

def notest_prepare_binding():
    dirname = utils.get_data_filename("../examples/benzene-toluene-implicit/setup/")  # Could only figure out how to install things like yank.egg/examples/, rather than yank.egg/yank/examples/
    with omt.utils.temporary_directory() as store_dir:
        run_cli('prepare binding amber --setupdir=%(dirname)s --ligand="resname TOL" --store %(store_dir)s --gbsa OBC1' % vars())

def test_script_yaml():
    """Check that yank script --yaml command works."""
    setup_dir = utils.get_data_filename(os.path.join('..', 'Yank', 'tests', 'data', 'p-xylene-implicit'))
    pxylene_path = os.path.join(setup_dir, 'p-xylene.mol2')
    lysozyme_path = os.path.join(setup_dir, '181L-pdbfixer.pdb')
    with omt.utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            number_of_iterations: 1
            output_dir: '.'
        molecules:
            T4lysozyme:
                filepath: {}
            p-xylene:
                filepath: {}
                antechamber:
                    charge_method: bcc
        solvents:
            vacuum:
                nonbonded_method: NoCutoff
        protocols:
            absolute-binding:
                complex:
                    alchemical_path:
                        lambda_electrostatics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
                        lambda_sterics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
                solvent:
                    alchemical_path:
                        lambda_electrostatics: [1.0, 0.8, 0.6, 0.3, 0.0]
                        lambda_sterics: [1.0, 0.8, 0.6, 0.3, 0.0]
        systems:
            system:
                receptor: T4lysozyme
                ligand: p-xylene
                solvent: vacuum
                leap:
                    parameters: [leaprc.gaff, leaprc.ff14SB]
        experiments:
            system: system
            protocol: absolute-binding
        """.format(lysozyme_path, pxylene_path)

        yaml_file_path = os.path.join(tmp_dir, 'yank.yaml')
        with open(yaml_file_path, 'w') as f:
            f.write(textwrap.dedent(yaml_content))
        run_cli('script --yaml={}'.format(yaml_file_path))
