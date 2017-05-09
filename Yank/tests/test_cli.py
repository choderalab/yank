#!/usr/bin/python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Test command-line interface.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import os
import textwrap
import subprocess

import openmoltools as omt

from yank import utils

# =============================================================================================
# UNIT TESTS
# =============================================================================================


def run_cli(arguments, expected_output=None):
    """Generic helper to run command line arguments"""
    # cli.main(argv=arguments.split())
    command = 'yank ' + arguments
    [stoutdata, sterrdata] = subprocess.Popen(command.split()).communicate()

    # TODO: Interprety suprocess data better
    if sterrdata:
        message = "An error return value (%s) was obtained:\n" % str(sterrdata)
        message += "\n"
        message += stoutdata
        message += "\n"
        raise Exception(message)

    if expected_output:
        if stoutdata != expected_output:
            message = "Output differs from expected output.\n"
            message += "\n"
            message += "Expected output:\n"
            message += expected_output
            message += "\n"
            message += "Actual output:\n"
            message += stoutdata
            message += "\n"
            raise Exception(message)


def test_help():
    """Test that the help command works"""
    run_cli('--help')


def test_cite():
    """Test that the cite command works"""
    run_cli('--cite')


def test_selftest():
    """Test that the selftest command works"""
    try:
        run_cli('selftest')
    except ImportError as e:
        # Trap the libOpenCl error
        if "libOpenCL.so" in e.message:
            print("Failed to load OpenCL. If this is an expected result, carry on, if not, please debug!")
        else:
            raise e


def test_script_yaml():
    """Check that yank script --yaml command works."""
    setup_dir = utils.get_data_filename(os.path.join('tests', 'data', 'p-xylene-implicit'))
    pxylene_path = os.path.join(setup_dir, 'p-xylene.mol2')
    lysozyme_path = os.path.join(setup_dir, '181L-pdbfixer.pdb')
    yaml_content = """
        ---
        options:
            number_of_iterations: 0
            output_dir: '.'
            resume_setup: yes
            resume_simulation: yes
            minimize: no
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
                        lambda_electrostatics: [1.0, 0.5, 0.0]
                        lambda_sterics: [1.0, 0.5, 0.0]
                solvent:
                    alchemical_path:
                        lambda_electrostatics: [1.0, 0.5, 0.0]
                        lambda_sterics: [1.0, 0.5, 0.0]
        systems:
            system:
                receptor: T4lysozyme
                ligand: p-xylene
                solvent: vacuum
                leap:
                    parameters: [leaprc.gaff, oldff/leaprc.ff14SB]
        experiments:
            system: system
            protocol: absolute-binding
            restraint:
                type: FlatBottom
        """.format(lysozyme_path, pxylene_path)
    with omt.utils.temporary_directory() as tmp_dir:
        yaml_file_path = os.path.join(tmp_dir, 'yank.yaml')
        with open(yaml_file_path, 'w') as f:
            f.write(textwrap.dedent(yaml_content))
        run_cli('script --yaml={}'.format(yaml_file_path))
    # Test with override
    with omt.utils.temporary_directory() as tmp_dir:
        yaml_file_path = os.path.join(tmp_dir, 'yank.yaml')
        with open(yaml_file_path, 'w') as f:
            f.write(textwrap.dedent(yaml_content))
        extension = "options:number_of_iterations:1"
        run_cli('script --yaml={} -o {}'.format(yaml_file_path, extension))
