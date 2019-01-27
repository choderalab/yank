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
            default_number_of_iterations: 0
            output_dir: '.'
            resume_setup: yes
            resume_simulation: no
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

        # Test option overriding.
        run_cli('script --yaml={} -o options:resume_simulation:yes'.format(yaml_file_path))

def test_script_yaml_status():
    """Check that 'yank script --yaml --status' works."""
    setup_dir = utils.get_data_filename(os.path.join('tests', 'data', 'cyclodextrin'))
    host_path = os.path.join(setup_dir, 'host-bcd.mol2')
    guest_path = os.path.join(setup_dir, 'guest-s17.mol2')
    yaml_content = """\
        ---
        options:
            output_dir: 'output'
            resume_setup: yes
            resume_simulation: no
            minimize: no
            verbose: no
            switch_experiment_interval: 20

        molecules:
            host:
                filepath: {}
                antechamber:
                    charge_method: null
            guest:
                filepath: {}
                antechamber:
                    charge_method: null

        mcmc_moves:
            langevin:
                type: LangevinSplittingDynamicsMove
                timestep: 4.0*femtosecond
                collision_rate: 1.0/picosecond
                reassign_velocities: yes
                splitting: 'V R O R V'
                n_steps: 10
                n_restart_attempts: 4

        samplers:
            repex:
                type: ReplicaExchangeSampler
                mcmc_moves: langevin
                number_of_iterations: 40

        solvents:
            vacuum:
                nonbonded_method: NoCutoff

        protocols:
            absolute-binding:
                complex:
                    alchemical_path:
                        lambda_restraints: [0.0, 1.0]
                        lambda_electrostatics: [1.0, 0.0]
                        lambda_sterics: [1.0, 0.0]
                solvent:
                    alchemical_path:
                        lambda_electrostatics: [1.0, 0.0]
                        lambda_sterics: [1.0, 0.0]
        systems:
            system:
                receptor: host
                ligand: guest
                solvent: vacuum
                leap:
                    parameters: [leaprc.gaff, oldff/leaprc.ff14SB]
        experiments:
            system: system
            sampler: repex
            protocol: absolute-binding
            restraint:
                type: Harmonic
        """.format(host_path, guest_path)

    with omt.utils.temporary_directory() as tmp_dir:
        yaml_file_path = os.path.join(tmp_dir, 'yank.yaml')
        with open(yaml_file_path, 'w') as f:
            f.write(textwrap.dedent(yaml_content))
        # Test status output.
        run_cli('script --yaml={} --status'.format(yaml_file_path))
        # Ensure pickle file is found
        output_path = os.path.join(tmp_dir, 'output', 'experiments')
        filenames = os.listdir(output_path)
        if 'status.pkl' not in filenames:
            msg = 'Status file not found in experiment directory\n'
            msg += 'contents: {}'.format(filenames)
            raise Exception(msg)
