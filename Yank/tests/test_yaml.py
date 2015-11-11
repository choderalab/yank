#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test YAML functions.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import tempfile
import textwrap

from simtk import unit
from nose.tools import raises

from yank.utils import temporary_directory
from yank.yamlbuild import YamlBuilder, YamlParseError

#=============================================================================================
# SUBROUTINES FOR TESTING
#=============================================================================================

def example_dir():
    """Return the absolute path to the Yank examples directory."""
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_file_dir, '..', '..', 'examples')

def parse_yaml_str(yaml_content):
    """Parse the YAML string and return the YamlBuilder object used."""
    yaml_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        # Check that handles no options
        yaml_file.write(textwrap.dedent(yaml_content))
        yaml_file.close()
        yaml_builder = YamlBuilder(yaml_file.name)
    finally:
        os.remove(yaml_file.name)
    return yaml_builder

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def test_yaml_parsing():
    """Check that YAML file is parsed correctly."""

    # Parser handles no options
    yaml_content = """
    ---
    test: 2
    """
    yaml_builder = parse_yaml_str(yaml_content)
    assert len(yaml_builder.options) == 0

    # Correct parsing
    yaml_content = """
    ---
    metadata:
        title: Test YANK YAML YAY!

    options:
        verbose: true
        mpi: yes
        platform: CUDA
        precision: mixed
        resume: true
        output_dir: /path/to/output/
        restraint_type: harmonic
        randomize_ligand: yes
        randomize_ligand_sigma_multiplier: 2.0
        randomize_ligand_close_cutoff: 1.5 * angstrom
        mc_displacement_sigma: 10.0 * angstroms
        collision_rate: 5.0 / picosecond
        constraint_tolerance: 1.0e-6
        timestep: 2.0 * femtosecond
        nsteps_per_iteration: 2500
        number_of_iterations: 1000.999
        equilibration_timestep: 1.0 * femtosecond
        number_of_equilibration_iterations: 100
        minimize: False
        minimize_tolerance: 1.0 * kilojoules_per_mole / nanometers
        minimize_maxIterations: 0
        replica_mixing_scheme: swap-all
        online_analysis: no
        online_analysis_min_iterations: 20
        show_energies: True
        show_mixing_statistics: yes
    """

    yaml_builder = parse_yaml_str(yaml_content)
    assert len(yaml_builder.options) == 21

    # Check correct types
    assert yaml_builder.options['replica_mixing_scheme'] == 'swap-all'
    assert yaml_builder.options['timestep'] == 2.0 * unit.femtoseconds
    assert yaml_builder.options['constraint_tolerance'] == 1.0e-6
    assert yaml_builder.options['nsteps_per_iteration'] == 2500
    assert type(yaml_builder.options['nsteps_per_iteration']) is int
    assert yaml_builder.options['number_of_iterations'] == 1000
    assert type(yaml_builder.options['number_of_iterations']) is int
    assert yaml_builder.options['minimize'] is False
    assert yaml_builder.options['show_mixing_statistics'] is True

@raises(YamlParseError)
def test_yaml_unknown_options():
    """Check that YamlBuilder raises an exception when an unknown option is found."""
    # Raise exception on unknown options
    yaml_content = """
    ---
    options:
        wrong_option: 3
    """
    parse_yaml_str(yaml_content)

@raises(YamlParseError)
def test_yaml_wrong_option_value():
    """Check that YamlBuilder raises an exception when option type is wrong."""
    yaml_content = """
    ---
    options:
        minimize: 100
    """
    parse_yaml_str(yaml_content)

def test_yaml_mol2_antechamber():
    """Test antechamber setup of molecule files."""
    imatinib_path = os.path.join(example_dir(), 'abl-imatinib-explicit',
                                 'setup', 'STI02.mol2')

    with temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            imatinib:
                filepath: {}
                parameters: antechamber
        """.format(tmp_dir, imatinib_path)

        yaml_builder = parse_yaml_str(yaml_content)
        yaml_builder._setup_molecule('imatinib')

        output_dir = os.path.join(tmp_dir, YamlBuilder.SETUP_MOLECULES_DIR, 'imatinib')
        assert os.path.exists(os.path.join(output_dir, 'imatinib.gaff.mol2'))
        assert os.path.exists(os.path.join(output_dir, 'imatinib.frcmod'))
        assert os.path.getsize(os.path.join(output_dir, 'imatinib.gaff.mol2')) > 0
        assert os.path.getsize(os.path.join(output_dir, 'imatinib.frcmod')) > 0
