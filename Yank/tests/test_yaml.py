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

from yank.yamlbuild import YamlBuilder, YamlParseError

#=============================================================================================
# SUBROUTINES FOR TESTING
#=============================================================================================

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
        timestep: 2.0 * femtoseconds
        nsteps_per_iteration: 2500
        number_of_iterations: 10
        minimize: false
        equilibrate: true
        equilibration_timestep: 1.0 * femtoseconds
        number_of_equilibration_iterations: 1000
    """

    yaml_builder = parse_yaml_str(yaml_content)
    assert len(yaml_builder.options) == 8
    assert yaml_builder.options['title'] == 'Test YANK YAML YAY!'
    assert yaml_builder.options['timestep'] == 2.0 * unit.femtoseconds
    assert yaml_builder.options['nsteps_per_iteration'] == 2500
    assert yaml_builder.options['number_of_iterations'] == 10
    assert yaml_builder.options['minimize'] is False
    assert yaml_builder.options['equilibrate'] is True
    assert yaml_builder.options['equilibration_timestep'] == 1.0 * unit.femtoseconds
    assert yaml_builder.options['number_of_equilibration_iterations'] == 1000

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
