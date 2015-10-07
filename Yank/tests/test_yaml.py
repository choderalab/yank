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

from nose.tools import raises

from yank.yamlbuild import *

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

def test_set_dict_path():
    """Test utility function set_dict_path()."""
    test = {'a': 2}
    test_nested = {'a': {'b': 2}}
    set_dict_path(test, ('a',), 3)
    assert test == {'a': 3}
    set_dict_path(test_nested, ('a','b'), 3)
    assert test_nested == {'a': {'b': 3}}
    set_dict_path(test_nested, ('a',), 5)
    assert test_nested == {'a': 5}

def test_not_iterable_container():
    """Test utility function not_iterable_container()."""
    assert not_iterable_container(3) == True
    assert not_iterable_container('ciao') == True
    assert not_iterable_container([1, 2, 3]) == False

def test_find_leaves():
    """Test utility function find_leaves()."""
    simple_tree = {'simple': {'scalar': 1,
                              'vector': [2, 3, 4],
                              'nested': {'leaf': ['a', 'b', 'c']}}}
    leaf_paths, leaf_vals = find_leaves(simple_tree)
    assert leaf_paths == [('simple', 'scalar'), ('simple', 'vector'),
                          ('simple', 'nested', 'leaf')]
    assert leaf_vals == [1, [2, 3, 4], ['a', 'b', 'c']]

def test_expand_tree():
    """Test utility function expand_tree()."""
    simple_tree = {'simple': {'scalar': 1,
                              'vector': [2, 3, 4],
                              'nested': {'leaf': ['a', 'b', 'c']}}}
    result = [{'simple': {'scalar': 1, 'vector': 2, 'nested': {'leaf': 'a'}}},
              {'simple': {'scalar': 1, 'vector': 2, 'nested': {'leaf': 'b'}}},
              {'simple': {'scalar': 1, 'vector': 2, 'nested': {'leaf': 'c'}}},
              {'simple': {'scalar': 1, 'vector': 3, 'nested': {'leaf': 'a'}}},
              {'simple': {'scalar': 1, 'vector': 3, 'nested': {'leaf': 'b'}}},
              {'simple': {'scalar': 1, 'vector': 3, 'nested': {'leaf': 'c'}}},
              {'simple': {'scalar': 1, 'vector': 4, 'nested': {'leaf': 'a'}}},
              {'simple': {'scalar': 1, 'vector': 4, 'nested': {'leaf': 'b'}}},
              {'simple': {'scalar': 1, 'vector': 4, 'nested': {'leaf': 'c'}}}]
    assert result == [exp for exp in expand_tree(simple_tree)]

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
        equilibrate: yes
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

@raises(YamlParseError)
def test_yaml_wrong_option_value():
    """Check that YamlBuilder raises an exception when option type is wrong."""
    yaml_content = """
    ---
    options:
        equilibrate: 1000
    """
    parse_yaml_str(yaml_content)
