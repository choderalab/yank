#!/usr/local/bin/env python

"""
Test various utility functions.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

from nose import tools
from yank.utils import *

#=============================================================================================
# TESTING FUNCTIONS
#=============================================================================================

def test_is_iterable_container():
    """Test utility function not_iterable_container()."""
    assert is_iterable_container(3) == False
    assert is_iterable_container('ciao') == False
    assert is_iterable_container([1, 2, 3]) == True

def test_set_tree_path():
    """Test getting and setting of CombinatorialTree paths."""
    test = CombinatorialTree({'a': 2})
    test_nested = CombinatorialTree({'a': {'b': 2}})
    test['a'] = 3
    assert test == {'a': 3}
    test_nested[('a','b')] = 3
    assert test_nested == {'a': {'b': 3}}
    test_nested[('a',)] = 5
    assert test_nested == {'a': 5}

def test_find_leaves():
    """Test CombinatorialTree private function _find_leaves()."""
    simple_tree = CombinatorialTree({'simple': {'scalar': 1,
                                                'vector': [2, 3, 4],
                                                'nested': {
                                                    'leaf': ['a', 'b', 'c']}}})
    leaf_paths, leaf_vals = simple_tree._find_leaves()
    assert leaf_paths == [('simple', 'scalar'), ('simple', 'vector'),
                          ('simple', 'nested', 'leaf')]
    assert leaf_vals == [1, [2, 3, 4], ['a', 'b', 'c']]

def test_expand_tree():
    """Test CombinatorialTree expansion."""
    simple_tree = CombinatorialTree({'simple': {'scalar': 1,
                                                'vector': [2, 3, 4],
                                                'nested': {
                                                    'leaf': ['a', 'b', 'c']}}})
    result = [{'simple': {'scalar': 1, 'vector': 2, 'nested': {'leaf': 'a'}}},
              {'simple': {'scalar': 1, 'vector': 2, 'nested': {'leaf': 'b'}}},
              {'simple': {'scalar': 1, 'vector': 2, 'nested': {'leaf': 'c'}}},
              {'simple': {'scalar': 1, 'vector': 3, 'nested': {'leaf': 'a'}}},
              {'simple': {'scalar': 1, 'vector': 3, 'nested': {'leaf': 'b'}}},
              {'simple': {'scalar': 1, 'vector': 3, 'nested': {'leaf': 'c'}}},
              {'simple': {'scalar': 1, 'vector': 4, 'nested': {'leaf': 'a'}}},
              {'simple': {'scalar': 1, 'vector': 4, 'nested': {'leaf': 'b'}}},
              {'simple': {'scalar': 1, 'vector': 4, 'nested': {'leaf': 'c'}}}]
    assert result == [exp for exp in simple_tree]

def test_validate_parameters():
    """Test validate_parameters function."""

    template_pars = {
        'bool': True,
        'int': 2,
        'float': 1e4,
        'str': 'default',
        'length': 2.0 * unit.nanometers,
        'time': 2.0 * unit.femtoseconds
    }
    input_pars = {
        'bool': False,
        'int': 4,
        'float': 3.0,
        'str': 'input',
        'length': 1.0 * unit.nanometers,
        'time': 1.0 * unit.femtoseconds
    }

    # Accept correct parameters
    assert input_pars == validate_parameters(input_pars, template_pars)

    # Convert float, length and time
    convert_pars = {
        'int': 3.0,
        'length': '1.0*nanometers',
        'time': '1.0*femtoseconds'
    }
    convert_pars = validate_parameters(convert_pars, template_pars,
                                       process_units_str=True, float_to_int=True)
    assert isinstance(convert_pars['int'], int)
    assert convert_pars['length'] == 1.0 * unit.nanometers
    assert convert_pars['time'] == 1.0 * unit.femtoseconds

@tools.raises(ValueError)
def test_incompatible_parameters():
    """Check that validate_parameters raises exception with unknown parameter."""
    template_pars = {'int': 3}
    wrong_pars = {'int': 3.0}
    validate_parameters(wrong_pars, template_pars)

@tools.raises(TypeError)
def test_unknown_parameters():
    """Test that validate_parameters() raises exception with unknown parameter."""
    template_pars = {'known_par': 3}
    wrong_pars = {'unknown_par': 3}
    validate_parameters(wrong_pars, template_pars, check_unknown=True)



def test_yank_options():
    """Test option priorities and handling."""

    cl_opt = {'option1': 1}
    yaml_opt = {'option1': 2, 'option2': 'test'}
    default_yank_opt = YankOptions()
    yank_opt = YankOptions(cl_opt=cl_opt, yaml_opt=yaml_opt)

    assert yank_opt['option2'] == 'test'
    assert yank_opt['option1'] == 1  # command line > yaml
    assert len(yank_opt) == len(default_yank_opt) + 2, "Excepted two additional options beyond default, found: %s" % str([x for x in yank_opt])

    # runtime > command line
    yank_opt['option1'] = 0
    assert yank_opt['option1'] == 0

    # restore old option when deleted at runtime
    del yank_opt['option1']
    assert yank_opt['option1'] == 1

    # modify specific priority level
    yank_opt.default = {'option3': -2}
    assert len(yank_opt) == 3
    assert yank_opt['option3'] == -2

    # test iteration interface
    assert yank_opt.items() == [('option1', 1), ('option2', 'test'), ('option3', -2)]
    assert yank_opt.keys() == ['option1', 'option2', 'option3']

