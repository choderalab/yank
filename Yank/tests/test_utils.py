#!/usr/local/bin/env python

"""
Test various utility functions.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

from yank.utils import YankOptions, CombinatorialTree, is_iterable_container

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

