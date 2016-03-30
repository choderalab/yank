#!/usr/local/bin/env python

"""
Test various utility functions.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import textwrap

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
    """Test CombinatorialTree generators."""
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

    # Test named_combinations generator
    expected_names = set(['2_a', '3_a', '4_a', '2_b', '3_b', '4_b', '2_c', '3_c', '4_c'])
    assert expected_names == set([name for name, _ in simple_tree.named_combinations(
                                                       separator='_', max_name_length=3)])

    # Test maximum length, similar names and special characters
    long_tree = CombinatorialTree({'key1': ['th#*&^isnameistoolong1', 'th#*&^isnameistoolong2'],
                                   'key2': ['test1', 'test2']})
    expected_names = set(['test-thisn', 'test-thisn-2', 'test-thisn-3', 'test-thisn-4'])
    assert expected_names == set([name for name, _ in long_tree.named_combinations(
                                                       separator='-', max_name_length=10)])

def test_get_keyword_args():
    """Test get_keyword_args() function."""
    def f(a, b, c=True, d=3.0):
        pass
    expected = {'c': True, 'd': 3.0}
    assert expected == get_keyword_args(f)

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
        'bool': True,
        'int': 3.0,
        'length': '1.0*nanometers',
        'time': '1.0*femtoseconds'
    }
    convert_pars = validate_parameters(convert_pars, template_pars,
                                       process_units_str=True, float_to_int=True)
    assert type(convert_pars['bool']) is bool
    assert type(convert_pars['int']) is int
    assert convert_pars['length'] == 1.0 * unit.nanometers
    assert convert_pars['time'] == 1.0 * unit.femtoseconds

    # If check_unknown flag is not True it should not raise an error
    validate_parameters({'unkown': 0}, template_pars)

    # Test special conversion
    def convert_length(length):
        return str(length)
    special_conv = {'length': convert_length}
    convert_pars = {'length': '1.0*nanometers'}
    convert_pars = validate_parameters(convert_pars, template_pars, process_units_str=True,
                                       special_conversions=special_conv)
    assert convert_pars['length'] == '1.0*nanometers'

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

def test_temp_dir_context():
    """Test the context temporary_directory()."""
    with temporary_directory() as tmp_dir:
        assert os.path.isdir(tmp_dir)
    assert not os.path.exists(tmp_dir)

def test_temp_cd_context():
    """Test the context temporary_cd()."""
    with temporary_directory() as tmp_dir:
        with temporary_cd(tmp_dir):
            assert os.getcwd() == os.path.realpath(tmp_dir)
        assert os.getcwd() != os.path.realpath(tmp_dir)

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

def test_underscore_to_camelcase():
    """Test underscore_to_camelCase() conversion function."""
    cases = ['', '__', 'foo', 'foo_bar', '_foo_bar_', '__foo_bar__', '__foo__bar_']
    expected = ['', '__', 'foo', 'fooBar', '_fooBar_', '__fooBar__', '__fooBar_']
    for exp, case in zip(expected, cases):
        assert exp == underscore_to_camelcase(case)

def test_TLeap_script():
    """Test TLeap script creation."""
    expected_script = """
    source oldff/leaprc.ff99SBildn
    source leaprc.gaff
    receptor = loadPdb receptor.pdbfixer.pdb
    loadAmberParams ligand.gaff.frcmod
    ligand = loadMol2 path/to/ligand.gaff.mol2
    transform ligand {{ 1 0 0 6} { 0 -1 0 0} { 0 0 1 0} { 0 0 0 1}}
    complex = combine { receptor ligand }
    solvateBox complex TIP3PBOX 10.0 iso
    check complex
    charge complex

    # New section
    saveAmberParm complex complex.prmtop complex.inpcrd
    savePDB complex complex.pdb
    solvateBox ligand TIP3PBOX 10.0 iso
    saveAmberParm ligand solvent.inpcrd solvent.prmtop
    savePDB ligand solvent.pdb

    quit
    """
    expected_script = textwrap.dedent(expected_script[1:])  # delete first \n char

    tleap = TLeap()
    tleap.load_parameters('oldff/leaprc.ff99SBildn', 'leaprc.gaff')
    tleap.load_group(name='receptor', file_path='receptor.pdbfixer.pdb')
    tleap.load_parameters('ligand.gaff.frcmod')
    tleap.load_parameters('ligand.gaff.frcmod')  # tLeap should not load this twice
    tleap.load_group(name='ligand', file_path='path/to/ligand.gaff.mol2')
    tleap.transform('ligand', np.array([[1, 0, 0, 6], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    tleap.combine('complex', 'receptor', 'ligand')
    tleap.solvate(group='complex', water_model='TIP3PBOX', clearance=10.0)
    tleap.add_commands('check complex', 'charge complex')
    tleap.new_section('New section')
    tleap.save_group(group='complex', output_path='complex.prmtop')
    tleap.save_group(group='complex', output_path='complex.pdb')
    tleap.solvate(group='ligand', water_model='TIP3PBOX', clearance=10.0)
    tleap.save_group(group='ligand', output_path='solvent.inpcrd')
    tleap.save_group(group='ligand', output_path='solvent.pdb')

    assert tleap.script == expected_script

def test_TLeap_export_run():
    """Check that TLeap saves and runs scripts correctly."""
    setup_dir = get_data_filename(os.path.join('..', 'examples',
                                               'benzene-toluene-explicit', 'setup'))
    benzene_gaff = os.path.join(setup_dir, 'benzene.gaff.mol2')
    benzene_frcmod = os.path.join(setup_dir, 'benzene.frcmod')

    tleap = TLeap()
    tleap.load_parameters('oldff/leaprc.ff99SB', 'leaprc.gaff')
    tleap.load_group(name='benzene', file_path=benzene_gaff)
    tleap.load_parameters(benzene_frcmod)

    with temporary_directory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'benzene')
        tleap.save_group(group='benzene', output_path=output_path + '.prmtop')

        export_path = os.path.join(tmp_dir, 'leap.in')
        tleap.export_script(export_path)
        assert os.path.isfile(export_path)
        assert os.path.getsize(export_path) > 0

        tleap.run()
        assert os.path.isfile(output_path + '.prmtop')
        assert os.path.isfile(output_path + '.inpcrd')
        assert os.path.getsize(output_path + '.prmtop') > 0
        assert os.path.getsize(output_path + '.inpcrd') > 0
        assert os.path.isfile(os.path.join(tmp_dir, 'leap.log'))

