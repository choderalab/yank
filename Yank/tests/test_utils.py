#!/usr/local/bin/env python

"""
Test various utility functions.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import abc
import textwrap

import openmoltools as omt
from simtk import openmm, unit

from nose import tools
from yank.utils import * # TODO: Don't use 'import *'
from yank.schema.validator import * # TODO: Don't use 'import *'


# =============================================================================================
# COMBINATORIAL TREE
# =============================================================================================

def test_set_tree_path():
    """Test getting and setting of CombinatorialTree paths."""
    test = CombinatorialTree({'a': 2})
    test_nested = CombinatorialTree({'a': {'b': 2}})
    test['a'] = 3
    assert test == {'a': 3}
    test_nested[('a', 'b')] = 3
    assert test_nested == {'a': {'b': 3}}
    test_nested[('a',)] = 5
    assert test_nested == {'a': 5}


def test_find_leaves():
    """Test CombinatorialTree._find_leaves()."""
    simple_tree = CombinatorialTree({'simple': {'scalar': 1,
                                                'vector': [2, 3, 4],
                                                'nested': {
                                                    'leaf': ['a', 'b', 'c']}}})
    leaf_paths, leaf_vals = simple_tree._find_leaves()
    print(leaf_paths)
    assert all(leaf_path in [('simple', 'scalar'), ('simple', 'vector'),
                             ('simple', 'nested', 'leaf')]
                         for leaf_path in leaf_paths)
    assert all(leaf_val in  [1, [2, 3, 4], ['a', 'b', 'c']]
                        for leaf_val in leaf_vals)


def test_find_combinatorial_leaves():
    """Test CombinatorialTree._find_combinatorial_leaves()."""
    simple_tree = CombinatorialTree({'simple': {
        'scalar': 1,
        'vector': CombinatorialLeaf([2, 3, 4]),
        'nested': {
            'leaf': ['a', 'b', 'c'],
            'comb-leaf': CombinatorialLeaf(['d', 'e'])}}})
    leaf_paths, leaf_vals = simple_tree._find_combinatorial_leaves()

    # Paths must be in alphabetical order with their associated values
    assert leaf_paths == (('simple', 'nested', 'comb-leaf'), ('simple', 'vector'))
    assert leaf_vals == (['d', 'e'], [2, 3, 4])


def test_expand_tree():
    """Test CombinatorialTree generators."""
    simple_tree = CombinatorialTree({'simple': {'scalar': 1,
                                                'vector': CombinatorialLeaf([2, 3, 4]),
                                                'nested': {
                                                    'leaf': ['d', 'e'],
                                                    'combleaf': CombinatorialLeaf(['a', 'b', 'c'])}}})
    result = [{'simple': {'scalar': 1, 'vector': 2, 'nested': {'leaf': ['d', 'e'], 'combleaf': 'a'}}},
              {'simple': {'scalar': 1, 'vector': 3, 'nested': {'leaf': ['d', 'e'], 'combleaf': 'a'}}},
              {'simple': {'scalar': 1, 'vector': 4, 'nested': {'leaf': ['d', 'e'], 'combleaf': 'a'}}},
              {'simple': {'scalar': 1, 'vector': 2, 'nested': {'leaf': ['d', 'e'], 'combleaf': 'b'}}},
              {'simple': {'scalar': 1, 'vector': 3, 'nested': {'leaf': ['d', 'e'], 'combleaf': 'b'}}},
              {'simple': {'scalar': 1, 'vector': 4, 'nested': {'leaf': ['d', 'e'], 'combleaf': 'b'}}},
              {'simple': {'scalar': 1, 'vector': 2, 'nested': {'leaf': ['d', 'e'], 'combleaf': 'c'}}},
              {'simple': {'scalar': 1, 'vector': 3, 'nested': {'leaf': ['d', 'e'], 'combleaf': 'c'}}},
              {'simple': {'scalar': 1, 'vector': 4, 'nested': {'leaf': ['d', 'e'], 'combleaf': 'c'}}}]
    assert result == [exp for exp in simple_tree]

    # Test named_combinations generator using either order to account for generator randomness
    expected_names = {'a_2', 'a_3', 'a_4', 'b_2', 'b_3', 'b_4', 'c_2', 'c_3', 'c_4'}
    assert expected_names == set([name for name, _ in simple_tree.named_combinations(separator='_',
                                                                                     max_name_length=3)])

    # Test maximum length, similar names and special characters
    long_tree = CombinatorialTree({'key1': CombinatorialLeaf(['th#*&^isnameistoolong1',
                                                              'th#*&^isnameistoolong2']),
                                   'key2': CombinatorialLeaf(['test1', 'test2'])})
    expected_names = {'thisn-test', 'thisn-test-2', 'thisn-test-3', 'thisn-test-4'}
    assert expected_names == set([name for name, _ in long_tree.named_combinations(separator='-',
                                                                                   max_name_length=10)])

    # Test file paths are handled correctly
    data_dir = get_data_filename(os.path.join('tests', 'data'))
    abl = os.path.join(data_dir, 'abl-imatinib-explicit', '2HYY-pdbfixer.pdb')
    benzene = os.path.join(data_dir, 'benzene-toluene-explicit', 'benzene.tripos.mol2')
    long_tree = CombinatorialTree({'key1': CombinatorialLeaf([abl, benzene]),
                                   'key2': CombinatorialLeaf([benzene, benzene, 'notapath'])})
    expected_names = {'2HYYpdbfixer-benzene', '2HYYpdbfixer-benzene-2', '2HYYpdbfixer-notapath',
                      'benzene-benzene', 'benzene-benzene-2', 'benzene-notapath'}
    assert expected_names == set([name for name, _ in long_tree.named_combinations(separator='-',
                                                                                   max_name_length=25)])


def test_expand_id_nodes():
    """CombinatorialTree.expand_id_nodes()"""
    d = {'molecules':
             {'mol1': {'mol_value': CombinatorialLeaf([1, 2])},
              'mol2': {'mol_value': CombinatorialLeaf([3, 4])}},
         'systems':
             {'sys1': {'molecules': 'mol1'},
              'sys2': {'molecules': CombinatorialLeaf(['mol1', 'mol2'])},
              'sys3': {'prmtopfile': 'mysystem.prmtop'}}}
    t = CombinatorialTree(d).expand_id_nodes('molecules', [('systems', '*', 'molecules')])
    assert t['molecules'] == {'mol1_1': {'mol_value': 1}, 'mol1_2': {'mol_value': 2},
                              'mol2_3': {'mol_value': 3}, 'mol2_4': {'mol_value': 4}}
    assert t['systems'] == {'sys1': {'molecules': CombinatorialLeaf(['mol1_1', 'mol1_2'])},
                            'sys2': {'molecules': CombinatorialLeaf(['mol1_1', 'mol1_2', 'mol2_3', 'mol2_4'])},
                            'sys3': {'prmtopfile': 'mysystem.prmtop'}}


# ==============================================================================
# CONVERSION UTILITIES
# ==============================================================================

def test_get_keyword_args():
    """Test get_keyword_args() function."""
    def f(a, b, c=True, d=3.0):
        pass

    class dummy(object):
        def __init__(self, an_arg, a_kw_arg=True):
            pass

    class subdummy(dummy):
        def __init__(self, *args, my_own_kw=False, **kwargs):
            super().__init__(*args, **kwargs)

    expected_fn = {'c': True, 'd': 3.0}
    expected_cls = {"a_kw_arg": True}
    expected_subcls = {"my_own_kw": False}
    expected_true_cls = {"my_own_kw": False, "a_kw_arg": True}
    # Ensure the expected outcome is achieved, both with and without the mro class to search
    assert expected_fn == get_keyword_args(f)
    assert expected_fn == get_keyword_args(f, try_mro_from_class=dummy)
    assert expected_cls == get_keyword_args(dummy.__init__)
    assert expected_cls == get_keyword_args(dummy.__init__, try_mro_from_class=dummy)
    assert expected_subcls == get_keyword_args(subdummy.__init__)
    assert expected_true_cls == get_keyword_args(subdummy.__init__, try_mro_from_class=subdummy)


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
    validate_parameters({'unknown': 0}, template_pars)

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


def test_underscore_to_camelcase():
    """Test underscore_to_camelCase() conversion function."""
    cases = ['', '__', 'foo', 'foo_bar', '_foo_bar_', '__foo_bar__', '__foo__bar_']
    expected = ['', '__', 'foo', 'fooBar', '_fooBar_', '__fooBar__', '__fooBar_']
    for exp, case in zip(expected, cases):
        assert exp == underscore_to_camelcase(case)


def test_quantity_from_string():
    """Test the quantity from string function to ensure output is as expected."""
    test_cases = [
        # (string,                                 expected Unit)
        ('3',                                      3.0),  # Handle basic float
        ('meter',                                  unit.meter),  # Handle basic unit object
        ('300 * kelvin',                           300*unit.kelvin),  # Handle standard Quantity
        ('" 0.3 * kilojoules_per_mole / watt**3"', 0.3*unit.kilojoules_per_mole/unit.watt**3),  # Handle division, exponent, nested string
        ('1*meter / (4*second)',                   0.25*unit.meter/unit.second),  # Handle compound math and parenthesis
        ('1 * watt**2 /((1* kelvin)**3 / gram)',   1*(unit.watt**2)*(unit.gram)/(unit.kelvin**3))  #Handle everything
    ]

    for expression, expected in test_cases:
        assert expected == quantity_from_string(expression)

        # Test incompatible units.
        with tools.assert_raises(TypeError):
            quantity_from_string(expression, compatible_units=unit.second)

        # Test compatibile units.
        try:
            expected_unit = expected.unit.in_unit_system(unit.md_unit_system)
        except AttributeError:
            continue
        assert expected == quantity_from_string(expression, compatible_units=expected_unit)


def test_TLeap_script():
    """Test TLeap script creation."""
    expected_script = """
    source oldff/leaprc.ff99SBildn
    source leaprc.gaff
    M5receptor = loadPdb receptor.pdbfixer.pdb
    loadAmberParams ligand.gaff.frcmod
    ligand = loadMol2 path/to/ligand.gaff.mol2
    transform ligand {{ 1 0 0 6} { 0 -1 0 0} { 0 0 1 0} { 0 0 0 1}}
    complex = combine { M5receptor ligand }
    solvateBox complex TIP3PBOX 10.0 iso
    check complex
    charge complex

    # New section
    saveAmberParm complex complex.prmtop complex.inpcrd
    savePDB complex complex.pdb
    solvateBox ligand TIP3PBOX 10.0 iso
    saveAmberParm ligand solvent.prmtop solvent.inpcrd
    savePDB ligand solvent.pdb

    quit
    """
    expected_script = textwrap.dedent(expected_script[1:])  # delete first \n char

    tleap = TLeap()
    tleap.load_parameters('oldff/leaprc.ff99SBildn', 'leaprc.gaff')
    # TLeap should prepend a character to names starting with a digit.
    tleap.load_unit(unit_name='5receptor', file_path='receptor.pdbfixer.pdb')
    tleap.load_parameters('ligand.gaff.frcmod')
    tleap.load_parameters('ligand.gaff.frcmod')  # tLeap should not load this twice
    tleap.load_unit(unit_name='ligand', file_path='path/to/ligand.gaff.mol2')
    tleap.transform('ligand', np.array([[1, 0, 0, 6], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    tleap.combine('complex', '5receptor', 'ligand')
    tleap.solvate(unit_name='complex', solvent_model='TIP3PBOX', clearance=10.0*unit.angstrom)
    tleap.add_commands('check complex', 'charge complex')
    tleap.new_section('New section')
    tleap.save_unit(unit_name='complex', output_path='complex.prmtop')
    tleap.save_unit(unit_name='complex', output_path='complex.pdb')
    tleap.solvate(unit_name='ligand', solvent_model='TIP3PBOX', clearance=10.0*unit.angstrom)
    tleap.save_unit(unit_name='ligand', output_path='solvent.inpcrd')
    tleap.save_unit(unit_name='ligand', output_path='solvent.pdb')

    assert tleap.script == expected_script


def test_TLeap_export_run():
    """Check that TLeap saves and runs scripts correctly."""
    setup_dir = get_data_filename(os.path.join('tests', 'data',
                                               'benzene-toluene-explicit'))
    benzene_gaff = os.path.join(setup_dir, 'benzene.gaff.mol2')
    benzene_frcmod = os.path.join(setup_dir, 'benzene.frcmod')

    # Leap has problems with unit names that contain too many numbers.
    # Test if we are able to run even with something like this.
    unit_name = '534E56'
    tleap = TLeap()
    tleap.load_parameters('oldff/leaprc.ff99SB', 'leaprc.gaff')
    tleap.load_unit(unit_name=unit_name, file_path=benzene_gaff)
    tleap.load_parameters(benzene_frcmod)

    with omt.utils.temporary_directory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'benzene')
        tleap.save_unit(unit_name=unit_name, output_path=output_path + '.prmtop')

        export_path = os.path.join(tmp_dir, 'leap.in')
        tleap.export_script(export_path)
        assert os.path.isfile(export_path)
        assert os.path.getsize(export_path) > 0

        tleap.run()
        assert os.path.isfile(output_path + '.prmtop')
        assert os.path.isfile(output_path + '.inpcrd')
        assert os.path.getsize(output_path + '.prmtop') > 0
        assert os.path.getsize(output_path + '.inpcrd') > 0
        assert os.path.isfile(os.path.join(tmp_dir, 'benzene.leap.log'))
