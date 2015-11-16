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

import tempfile
import textwrap
import unittest

from nose.tools import raises

from yank.yamlbuild import *

#=============================================================================================
# SUBROUTINES FOR TESTING
#=============================================================================================

def example_dir():
    """Return the absolute path to the Yank examples directory."""
    return utils.get_data_filename(os.path.join('..', 'examples'))

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

def test_compute_min_dist():
    """Test computation of minimum distance between two molecules"""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol2_pos = np.array([[3, 3, 3], [3, 4, 5]], np.float)
    mol3_pos = np.array([[2, 2, 2], [2, 4, 5]], np.float)
    assert compute_min_dist(mol1_pos, mol2_pos, mol3_pos) == np.sqrt(3)

def test_remove_overlap():
    """Test function remove_overlap()."""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol2_pos = np.array([[1, 1, 1], [3, 4, 5]], np.float)
    mol3_pos = np.array([[2, 2, 2], [2, 4, 5]], np.float)
    assert compute_min_dist(mol1_pos, mol2_pos, mol3_pos) < 0.1
    mol1_pos = remove_overlap(mol1_pos, mol2_pos, mol3_pos, min_distance=0.1, sigma=2.0)
    assert compute_min_dist(mol1_pos, mol2_pos, mol3_pos) >= 0.1

def test_yaml_parsing():
    """Check that YAML file is parsed correctly."""

    # Parser handles no options
    yaml_content = """
    ---
    test: 2
    """
    yaml_builder = parse_yaml_str(yaml_content)
    assert len(yaml_builder.yank_options) == 0

    # Correct parsing
    yaml_content = """
    ---
    metadata:
        title: Test YANK YAML YAY!

    options:
        verbose: true
        mpi: yes
        resume: true
        output_dir: /path/to/output/
        temperature: 300*kelvin
        pressure: 1*atmosphere
        constraints: AllBonds
        hydrogenMass: 2*amus
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
    assert len(yaml_builder.yank_options) == 21

    # Check correct types
    assert yaml_builder._constraints == openmm.app.AllBonds
    assert yaml_builder.yank_options['replica_mixing_scheme'] == 'swap-all'
    assert yaml_builder.yank_options['timestep'] == 2.0 * unit.femtoseconds
    assert yaml_builder.yank_options['constraint_tolerance'] == 1.0e-6
    assert yaml_builder.yank_options['nsteps_per_iteration'] == 2500
    assert type(yaml_builder.yank_options['nsteps_per_iteration']) is int
    assert yaml_builder.yank_options['number_of_iterations'] == 1000
    assert type(yaml_builder.yank_options['number_of_iterations']) is int
    assert yaml_builder.yank_options['minimize'] is False
    assert yaml_builder.yank_options['show_mixing_statistics'] is True

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
    benzene_path = os.path.join(example_dir(), 'benzene-toluene-explicit',
                                'setup', 'benzene.tripos.mol2')
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            benzene:
                filepath: {}
                parameters: antechamber
        """.format(tmp_dir, benzene_path)

        yaml_builder = parse_yaml_str(yaml_content)
        yaml_builder._setup_molecules('benzene')

        output_dir = os.path.join(tmp_dir, YamlBuilder.SETUP_MOLECULES_DIR, 'benzene')
        assert os.path.exists(os.path.join(output_dir, 'benzene.gaff.mol2'))
        assert os.path.exists(os.path.join(output_dir, 'benzene.frcmod'))
        assert os.path.getsize(os.path.join(output_dir, 'benzene.gaff.mol2')) > 0
        assert os.path.getsize(os.path.join(output_dir, 'benzene.frcmod')) > 0

@unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
def test_setup_name_smiles_antechamber():
    """Setup molecule from name and SMILES with antechamber parametrization."""
    benzene_path = os.path.join(example_dir(), 'benzene-toluene-explicit',
                                'setup', 'benzene.tripos.mol2')
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            p-xylene:
                name: p-xylene
                parameters: antechamber
            benzene:
                filepath: {}
                parameters: antechamber
            toluene:
                smiles: Cc1ccccc1
                parameters: antechamber
        """.format(tmp_dir, benzene_path)

        yaml_builder = parse_yaml_str(yaml_content)

        # The order of the arguments in _setup_molecules is important, we want to test
        # that overlapping molecules in toluene are removed even if benzene has been
        # indicated to be processed afterwards
        yaml_builder._setup_molecules('toluene', 'benzene', 'p-xylene')

        for mol in ['toluene', 'p-xylene']:
            output_dir = os.path.join(tmp_dir, YamlBuilder.SETUP_MOLECULES_DIR, mol)
            assert os.path.exists(os.path.join(output_dir, mol + '.mol2'))
            assert os.path.exists(os.path.join(output_dir, mol + '.gaff.mol2'))
            assert os.path.exists(os.path.join(output_dir, mol + '.frcmod'))
            assert os.path.getsize(os.path.join(output_dir, mol + '.mol2')) > 0
            assert os.path.getsize(os.path.join(output_dir, mol + '.gaff.mol2')) > 0
            assert os.path.getsize(os.path.join(output_dir, mol + '.frcmod')) > 0

        # Test that molecules do not overlap
        toluene_path = os.path.join(tmp_dir, YamlBuilder.SETUP_MOLECULES_DIR,
                                    'toluene', 'toluene.gaff.mol2')
        toluene_pos = utils.get_oe_mol_positions(utils.read_oe_molecule(toluene_path))
        benzene_pos = utils.get_oe_mol_positions(utils.read_oe_molecule(benzene_path))
        assert compute_min_dist(toluene_pos, benzene_pos) >= 1.0

@raises(YamlParseError)
@unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
def test_overlapping_atoms():
    """Check that exception is raised when overlapping atoms."""
    setup_dir = os.path.join(example_dir(), 'benzene-toluene-explicit', 'setup')
    benzene_path = os.path.join(setup_dir, 'benzene.tripos.mol2')
    toluene_path = os.path.join(setup_dir, 'toluene.tripos.mol2')
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            benzene:
                filepath: {}
                parameters: antechamber
            toluene:
                filepath: {}
                parameters: antechamber
        """.format(tmp_dir, benzene_path, toluene_path)

        yaml_builder = parse_yaml_str(yaml_content)
        yaml_builder._setup_molecules('benzene', 'toluene')

def test_setup_implicit_system_leap():
    """Create prmtop and inpcrd for implicit solvent protein-ligand system."""
    setup_dir = os.path.join(example_dir(), 'p-xylene-implicit', 'setup')
    receptor_path = os.path.join(setup_dir, 'receptor.pdbfixer.pdb')
    ligand_path = os.path.join(setup_dir, 'ligand.tripos.mol2')
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            T4lysozyme:
                filepath: {}
                parameters: oldff/leaprc.ff99SBildn
            p-xylene:
                filepath: {}
                parameters: antechamber
        solvents:
            GBSA-OBC2:
                nonbondedMethod: NoCutoff
                implicitSolvent: OBC2
        """.format(tmp_dir, receptor_path, ligand_path)

        yaml_builder = parse_yaml_str(yaml_content)
        components = {'receptor': 'T4lysozyme',
                      'ligand': 'p-xylene',
                      'solvent': 'GBSA-OBC2'}

        output_dir = yaml_builder._setup_system(components)

        # Test that output files exist and there is no water
        for phase in ['complex', 'solvent']:
            found_resnames = set()
            pdb_path = os.path.join(output_dir, phase + '.pdb')
            prmtop_path = os.path.join(output_dir, phase + '.prmtop')
            inpcrd_path = os.path.join(output_dir, phase + '.inpcrd')

            with open(pdb_path, 'r') as f:
                for line in f:
                    if len(line) > 10:
                        found_resnames.add(line[17:20])

            assert os.path.exists(prmtop_path)
            assert os.path.exists(inpcrd_path)
            assert os.path.getsize(prmtop_path) > 0
            assert os.path.getsize(inpcrd_path) > 0
            assert 'MOL' in found_resnames
            assert 'WAT' not in found_resnames

@unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
def test_setup_explicit_system_leap():
    """Create prmtop and inpcrd protein-ligand system in explicit solvent."""
    benzene_path = os.path.join(example_dir(), 'benzene-toluene-explicit',
                                'setup', 'benzene.tripos.mol2')
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            benzene:
                filepath: {}
                parameters: antechamber
            toluene:
                name: toluene
                parameters: antechamber
        solvents:
            PMEtip3p:
                nonbondedMethod: PME
                clearance: 10*angstroms
        """.format(tmp_dir, benzene_path)

        yaml_builder = parse_yaml_str(yaml_content)
        components = {'receptor': 'benzene',
                      'ligand': 'toluene',
                      'solvent': 'PMEtip3p'}

        output_dir = yaml_builder._setup_system(components)

        # Test that output file exists and that there is water
        expected_resnames = {'complex': set(['BEN', 'TOL', 'WAT']),
                             'solvent': set(['TOL', 'WAT'])}
        for phase in expected_resnames:
            found_resnames = set()
            pdb_path = os.path.join(output_dir, phase + '.pdb')
            prmtop_path = os.path.join(output_dir, phase + '.prmtop')
            inpcrd_path = os.path.join(output_dir, phase + '.inpcrd')

            with open(pdb_path, 'r') as f:
                for line in f:
                    if len(line) > 10:
                        found_resnames.add(line[17:20])

            assert os.path.exists(prmtop_path)
            assert os.path.exists(inpcrd_path)
            assert os.path.getsize(prmtop_path) > 0
            assert os.path.getsize(inpcrd_path) > 0
            assert found_resnames == expected_resnames[phase]

def test_run_experiment():
    setup_dir = os.path.join(example_dir(), 'p-xylene-implicit', 'setup')
    receptor_path = os.path.join(setup_dir, 'receptor.pdbfixer.pdb')
    ligand_path = os.path.join(setup_dir, 'ligand.tripos.mol2')
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            number_of_iterations: 1
            output_dir: {}
        molecules:
            T4lysozyme:
                filepath: {}
                parameters: oldff/leaprc.ff99SBildn
            p-xylene:
                filepath: {}
                parameters: antechamber
        solvents:
            vacuum:
                nonbondedMethod: NoCutoff
            GBSA-OBC2:
                nonbondedMethod: NoCutoff
                implicitSolvent: OBC2
        experiment:
            components:
                receptor: T4lysozyme
                ligand: p-xylene
                solvent: [vacuum, GBSA-OBC2]
        """.format(tmp_dir, receptor_path, ligand_path)

        yaml_builder = parse_yaml_str(yaml_content)
        yaml_builder.build_experiment()

        for exp_name in ['vacuum', 'GBSAOBC2']:
            output_dir = os.path.join(tmp_dir, yaml_builder.EXPERIMENTS_DIR, exp_name)
            assert os.path.isdir(output_dir)
            assert os.path.isfile(os.path.join(output_dir, 'complex-implicit.nc'))
            assert os.path.isfile(os.path.join(output_dir, 'solvent-implicit.nc'))

# TODO handle resume molecule setup for combinatorial experiments
# TODO save YAML format for each experiment
# TODO validate syntax
# TODO start from prmtop and inpcrd files
# TODO start form gro and top files
# TODO epik

# TODO documentation validate_parameters, future openmoltools methods, YamlBuilder methods
# TODO ModifiedHamiltonianExchange use very similar algorithm to remove_overlap: refactor
# TODO refactor common code yamlbuilder and commands in pipeline

# TODO default solvents?
# TODO default protocol?
