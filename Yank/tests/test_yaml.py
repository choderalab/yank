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
import yaml
import tempfile
import textwrap
import unittest

from simtk import unit
from nose.tools import raises

from yank.utils import temporary_directory
from yank.yamlbuild import YamlBuilder, YamlParseError

#=============================================================================================
# SUBROUTINES FOR TESTING
#=============================================================================================

# TODO move this to openmoltools
def is_openeye_installed():
    try:
        from openeye import oechem
        from openeye import oequacpac
        from openeye import oeiupac
        from openeye import oeomega

        if not (oechem.OEChemIsLicensed() and oequacpac.OEQuacPacIsLicensed()
                and oeiupac.OEIUPACIsLicensed() and oeomega.OEOmegaIsLicensed()):
            raise ImportError
    except ImportError:
        return False
    return True


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
    benzene_path = os.path.join(example_dir(), 'benzene-toluene-explicit',
                                'setup', 'benzene.tripos.mol2')
    with temporary_directory() as tmp_dir:
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
        yaml_builder._setup_molecule('benzene')

        output_dir = os.path.join(tmp_dir, YamlBuilder.SETUP_MOLECULES_DIR, 'benzene')
        assert os.path.exists(os.path.join(output_dir, 'benzene.gaff.mol2'))
        assert os.path.exists(os.path.join(output_dir, 'benzene.frcmod'))
        assert os.path.getsize(os.path.join(output_dir, 'benzene.gaff.mol2')) > 0
        assert os.path.getsize(os.path.join(output_dir, 'benzene.frcmod')) > 0

@unittest.skipIf(not is_openeye_installed(), 'This test requires OpenEye installed.')
def test_setup_name_antechamber():
    """Setup molecule from name with antechamber parametrization."""
    with temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            p-xylene:
                name: p-xylene
                parameters: antechamber
        """.format(tmp_dir)

        yaml_builder = parse_yaml_str(yaml_content)
        yaml_builder._setup_molecule('p-xylene')

        output_dir = os.path.join(tmp_dir, YamlBuilder.SETUP_MOLECULES_DIR, 'p-xylene')
        assert os.path.exists(os.path.join(output_dir, 'p-xylene.mol2'))
        assert os.path.exists(os.path.join(output_dir, 'p-xylene.gaff.mol2'))
        assert os.path.exists(os.path.join(output_dir, 'p-xylene.frcmod'))
        assert os.path.getsize(os.path.join(output_dir, 'p-xylene.mol2')) > 0
        assert os.path.getsize(os.path.join(output_dir, 'p-xylene.gaff.mol2')) > 0
        assert os.path.getsize(os.path.join(output_dir, 'p-xylene.frcmod')) > 0

@unittest.skipIf(not is_openeye_installed(), 'This test requires OpenEye installed.')
def test_setup_smiles_antechamber():
    """Setup molecule from SMILES with antechamber parametrization."""
    with temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            toluene:
                smiles: Cc1ccccc1
                parameters: antechamber
        """.format(tmp_dir)

        yaml_builder = parse_yaml_str(yaml_content)
        yaml_builder._setup_molecule('toluene')

        output_dir = os.path.join(tmp_dir, YamlBuilder.SETUP_MOLECULES_DIR, 'toluene')
        assert os.path.exists(os.path.join(output_dir, 'toluene.mol2'))
        assert os.path.exists(os.path.join(output_dir, 'toluene.gaff.mol2'))
        assert os.path.exists(os.path.join(output_dir, 'toluene.frcmod'))
        assert os.path.getsize(os.path.join(output_dir, 'toluene.mol2')) > 0
        assert os.path.getsize(os.path.join(output_dir, 'toluene.gaff.mol2')) > 0
        assert os.path.getsize(os.path.join(output_dir, 'toluene.frcmod')) > 0

def test_experiment_iteration():
    """Test iteration over combinatorial experiments."""
    exp_template = 'components: {{ligands: {}, receptors: {}, solvents: {}}}'
    yaml_content = """
    ---
    options:
        output_dir: output
    experiment:
        {}
    """.format(exp_template.format('[ligand1, ligand2]',
                                   '[receptor1, receptor2]',
                                   '[solvent1, solvent2]'))
    yaml_builder = parse_yaml_str(yaml_content)
    generated_exp = {yaml.dump(exp).strip() for exp in yaml_builder._expand_experiments()}
    assert len(generated_exp) == 8
    assert exp_template.format('ligand1', 'receptor1', 'solvent1') in generated_exp
    assert exp_template.format('ligand1', 'receptor1', 'solvent2') in generated_exp
    assert exp_template.format('ligand1', 'receptor2', 'solvent1') in generated_exp
    assert exp_template.format('ligand1', 'receptor2', 'solvent2') in generated_exp
    assert exp_template.format('ligand2', 'receptor1', 'solvent1') in generated_exp
    assert exp_template.format('ligand2', 'receptor1', 'solvent2') in generated_exp
    assert exp_template.format('ligand2', 'receptor2', 'solvent1') in generated_exp
    assert exp_template.format('ligand2', 'receptor2', 'solvent2') in generated_exp

def test_multiple_experiments_iteration():
    """Test iteration over sequence of combinatorial experiments."""
    exp_template = 'components: {{ligands: {}, receptors: {}, solvents: solvent}}'
    yaml_content = """
    ---
    options:
        output_dir: output
    exp1:
        {}
    exp-name2:
        {}
    experiments: [exp1, exp-name2]
    """.format(exp_template.format('[ligand1, ligand2]', 'receptor1'),
               exp_template.format('[ligand1, ligand2]', 'receptor2'))

    yaml_builder = parse_yaml_str(yaml_content)
    generated_exp = {yaml.dump(exp).strip() for exp in yaml_builder._expand_experiments()}
    for exp in generated_exp:
        print exp
    assert len(generated_exp) == 4
    assert exp_template.format('ligand1', 'receptor1') in generated_exp
    assert exp_template.format('ligand2', 'receptor1') in generated_exp
    assert exp_template.format('ligand1', 'receptor2') in generated_exp
    assert exp_template.format('ligand2', 'receptor2') in generated_exp
