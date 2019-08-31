#!/usr/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Test YAML functions.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import itertools
import shutil
import tempfile
import textwrap
import time
import unittest

import mdtraj
from nose.plugins.attrib import attr
from nose.tools import assert_raises, assert_equal, assert_raises_regexp

from yank.experiment import *

# silence the citations at a global level
mmtools.multistate.MultiStateSampler._global_citation_silence = True


# ==============================================================================
# Subroutines for testing
# ==============================================================================

standard_protocol = """
        absolute-binding:
            complex:
                alchemical_path:
                    lambda_electrostatics: [1.0, 0.5, 0.0]
                    lambda_sterics: [1.0, 0.5, 0.0]
            solvent:
                alchemical_path:
                    lambda_electrostatics: [1.0, 0.5, 0.0]
                    lambda_sterics: [1.0, 0.5, 0.0]"""


def indent(input_string):
    """Put 4 extra spaces in front of every line."""
    return '\n    '.join(input_string.split('\n'))


def examples_paths():
    """Return the absolute path to the Yank examples relevant to tests."""
    data_dir = utils.get_data_filename(os.path.join('tests', 'data'))
    p_xylene_dir = os.path.join(data_dir, 'p-xylene-implicit')
    p_xylene_gro_dir = os.path.join(data_dir, 'p-xylene-gromacs-example')
    ben_tol_dir = os.path.join(data_dir, 'benzene-toluene-explicit')
    abl_imatinib_dir = os.path.join(data_dir, 'abl-imatinib-explicit')
    tol_dir = os.path.join(data_dir, 'toluene-explicit')
    benz_tol_dir = os.path.join(data_dir, 'benzene-toluene-standard-state')

    paths = dict()
    paths['lysozyme'] = os.path.join(p_xylene_dir, '181L-pdbfixer.pdb')
    paths['p-xylene'] = os.path.join(p_xylene_dir, 'p-xylene.mol2')
    paths['benzene'] = os.path.join(ben_tol_dir, 'benzene.tripos.mol2')
    paths['toluene'] = os.path.join(ben_tol_dir, 'toluene.tripos.mol2')
    paths['abl'] = os.path.join(abl_imatinib_dir, '2HYY-pdbfixer.pdb')
    paths['imatinib'] = os.path.join(abl_imatinib_dir, 'STI02.mol2')
    paths['bentol-complex'] = [os.path.join(ben_tol_dir, 'complex.prmtop'),
                               os.path.join(ben_tol_dir, 'complex.inpcrd')]
    paths['bentol-solvent'] = [os.path.join(ben_tol_dir, 'solvent.prmtop'),
                               os.path.join(ben_tol_dir, 'solvent.inpcrd')]
    paths['pxylene-complex'] = [os.path.join(p_xylene_gro_dir, 'complex.top'),
                                os.path.join(p_xylene_gro_dir, 'complex.gro')]
    paths['pxylene-solvent'] = [os.path.join(p_xylene_gro_dir, 'solvent.top'),
                                os.path.join(p_xylene_gro_dir, 'solvent.gro')]
    paths['pxylene-gro-include'] = os.path.join(p_xylene_gro_dir, 'top')
    paths['toluene-solvent'] = [os.path.join(tol_dir, 'solvent.pdb'),
                                os.path.join(tol_dir, 'solvent.xml')]
    paths['toluene-vacuum'] = [os.path.join(tol_dir, 'vacuum.pdb'),
                               os.path.join(tol_dir, 'vacuum.xml')]
    paths['benzene-toluene-boxless'] = [os.path.join(benz_tol_dir, 'standard_state_complex_boxless.inpcrd'),
                                        os.path.join(benz_tol_dir, 'standard_state_complex.prmtop')]
    paths['benzene-toluene-nan'] = [os.path.join(benz_tol_dir, 'standard_state_complex_nan.inpcrd'),
                                    os.path.join(benz_tol_dir, 'standard_state_complex.prmtop')]
    return paths


def yank_load(script):
    """Shortcut to load a string YAML script with YankLoader."""
    return yaml.load(textwrap.dedent(script), Loader=YankLoader)


def get_template_script(output_dir='.', keep_schrodinger=False, keep_openeye=False,
                        systems='all'):
    """Return a YAML template script as a dict.

    Parameters
    ----------
    output_dir : str, optional
        The YANK output directory to set in the YAML options.
    keep_schrodinger : bool, optional
        If False, removes the molecules that depend on the Schrodinger
        toolkit. Default is False.
    keep_openeye : bool, optional
        If False, removes the molecules that depend on the OpenEye
        toolkit. Default is False.
    systems : List[str], optional
        Limits the systems in the YAML to those identified by the given
        IDs. If 'all', all systems are included in the script, which means
        that the setup pipeline will build them all.
    """
    paths = examples_paths()
    template_script = """
    ---
    options:
        output_dir: {output_dir}
        default_number_of_iterations: 0
        temperature: 300*kelvin
        pressure: 1*atmosphere
        minimize: no
        verbose: no
        default_nsteps_per_iteration: 1
    molecules:
        benzene:
            filepath: {benzene_path}
            antechamber: {{charge_method: bcc}}
        benzene-epik0:
            filepath: {benzene_path}
            epik:
                select: 0
            antechamber: {{charge_method: bcc}}
        benzene-epikcustom:
            filepath: {benzene_path}
            epik:
                select: 0
                ph: 7.0
                tautomerize: yes
            antechamber: {{charge_method: bcc}}
        p-xylene:
            filepath: {pxylene_path}
            antechamber: {{charge_method: bcc}}
        p-xylene-name:
            name: p-xylene
            openeye: {{quacpac: am1-bcc}}
            antechamber: {{charge_method: null}}
        toluene:
            filepath: {toluene_path}
            antechamber: {{charge_method: bcc}}
        toluene-smiles:
            smiles: Cc1ccccc1
            antechamber: {{charge_method: bcc}}
        toluene-name:
            name: toluene
            antechamber: {{charge_method: bcc}}
        Abl:
            filepath: {abl_path}
        T4Lysozyme:
            filepath: {lysozyme_path}
    solvents:
        vacuum:
            nonbonded_method: NoCutoff
        GBSA-OBC2:
            nonbonded_method: NoCutoff
            implicit_solvent: OBC2
        PME:
            nonbonded_method: PME
            nonbonded_cutoff: 1*nanometer
            clearance: 10*angstroms
            positive_ion: Na+
            negative_ion: Cl-
            leap:
                parameters: [leaprc.water.tip4pew]
    systems:
        explicit-system:
            receptor: benzene
            ligand: toluene
            solvent: PME
            leap:
                parameters: [leaprc.protein.ff14SB, leaprc.gaff]
        implicit-system:
            receptor: T4Lysozyme
            ligand: p-xylene
            solvent: GBSA-OBC2
            leap:
                parameters: [leaprc.protein.ff14SB, leaprc.gaff]
        hydration-system:
            solute: toluene
            solvent1: PME
            solvent2: vacuum
            leap:
                parameters: [leaprc.protein.ff14SB, leaprc.gaff]
    mcmc_moves:
        single:
            type: LangevinSplittingDynamicsMove
        sequence:
            type: SequenceMove
            move_list:
                - type: MCDisplacementMove
                - type: LangevinDynamicsMove
    samplers:
        repex:
            type: ReplicaExchangeSampler
        sams:
            type: SAMSSampler
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
        hydration-protocol:
            solvent1:
                alchemical_path:
                    lambda_electrostatics: [1.0, 0.0]
                    lambda_sterics: [1.0, 0.0]
            solvent2:
                alchemical_path:
                    lambda_electrostatics: [1.0, 0.0]
                    lambda_sterics: [1.0, 1.0]
    experiments:
        system: explicit-system
        protocol: absolute-binding
    """.format(output_dir=output_dir, benzene_path=paths['benzene'],
               pxylene_path=paths['p-xylene'], toluene_path=paths['toluene'],
               abl_path=paths['abl'], lysozyme_path=paths['lysozyme'])

    # Load script as dictionary.
    script_dict = yank_load(template_script)

    # Find all molecules that require optional tools.
    molecules_to_remove = []
    for molecule_id, molecule_description in script_dict['molecules'].items():
        need_schrodinger = 'epik' in molecule_description
        need_openeye = any([k in molecule_description for k in ['name', 'smiles', 'openeye']])
        if ((need_schrodinger and not keep_schrodinger) or
                (need_openeye and not keep_openeye)):
            molecules_to_remove.append(molecule_id)

    # Remove molecules.
    for molecule_id in molecules_to_remove:
        del script_dict['molecules'][molecule_id]

    # Remove systems.
    if systems != 'all':
        systems_to_remove = [s for s in script_dict['systems'] if s not in systems]
        for system_id in systems_to_remove:
            del script_dict['systems'][system_id]

    return script_dict


def get_functionality_script(output_directory=',', number_of_iter=0, experiment_repeats=1, number_nan_repeats=0):
    """
    A computationally simple pre-setup system which can be loaded to manipulate a formal experiment
    Should not be used for scientific testing per-se, but can be used to test functional components of experiment

    Parameters
    ==========
    output_directory : str, Optional
        Output directory to set in script
    number_of_iter : int Optional, Default: 1
        Number of iterations to run
    experiment_repeats : int, Optional, Default: 1
        Number of times the experiment is repeated in a "experiments" header
    number_nan_repeats : int, Optional, Default: 0
        Number of times the experiment with a NaN is repeated, this will be added to the end of the stack
    """
    paths = examples_paths()
    template_script = """
    ---
    options:
      minimize: no
      verbose: no
      output_dir: {output_directory}
      default_number_of_iterations: {number_of_iter}
      default_nsteps_per_iteration: 10
      temperature: 300*kelvin
      pressure: null
      anisotropic_dispersion_cutoff: null

    solvents:
      vacuum:
        nonbonded_method: NoCutoff

    systems:
      premade:
        phase1_path: {boxless_path}
        phase2_path: {boxless_path}
        ligand_dsl: resname ene
        solvent: vacuum
      premade_nan:
        phase1_path: {nan_path}
        phase2_path: {nan_path}
        ligand_dsl: resname ene
        solvent: vacuum

    protocols:
      absolute-binding:
        complex:
          alchemical_path:
            lambda_electrostatics: [0.0, 0.0]
            lambda_sterics:        [0.0, 0.0]
        solvent:
          alchemical_path:
            lambda_electrostatics: [1.0, 1.0]
            lambda_sterics:        [1.0, 1.0]

    the_exp:
      system: premade
      protocol: absolute-binding
      restraint:
        type: FlatBottom
    the_nan_exp:
      system: premade_nan
      protocol: absolute-binding
      restraint:
        type: FlatBottom

    experiments: [{repeating}]

    """
    repeating_string = ', '.join(['the_exp'] * experiment_repeats)
    repeating_nan_string = ', '.join(['the_nan_exp'] * number_nan_repeats)
    if repeating_string != '':
        repeating_string += ', '
    repeating_string += repeating_nan_string
    return yank_load(template_script.format(output_directory=output_directory,
                                            number_of_iter=number_of_iter,
                                            repeating=repeating_string,
                                            boxless_path=paths['benzene-toluene-boxless'],
                                            nan_path=paths['benzene-toluene-nan']))


# ==============================================================================
# YAML parsing and validation
# ==============================================================================

def test_yaml_parsing():
    """Check that YAML file is parsed correctly."""

    # Parser handles no options
    yaml_content = """
    ---
    test: 2
    """
    exp_builder = ExperimentBuilder(textwrap.dedent(yaml_content))
    # The plus 1 is because we overwrite disable_alchemical_dispersion_correction.
    expected_n_options = (len(exp_builder.GENERAL_DEFAULT_OPTIONS) +
                          len(exp_builder.EXPERIMENT_DEFAULT_OPTIONS) + 1)
    assert len(exp_builder._options) == expected_n_options

    # Correct parsing
    yaml_content = """
    ---
    options:
        verbose: true
        resume_setup: true
        resume_simulation: true
        output_dir: /path/to/output/
        setup_dir: /path/to/output/setup/
        experiments_dir: /path/to/output/experiments/
        platform: CPU
        precision: mixed
        switch_experiment_interval: -2.0
        processes_per_experiment: 2
        max_n_contexts: 9
        switch_phase_interval: 32
        temperature: 300*kelvin
        pressure: null
        constraints: AllBonds
        hydrogen_mass: 2*amus
        randomize_ligand: yes
        randomize_ligand_sigma_multiplier: 1.0e-2
        randomize_ligand_close_cutoff: 1.5 * angstrom
        anisotropic_dispersion_cutoff: null
        default_timestep: 2.0 * femtosecond
        default_nsteps_per_iteration: 2500
        default_number_of_iterations: .inf
        equilibration_timestep: 1.0 * femtosecond
        number_of_equilibration_iterations: 100
        minimize: False
        minimize_tolerance: 1.0 * kilojoules_per_mole / nanometers
        minimize_max_iterations: 0
        annihilate_sterics: no
        annihilate_electrostatics: true
        alchemical_pme_treatment: direct-space
        disable_alchemical_dispersion_correction: no
    """

    exp_builder = ExperimentBuilder(textwrap.dedent(yaml_content))
    assert len(exp_builder._options) == 32

    # The global context cache has been set.
    assert mmtools.cache.global_context_cache.capacity == 9

    # Check correct types
    assert exp_builder._options['output_dir'] == '/path/to/output/'
    assert exp_builder._options['pressure'] is None
    assert exp_builder._options['constraints'] == openmm.app.AllBonds
    assert exp_builder._options['anisotropic_dispersion_cutoff'] is None
    assert exp_builder._options['default_timestep'] == 2.0 * unit.femtoseconds
    assert exp_builder._options['randomize_ligand_sigma_multiplier'] == 1.0e-2
    assert exp_builder._options['default_nsteps_per_iteration'] == 2500
    assert type(exp_builder._options['default_nsteps_per_iteration']) is int
    assert exp_builder._options['default_number_of_iterations'] == float('inf')
    assert exp_builder._options['number_of_equilibration_iterations'] == 100
    assert type(exp_builder._options['number_of_equilibration_iterations']) is int
    assert exp_builder._options['minimize'] is False


def test_paths_properties():
    """Test that setup directory is updated correctly when changing output paths."""
    template_script = get_template_script(output_dir='output1')
    template_script['options']['setup_dir'] = 'setup1'
    exp_builder = ExperimentBuilder(template_script)

    # The database path is configured correctly.
    assert exp_builder._db.setup_dir == os.path.join('output1', 'setup1')

    # Updating paths also updates the database main directory.
    exp_builder.output_dir = 'output2'
    exp_builder.setup_dir = 'setup2'
    assert exp_builder._db.setup_dir == os.path.join('output2', 'setup2')


def test_online_reads_checkpoint():
    """Test that online analysis reads the checkpoint correctly in all cases"""
    current_log_level = logger.level
    logger.setLevel(logging.ERROR)  # Temporarily suppress some of the logging output
    raw_template_script = get_template_script(systems=['explicit-system'])

    # Pair down the processing
    allowed_molecules = [raw_template_script['systems']['explicit-system']['receptor'],
                         raw_template_script['systems']['explicit-system']['ligand']]
    popables = []
    for molecule in raw_template_script['molecules'].keys():
        if molecule not in allowed_molecules:
            popables.append(molecule)
    for popable in popables:
        raw_template_script['molecules'].pop(popable)

    raw_template_script.pop('samplers')
    sampler_entry = {'type': 'SAMSSampler'}
    sampler = {'samplers': {'sams': sampler_entry}}
    base_template_script = {**raw_template_script, **sampler}
    base_template_script['experiments']['sampler'] = 'sams'

    def spinup_sampler(script):
        with mmtools.utils.temporary_directory() as tmp_dir:
            template_script['options']['output_dir'] = tmp_dir
            exp_builder = ExperimentBuilder(script)
            experiment = [ex for ex in exp_builder.build_experiments()][0]
            sampler = experiment.phases[0].sampler
        return sampler

    # Testing Note: All the test numbers for the checkpoint_interval below are different from the default and each
    # other to ensure making changes actually has the intended effect odd settings get carried over between tests
    # respectively.

    # Test that setting "checkpoint" for online analysis gets the checkpoint interval default
    template_script = copy.deepcopy(base_template_script)
    template_script['options'].pop("checkpoint_interval", None)
    template_script['samplers']['sams']['online_analysis_interval'] = "checkpoint"
    sampler = spinup_sampler(template_script)
    assert sampler.online_analysis_interval == AlchemicalPhaseFactory.DEFAULT_OPTIONS['checkpoint_interval']

    # Test that setting "checkpoint" for online analysis gets the checkpoint interval that is set
    template_script['options']["checkpoint_interval"] = 10
    sampler = spinup_sampler(template_script)
    assert sampler.online_analysis_interval == 10

    # Test that not setting online analysis gets the checkpoint interval default
    template_script = copy.deepcopy(base_template_script)
    template_script['options'].pop("checkpoint_interval", None)
    template_script['samplers']['sams'].pop('online_analysis_interval', None)
    sampler = spinup_sampler(template_script)
    assert sampler.online_analysis_interval == AlchemicalPhaseFactory.DEFAULT_OPTIONS['checkpoint_interval']

    # Test that not setting online analysis gets the checkpoint interval that is set
    template_script['options']["checkpoint_interval"] = 100
    sampler = spinup_sampler(template_script)
    assert sampler.online_analysis_interval == 100

    # Test that setting online analysis still returns the set value
    template_script = copy.deepcopy(base_template_script)
    template_script['options']["checkpoint_interval"] = 70
    template_script['samplers']['sams']['online_analysis_interval'] = 13
    sampler = spinup_sampler(template_script)
    assert sampler.online_analysis_interval == 13

    # Test that setting online analysis to None keeps online analysis None
    template_script = copy.deepcopy(base_template_script)
    template_script['options']["checkpoint_interval"] = 80
    template_script['samplers']['sams']['online_analysis_interval'] = None
    sampler = spinup_sampler(template_script)
    assert sampler.online_analysis_interval is None

    # Test that not setting a sampler gets a the checkpoint interval for online analysis
    template_script = copy.deepcopy(base_template_script)
    template_script['options']["checkpoint_interval"] = 90
    template_script.pop('samplers', None)
    template_script['experiments'].pop('sampler', None)
    sampler = spinup_sampler(template_script)
    assert sampler.online_analysis_interval == 90

    # Test that setting the checkpoint_interval in *experiments:options* block correctly sets the checkpoint interval
    template_script = copy.deepcopy(base_template_script)
    template_script['options']["checkpoint_interval"] = 110
    template_script.pop('samplers', None)
    opts = {'checkpoint_interval': 120}
    template_script['experiments'].pop('sampler', None)
    template_script['experiments']['options'] = opts
    sampler = spinup_sampler(template_script)
    assert sampler.online_analysis_interval == 120

    logger.setLevel(current_log_level)  # Reset logging to normal


def test_processes_per_experiment():
    """Test the determination of processes_per_experiment option."""
    # Create a script with 4 experiments.
    template_script = get_template_script()
    template_script['experiment1'] = copy.deepcopy(template_script['experiments'])
    template_script['experiment1']['system'] = utils.CombinatorialLeaf(['explicit-system', 'implicit-system'])
    # The first two experiments have less number of states than the other two.
    template_script['experiment1']['protocol'] = 'hydration-protocol'
    template_script['experiment2'] = copy.deepcopy(template_script['experiments'])
    template_script['experiment2']['system'] = 'hydration-system'
    # The last experiment uses SAMS.
    template_script['experiment2']['sampler'] = utils.CombinatorialLeaf(['repex', 'sams'])
    template_script['experiments'] = ['experiment1', 'experiment2']

    exp_builder = ExperimentBuilder(template_script)
    experiments = list(exp_builder._expand_experiments())

    # The default is auto.
    assert exp_builder._options['processes_per_experiment'] == 'auto'

    # When there is no MPI environment the calculation is serial.
    assert exp_builder._get_experiment_mpi_group_size(experiments) is None

    # In an MPI environment, the MPI communicator is split according
    # to the number of experiments still have to be completed. Each
    # test case is pair (experiments, MPICOMM size, expected return value).
    test_cases = [
        (experiments, 5, 1),  # This contains a SAMS sampler so only 1 MPI process is used.
        (experiments[:-1], 4, [1, 1, 2]),  # 3 repex samples, but last experiment has more intermediate states.
        (experiments[1:-1], 4, [2, 2]),  # 2 repex samples on 4 MPI processes.
        (experiments[1:-1], 6, [3, 3]),  # 2 repex samples on 4 MPI processes.
        (list(reversed(experiments[1:-1])), 3, [2, 1]),  # 2 repex samples on 3 MPI processes.
        (experiments[:-1], 2, 1),  # Less MPI processes than experiments, split everything.
    ]

    for i, (exp, mpicomm_size, expected_result) in enumerate(test_cases):
        with mpiplus.mpiplus._simulated_mpi_environment(size=mpicomm_size):
            result = exp_builder._get_experiment_mpi_group_size(exp)
            err_msg = ('experiments: {}\nMPICOMM size: {}\nexpected result: {}'
                       '\nresult: {}').format(*test_cases[i], result)
            assert result == expected_result, err_msg

    # Test manual setting of processes_per_experiments.
    test_cases = [2, None]
    for processes_per_experiment in test_cases:
        exp_builder._options['processes_per_experiment'] = processes_per_experiment
        # Serial execution is always None.
        assert exp_builder._get_experiment_mpi_group_size(experiments) is None
        with mpiplus.mpiplus._simulated_mpi_environment(size=5):
            assert exp_builder._get_experiment_mpi_group_size(experiments[:-1]) == processes_per_experiment
            # When there are SAMS sampler, it's always 1.
            assert exp_builder._get_experiment_mpi_group_size(experiments) == 1


def test_validation_wrong_options():
    """YAML validation raises exception with wrong molecules."""
    options = [
        ("found unknown parameter", {'unknown_options': 3}),
        ("parameter minimize=100 is incompatible with True", {'minimize': 100}),
        ("invalid literal for int", {'processes_per_experiment': 'incorrect_string'})
    ]
    for regex, option in options:
        yield assert_raises_regexp, YamlParseError, regex, ExperimentBuilder._validate_options, option, True


def test_validation_correct_molecules():
    """Correct molecules YAML validation."""
    paths = examples_paths()
    molecules = [
        {'name': 'toluene', 'leap': {'parameters': 'leaprc.gaff'}},
        {'name': 'toluene', 'leap': {'parameters': ['leaprc.gaff', 'toluene.frcmod']}},
        {'name': 'p-xylene', 'antechamber': {'charge_method': 'bcc'}},
        {'smiles': 'Cc1ccccc1', 'openeye': {'quacpac': 'am1-bcc'},
            'antechamber': {'charge_method': None}},
        {'name': 'p-xylene', 'antechamber': {'charge_method': 'bcc'},
            'epik': {'ph': 7.6, 'ph_tolerance': 0.7, 'tautomerize': False, 'select': 0}},
        {'smiles': 'Cc1ccccc1', 'openeye': {'quacpac': 'am1-bcc'},
            'antechamber': {'charge_method': None}, 'epik': {'select': 1}},

        {'filepath': paths['abl']},
        {'filepath': paths['abl'], 'leap': {'parameters': 'leaprc.ff99SBildn'}},
        {'filepath': paths['abl'], 'leap': {'parameters': 'leaprc.ff99SBildn'}, 'select': 1},
        {'filepath': paths['abl'], 'select': 'all'},
        {'filepath': paths['abl'], 'select': 'all', 'strip_protons': True},
        {'filepath': paths['abl'], 'select': 'all', 'pdbfixer': {}},
        {'filepath': paths['abl'], 'select': 'all', 'pdbfixer': {'add_missing_residues': True}},
        {'filepath': paths['abl'], 'select': 'all', 'pdbfixer': {'add_missing_atoms': 'all', 'ph': '8.0'}},
        {'filepath': paths['abl'], 'select': 'all', 'pdbfixer': {'remove_heterogens': 'all'}},
        {'filepath': paths['abl'], 'select': 'all', 'pdbfixer': {'replace_nonstandard_residues': True}},
        {'filepath': paths['abl'], 'select': 'all', 'pdbfixer': {'apply_mutations': {'chain_id': 'A', 'mutations': 'T85I'}}},
        {'filepath': paths['abl'], 'select': 'all', 'modeller': {'apply_mutations': {'chain_id': 'A', 'mutations': 'T85I'}}},
        {'filepath': paths['abl'], 'select': 'all', 'modeller': {'apply_mutations': {'chain_id': 'A', 'mutations': 'WT'}}},
        {'filepath': paths['abl'], 'select': 'all', 'pdbfixer': {'apply_mutations': {'chain_id': 'A', 'mutations': 'I8A/T9A'}}},
        {'filepath': paths['toluene'], 'leap': {'parameters': 'leaprc.gaff'}},
        {'filepath': paths['benzene'], 'epik': {'select': 1, 'tautomerize': False}},
        # Regions tests, make sure all other combos still work
        {'name': 'toluene', 'regions': {'a_region': 4}},
        {'name': 'toluene', 'regions': {'a_region': 'dsl string'}},
        {'name': 'toluene', 'regions': {'a_region': [0, 2, 3]}},
        {'name': 'toluene', 'regions': {'a_region': [0, 2, 3], 'another_region': [5, 4, 3]}},
        {'smiles': 'Cc1ccccc1', 'regions': {'a_region': 4}},
        {'smiles': 'Cc1ccccc1', 'regions': {'a_region': 'dsl string'}},
        {'smiles': 'Cc1ccccc1', 'regions': {'a_region': [0, 2, 3]}},
        {'smiles': 'Cc1ccccc1', 'regions': {'a_region': [0, 2, 3], 'another_region': [5, 4, 3]}},
        {'filepath': paths['abl'], 'regions': {'a_region': 4}},
        {'filepath': paths['abl'], 'regions': {'a_region': 'dsl string'}},
        {'filepath': paths['abl'], 'regions': {'a_region': [0, 2, 3]}},
        {'filepath': paths['abl'], 'regions': {'a_region': [0, 2, 3], 'another_region': [5, 4, 3]}},
        {'filepath': paths['toluene'], 'regions': {'a_region': 4}},
        {'filepath': paths['toluene'], 'regions': {'a_region': 'dsl string'}},
        {'filepath': paths['toluene'], 'regions': {'a_region': [0, 2, 3]}},
        {'filepath': paths['toluene'], 'regions': {'a_region': [0, 2, 3], 'another_region': [5, 4, 3]}}
    ]
    for molecule in molecules:
        yield ExperimentBuilder._validate_molecules, {'mol': molecule}


def test_validation_wrong_molecules():
    """YAML validation raises exception with wrong molecules."""
    paths = examples_paths()
    paths['wrongformat'] = utils.get_data_filename(os.path.join('tests', 'data', 'README.md'))
    molecules = [
        {'antechamber': {'charge_method': 'bcc'}},
        {'filepath': paths['wrongformat']},
        {'name': 'p-xylene', 'antechamber': {'charge_method': 'bcc'}, 'unknown': 4},
        {'smiles': 'Cc1ccccc1', 'openeye': {'quacpac': 'am1-bcc'}},
        {'smiles': 'Cc1ccccc1', 'openeye': {'quacpac': 'invalid'},
            'antechamber': {'charge_method': None}},
        {'smiles': 'Cc1ccccc1', 'openeye': {'quacpac': 'am1-bcc'},
            'antechamber': {'charge_method': 'bcc'}},
        {'filepath': 'nonexistentfile.pdb', 'leap': {'parameters': 'leaprc.ff14SB'}},
        {'filepath': paths['toluene'], 'smiles': 'Cc1ccccc1'},
        {'filepath': paths['toluene'], 'strip_protons': True},
        {'filepath': paths['abl'], 'leap': {'parameters': 'oldff/leaprc.ff14SB'}, 'epik': {'select': 0}},
        {'name': 'toluene', 'epik': 0},
        {'name': 'toluene', 'epik': {'tautomerize': 6}},
        {'name': 'toluene', 'epik': {'extract_range': 1}},
        {'name': 'toluene', 'smiles': 'Cc1ccccc1'},
        {'name': 3},
        {'smiles': 'Cc1ccccc1', 'select': 1},
        {'name': 'Cc1ccccc1', 'select': 1},
        {'filepath': paths['abl'], 'leap': {'parameters': 'oldff/leaprc.ff14SB'}, 'select': 'notanoption'},
        {'filepath': paths['abl'], 'regions': 5},
        {'filepath': paths['abl'], 'regions': {'a_region': [-56, 5.23]}},
        {'filepath': paths['toluene'], 'leap': {'parameters': 'leaprc.gaff'}, 'strip_protons': True},
    ]
    for molecule in molecules:
        yield assert_raises, YamlParseError, ExperimentBuilder._validate_molecules, {'mol': molecule}


def test_validation_correct_solvents():
    """Correct solvents YAML validation."""
    solvents = [
        {'nonbonded_method': 'Ewald', 'nonbonded_cutoff': '3*nanometers'},
        {'nonbonded_method': 'PME', 'solvent_model': 'tip4pew'},
        {'nonbonded_method': 'PME', 'solvent_model': 'tip3p', 'leap': {'parameters': 'leaprc.water.tip3p'}},
        {'nonbonded_method': 'PME', 'clearance': '3*angstroms'},
        {'nonbonded_method': 'PME'},
        {'nonbonded_method': 'NoCutoff', 'implicit_solvent': 'OBC2'},
        {'nonbonded_method': 'CutoffPeriodic', 'nonbonded_cutoff': '9*angstroms',
         'clearance': '9*angstroms', 'positive_ion': 'Na+', 'negative_ion': 'Cl-',
         'ionic_strength': '200*millimolar'},
        {'implicit_solvent': 'OBC2', 'implicit_solvent_salt_conc': '1.0*nanomolar'},
        {'nonbonded_method': 'PME', 'clearance': '3*angstroms', 'ewald_error_tolerance': 0.001},
    ]
    for solvent in solvents:
        yield ExperimentBuilder._validate_solvents, {'solv': solvent}


def test_validation_wrong_solvents():
    """YAML validation raises exception with wrong solvents."""
    # Each test case is a pair (regexp_error, solvent_description).
    solvents = [
        ("nonbonded_cutoff:\n- can be specified only with the following nonbonded methods \['CutoffPeriodic', 'CutoffNonPeriodic',\n  'Ewald', 'PME'\]",
            {'nonbonded_cutoff': '3*nanometers'}),
        ("solvent_model:\n- unallowed value unknown_solvent_model",
            {'nonbonded_method': 'PME', 'solvent_model': 'unknown_solvent_model'}),
        ("leap:\n- must be of dict type",
            {'nonbonded_method': 'PME', 'solvent_model': 'tip3p', 'leap': 'leaprc.water.tip3p'}),
        ("implicit_solvent:\n- can be specified only if nonbonded method is NoCutoff",
            {'nonbonded_method': 'PME', 'clearance': '3*angstroms', 'implicit_solvent': 'OBC2'}),
        ("blabla:\n- unknown field",
            {'nonbonded_method': 'NoCutoff', 'blabla': '3*nanometers'}),
        ("''implicit_solvent'' cannot be coerced: module ''simtk.openmm.app'' has no\n  attribute ''OBX2'''",
            {'nonbonded_method': 'NoCutoff', 'implicit_solvent': 'OBX2'}),
        ("''implicit_solvent_salt_conc'' cannot be coerced: Units of 1.0\*angstrom",
            {'implicit_solvent': 'OBC2', 'implicit_solvent_salt_conc': '1.0*angstrom'})
    ]
    for regexp, solvent in solvents:
        yield assert_raises_regexp, YamlParseError, regexp, ExperimentBuilder._validate_solvents, {'solv': solvent}


def test_validation_correct_systems():
    """Correct systems YAML validation."""
    data_paths = examples_paths()
    exp_builder = ExperimentBuilder()
    basic_script = """
    ---
    molecules:
        rec: {{filepath: {0}, leap: {{parameters: leaprc.ff14SB}}}}
        rec_reg: {{filepath: {0}, regions: {{receptregion: 'some dsl'}}, leap: {{parameters: leaprc.ff14SB}}}}
        lig: {{name: lig, leap: {{parameters: leaprc.gaff}}}}
        lig_reg: {{name: lig, regions: {{ligregion: [143, 123]}}, leap: {{parameters: leaprc.gaff}}}}
    solvents:
        solv: {{nonbonded_method: NoCutoff}}
        solv2: {{nonbonded_method: NoCutoff, implicit_solvent: OBC2}}
        solv3: {{nonbonded_method: PME, clearance: 10*angstroms}}
        solv4: {{nonbonded_method: PME}}
    """.format(data_paths['lysozyme'])
    basic_script = yaml.load(textwrap.dedent(basic_script), Loader=yaml.FullLoader)

    systems = [
        {'receptor': 'rec', 'ligand': 'lig', 'solvent': 'solv'},
        {'receptor': 'rec_reg', 'ligand': 'lig_reg', 'solvent': 'solv'},
        {'receptor': 'rec_reg', 'ligand': 'lig', 'solvent': 'solv'},
        {'receptor': 'rec', 'ligand': 'lig', 'solvent': 'solv', 'pack': True},
        {'receptor': 'rec', 'ligand': 'lig', 'solvent': 'solv3',
            'leap': {'parameters': ['leaprc.gaff', 'leaprc.ff14SB']}},

        {'phase1_path': data_paths['bentol-complex'],
         'phase2_path': data_paths['bentol-solvent'],
         'ligand_dsl': 'resname BEN', 'solvent': 'solv'},
        {'phase1_path': data_paths['bentol-complex'],
         'phase2_path': data_paths['bentol-solvent'],
         'ligand_dsl': 'resname BEN', 'solvent': 'solv4'},
        {'phase1_path': data_paths['bentol-complex'],
         'phase2_path': data_paths['bentol-solvent'],
         'ligand_dsl': 'resname BEN', 'solvent1': 'solv3',
         'solvent2': 'solv2'},

        {'phase1_path': data_paths['pxylene-complex'],
         'phase2_path': data_paths['pxylene-solvent'],
         'ligand_dsl': 'resname p-xylene', 'solvent': 'solv',
         'gromacs_include_dir': data_paths['pxylene-gro-include']},
        {'phase1_path': data_paths['pxylene-complex'],
         'phase2_path': data_paths['pxylene-solvent'],
         'ligand_dsl': 'resname p-xylene', 'solvent': 'solv'},

        {'phase1_path': data_paths['toluene-solvent'],
         'phase2_path': data_paths['toluene-vacuum'],
         'ligand_dsl': 'resname TOL'},
        {'phase1_path': data_paths['toluene-solvent'],
         'phase2_path': data_paths['toluene-vacuum'],
         'ligand_dsl': 'resname TOL', 'solvent_dsl': 'not resname TOL'},

        {'solute': 'lig', 'solvent1': 'solv', 'solvent2': 'solv'},
        {'solute': 'lig_reg', 'solvent1': 'solv', 'solvent2': 'solv'},
        {'solute': 'lig', 'solvent1': 'solv', 'solvent2': 'solv',
            'leap': {'parameters': 'leaprc.gaff'}}
    ]
    for system in systems:
        modified_script = basic_script.copy()
        modified_script['systems'] = {'sys': system}
        yield exp_builder.parse, modified_script


def test_validation_wrong_systems():
    """YAML validation raises exception with wrong systems specification."""
    data_paths = examples_paths()
    exp_builder = ExperimentBuilder()
    basic_script = """
    ---
    molecules:
        rec: {{filepath: {0}, leap: {{parameters: oldff/leaprc.ff14SB}}}}
        rec_region: {{filepath: {0}, regions: {{a_region: 'some string'}}, leap: {{parameters: oldff/leaprc.ff14SB}}}}
        lig: {{name: lig, leap: {{parameters: leaprc.gaff}}}}
        lig_region: {{name: lig, regions: {{a_region: 'some string'}}, leap: {{parameters: leaprc.gaff}}}}
    solvents:
        solv: {{nonbonded_method: NoCutoff}}
        solv2: {{nonbonded_method: NoCutoff, implicit_solvent: OBC2}}
        solv3: {{nonbonded_method: PME, clearance: 10*angstroms}}
        solv4: {{nonbonded_method: PME}}
    """.format(data_paths['lysozyme'])
    basic_script = yaml.load(textwrap.dedent(basic_script), Loader=yaml.FullLoader)

    # Each test case is a pair (regexp_error, system_description).
    systems = [
        ("'solvent' is required",
            {'receptor': 'rec', 'ligand': 'lig'}),

        ("regions\(s\) clashing",
            {'receptor': 'rec_region', 'ligand': 'lig_region', 'solvent': 'solv'}),

        ("ligand:\n- must be of string type",
            {'receptor': 'rec', 'ligand': 1, 'solvent': 'solv'}),

        ("solvent:\n- must be of string type",
            {'receptor': 'rec', 'ligand': 'lig', 'solvent': ['solv', 'solv']}),

        ("unallowed value unknown",
            {'receptor': 'rec', 'ligand': 'lig', 'solvent': 'unknown'}),

        ("solv4 does not specify clearance",
            {'receptor': 'rec', 'ligand': 'lig', 'solvent': 'solv4',
             'leap': {'parameters': ['leaprc.gaff', 'leaprc.ff14SB']}}),

        ("parameters:\n- unknown field",
            {'receptor': 'rec', 'ligand': 'lig', 'solvent': 'solv3',
             'parameters': 'leaprc.ff14SB'}),

        ("phase1_path:\n- must be of list type",
            {'phase1_path': data_paths['bentol-complex'][0],
             'phase2_path': data_paths['bentol-solvent'],
             'ligand_dsl': 'resname BEN', 'solvent': 'solv'}),

        ("File path nonexistingpath.prmtop does not exist.",
            {'phase1_path': ['nonexistingpath.prmtop', 'nonexistingpath.inpcrd'],
            'phase2_path': data_paths['bentol-solvent'],
            'ligand_dsl': 'resname BEN', 'solvent': 'solv'}),

        ("ligand_dsl:\n- must be of string type",
            {'phase1_path': data_paths['bentol-complex'],
             'phase2_path': data_paths['bentol-solvent'],
             'ligand_dsl': 3.4, 'solvent': 'solv'}),

        ("unallowed value unknown",
            {'phase1_path': data_paths['bentol-complex'],
             'phase2_path': data_paths['bentol-solvent'],
             'ligand_dsl': 'resname BEN', 'solvent1': 'unknown',
             'solvent2': 'solv2'}),

        ("unallowed value cantbespecified",
            {'phase1_path': data_paths['toluene-solvent'],
             'phase2_path': data_paths['toluene-vacuum'],
             'ligand_dsl': 'resname TOL', 'solvent': 'cantbespecified'}),

        ("field 'ligand' is required",
            {'receptor': 'rec', 'solute': 'lig', 'solvent1': 'solv', 'solvent2': 'solv'}),

        ("''ligand'' must not be present with ''solute''",
            {'ligand': 'lig', 'solute': 'lig', 'solvent1': 'solv', 'solvent2': 'solv'}),

        ("leap:\n- must be of dict type",
            {'solute': 'lig', 'solvent1': 'solv', 'solvent2': 'solv', 'leap': 'leaprc.gaff'})
    ]
    for regexp, system in systems:
        modified_script = basic_script.copy()
        modified_script['systems'] = {'sys': system}
        yield assert_raises_regexp, YamlParseError, regexp, exp_builder.parse, modified_script


def test_validation_correct_mcmc_moves():
    """Correct samplers YAML validation."""
    mcmc_moves = [
        {'type': 'LangevinSplittingDynamicsMove', 'reassign_velocities': False,
         'splitting': 'VRORV', 'n_steps': 10, 'timestep': '2.0*femtosecond'},
        {'type': 'SequenceMove', 'move_list': [
            {'type': 'MCDisplacementMove', 'displacement_sigma': '5.0*nanometers'},
            {'type': 'LangevinSplittingDynamicsMove'}
        ]},
    ]
    for mcmc_move in mcmc_moves:
        yield ExperimentBuilder._validate_mcmc_moves, {'mcmc_moves': {'mcmcmove1': mcmc_move}}


def test_validation_wrong_mcmc_moves():
    """YAML validation raises exception with wrong mcmc move specification."""
    # Each test case is a pair (regexp_error, mcmc_move_description).
    mcmc_moves = [
        ("The expression 2.0 must be a string",
            {'type': 'LangevinSplittingDynamicsMove', 'timestep': 2.0}),
        ("Could not find class UnknownMoveClass",
            {'type': 'UnknownMoveClass'}),
        ("Could not find class NestedUnknownMoveClass",
            {'type': 'SequenceMove', 'move_list': [
                {'type': 'MCDisplacementMove'},
                {'type': 'NestedUnknownMoveClass'}
        ]})
    ]
    for regexp, mcmc_move in mcmc_moves:
        script = {'mcmc_moves': {'mcmc_move1': mcmc_move}}
        yield assert_raises_regexp, YamlParseError, regexp, ExperimentBuilder._validate_mcmc_moves, script


def test_validation_correct_samplers():
    """Correct samplers YAML validation."""
    samplers = [
        {'type': 'MultiStateSampler', 'locality': 3},
        {'type': 'ReplicaExchangeSampler'},
        # MCMCMove 'single' is defined in get_template_script().
        {'type': 'SAMSSampler', 'mcmc_moves': 'single'},
        {'type': 'ReplicaExchangeSampler', 'number_of_iterations': 5, 'replica_mixing_scheme': 'swap-neighbors'},
        {'type': 'ReplicaExchangeSampler', 'number_of_iterations': 5, 'replica_mixing_scheme': None}
    ]
    exp_builder = ExperimentBuilder(get_template_script())
    for sampler in samplers:
        script = {'samplers': {'sampler1': sampler}}
        yield exp_builder._validate_samplers, script


def test_validation_wrong_samplers():
    """YAML validation raises exception with wrong experiments specification."""
    # Each test case is a pair (regexp_error, sampler_description).
    samplers = [
        ("locality must be an int",
            {'type': 'MultiStateSampler', 'locality': 3.0}),
        ("unallowed value unknown",
            {'type': 'ReplicaExchangeSampler', 'mcmc_moves': 'unknown'}),
        ("Could not find class NonExistentSampler",
            {'type': 'NonExistentSampler'}),
        ("found unknown parameter",
            {'type': 'ReplicaExchangeSampler', 'unknown_kwarg': 5}),
    ]
    exp_builder = ExperimentBuilder(get_template_script())
    for regexp, sampler in samplers:
        script = {'samplers': {'sampler1': sampler}}
        yield assert_raises_regexp, YamlParseError, regexp, exp_builder._validate_samplers, script


def test_order_phases():
    """YankLoader preserves protocol phase order."""
    yaml_content_template = """
    ---
    absolute-binding:
        {}:
            alchemical_path:
                lambda_electrostatics: [1.0, 0.5, 0.0]
                lambda_sterics: [1.0, 0.5, 0.0]
        {}:
            alchemical_path:
                lambda_electrostatics: [1.0, 0.5, 0.0]
                lambda_sterics: [1.0, 0.5, 0.0]
        {}:
            alchemical_path:
                lambda_electrostatics: [1.0, 0.5, 0.0]
                lambda_sterics: [1.0, 0.5, 0.0]"""

    # Find order of phases for which normal parsing is not ordered or the test is useless
    for ordered_phases in itertools.permutations(['athirdphase', 'complex', 'solvent']):
        yaml_content = yaml_content_template.format(*ordered_phases)
        parsed = yaml.load(textwrap.dedent(yaml_content), Loader=yaml.FullLoader)
        if tuple(parsed['absolute-binding'].keys()) != ordered_phases:
            break

    # Insert !Ordered tag
    yaml_content = yaml_content.replace('binding:', 'binding: !Ordered')
    parsed = yank_load(yaml_content)
    assert tuple(parsed['absolute-binding'].keys()) == ordered_phases


def test_validation_correct_protocols():
    """Correct protocols YAML validation."""
    basic_protocol = yank_load(standard_protocol)

    # Alchemical paths
    protocols = [
        {'lambda_electrostatics': [1.0, 0.5, 0.0], 'lambda_sterics': [1.0, 0.5, 0.0]},
        {'lambda_electrostatics': [1.0, 0.5, 0.0], 'lambda_sterics': [1.0, 0.5, 0.0],
         'lambda_torsions': [1.0, 0.5, 0.0], 'lambda_angles': [1.0, 0.5, 0.0]},
        {'lambda_electrostatics': [1.0, 0.5, 0.0], 'lambda_sterics': [1.0, 0.5, 0.0],
         'temperature': ['300*kelvin', '340*kelvin', '300*kelvin']},
        'auto',
    ]
    for protocol in protocols:
        modified_protocol = copy.deepcopy(basic_protocol)
        modified_protocol['absolute-binding']['complex']['alchemical_path'] = protocol
        yield ExperimentBuilder._validate_protocols, modified_protocol

    # Try different options both with 'auto' and a path with alchemical functions.
    function_protocol = copy.deepcopy(basic_protocol)
    function_protocol['absolute-binding']['complex']['alchemical_path'] = {
        'lambda_electrostatics': 'lambda**2',
        'lambda_sterics': 'sqrt(lambda)',
        'lambda': [1.0, 0.0]
    }
    auto_protocol = copy.deepcopy(basic_protocol)
    auto_protocol['absolute-binding']['complex']['alchemical_path'] = 'auto'

    trailblazer_options = [
        {'n_equilibration_iterations': 1000, 'n_samples_per_state': 100,
         'std_potential_threshold': 0.5, 'threshold_tolerance': 0.05},
        {'n_equilibration_iterations': 100, 'n_samples_per_state': 10},
        {'std_potential_threshold': 1.0, 'threshold_tolerance': 0.5},
        {'function_variable_name': 'lambda'},
        {'function_variable_name': 'lambda', 'reverse_direction': False}
    ]
    for opts in trailblazer_options:
        # Use the function protocol if the function variable is specified.
        if 'function_variable_name' in opts:
            modified_protocol = copy.deepcopy(function_protocol)
        else:
            modified_protocol = copy.deepcopy(auto_protocol)
        modified_protocol['absolute-binding']['complex']['trailblazer_options'] = opts
        yield ExperimentBuilder._validate_protocols, modified_protocol

    # Multiple phases.
    alchemical_path = copy.deepcopy(basic_protocol['absolute-binding']['complex'])
    protocols = [
        {'complex': alchemical_path, 'solvent': alchemical_path},
        {'complex': alchemical_path, 'solvent': {'alchemical_path': 'auto'}},
        {'my-complex': alchemical_path, 'my-solvent': alchemical_path},
        {'solvent1': alchemical_path, 'solvent2': alchemical_path},
        {'solvent1variant': alchemical_path, 'solvent2variant': alchemical_path},
        collections.OrderedDict([('a', alchemical_path), ('z', alchemical_path)]),
        collections.OrderedDict([('z', alchemical_path), ('a', alchemical_path)])
    ]
    for protocol in protocols:
        modified_protocol = copy.deepcopy(basic_protocol)
        modified_protocol['absolute-binding'] = protocol
        yield ExperimentBuilder._validate_protocols, modified_protocol
        sorted_protocol = ExperimentBuilder._validate_protocols(modified_protocol)['absolute-binding']
        if isinstance(protocol, collections.OrderedDict):
            assert sorted_protocol.keys() == protocol.keys()
        else:
            assert isinstance(sorted_protocol, collections.OrderedDict)
            first_phase = next(iter(sorted_protocol.keys()))  # py2/3 compatible
            assert 'complex' in first_phase or 'solvent1' in first_phase


def test_validation_wrong_protocols():
    """YAML validation raises exception with wrong alchemical protocols."""
    basic_protocol = yank_load(standard_protocol)

    # Alchemical paths
    protocols = [
        {'lambda_electrostatics': [1.0, 0.5, 0.0]},
        {'lambda_electrostatics': [1.0, 0.5, 0.0], 'lambda_sterics': [1.0, 0.5, 'wrong!']},
        {'lambda_electrostatics': [1.0, 0.5, 0.0], 'lambda_sterics': [1.0, 0.5, 11000.0]},
        {'lambda_electrostatics': [1.0, 0.5, 0.0], 'lambda_sterics': [1.0, 0.5, -0.5]},
        {'lambda_electrostatics': [1.0, 0.5, 0.0], 'lambda_sterics': 0.0},
        {'lambda_electrostatics': [1.0, 0.5, 0.0], 'lambda_sterics': [1.0, 0.5, 0.0], 3: 2}
    ]
    for protocol in protocols:
        modified_protocol = copy.deepcopy(basic_protocol)
        modified_protocol['absolute-binding']['complex']['alchemical_path'] = protocol
        yield assert_raises, YamlParseError, ExperimentBuilder._validate_protocols, modified_protocol

    # Try different options both with 'auto' and a path with alchemical functions.
    auto_path = 'auto'
    no_lambda_path = {'lambda_electrostatics': 'lambda**2', 'lambda_sterics': 'sqrt(lambda)'}
    hardcoded_path = {'lambda_electrostatics': [1.0, 0.0], 'lambda_sterics': [1.0, 0.0]}
    correct_lambda_path = {'lambda': [1.0, 0.0], **no_lambda_path}
    str_lambda_path = {'lambda': 'string', **no_lambda_path}
    three_lambda_path = {'lambda': [1.0, 0.5, 0.0], **no_lambda_path}

    # Each test case is (error_regex, options, alchemical_path)
    trailblazer_options = [
        ("n_equilibration_iterations:\n  - must be of integer type",
            {'n_equilibration_iterations': 'bla'}, auto_path),
        ("Only mathematical expressions have been given with no values for their variables",
            {}, no_lambda_path),
        ("Mathematical expressions were detected but no function variable name was given",
            {}, correct_lambda_path),
        ("Function variable name 'lambda' is not defined in 'alchemical_path'",
            {'function_variable_name': 'lambda'}, hardcoded_path),
        ("Only mathematical expressions have been given with no values for their variables",
            {'function_variable_name': 'lambda'}, str_lambda_path),
        ("Only the two end-point values of function variable 'lambda' should be given.",
            {'function_variable_name': 'lambda'}, three_lambda_path),
    ]
    for regex, opts, alchemical_path in trailblazer_options:
        modified_protocol = copy.deepcopy(basic_protocol)
        modified_protocol['absolute-binding']['complex']['alchemical_path'] = alchemical_path
        modified_protocol['absolute-binding']['complex']['trailblazer_options'] = opts
        yield assert_raises_regexp, YamlParseError, regex, ExperimentBuilder._validate_protocols, modified_protocol

    # Phases
    alchemical_path = copy.deepcopy(basic_protocol['absolute-binding']['complex'])
    protocols = [
        {'complex': alchemical_path},
        {2: alchemical_path, 'solvent': alchemical_path},
        {'complex': alchemical_path, 'solvent': alchemical_path, 'thirdphase': alchemical_path},
        {'my-complex-solvent': alchemical_path, 'my-solvent': alchemical_path},
        {'my-complex': alchemical_path, 'my-complex-solvent': alchemical_path},
        {'my-complex': alchemical_path, 'my-complex': alchemical_path},
        {'complex': alchemical_path, 'solvent1': alchemical_path, 'solvent2': alchemical_path},
        {'my-phase1': alchemical_path, 'my-phase2': alchemical_path},
        collections.OrderedDict([('my-phase1', alchemical_path), ('my-phase2', alchemical_path),
                                 ('my-phase3', alchemical_path)])
    ]
    for protocol in protocols:
        modified_protocol = copy.deepcopy(basic_protocol)
        modified_protocol['absolute-binding'] = protocol
        yield assert_raises, YamlParseError, ExperimentBuilder._validate_protocols, modified_protocol


def test_validation_correct_experiments():
    """Correct experimentYAML validation."""
    exp_builder = ExperimentBuilder()
    basic_script = """
    ---
    molecules:
        rec: {{filepath: {}, leap: {{parameters: oldff/leaprc.ff14SB}}}}
        lig: {{name: lig, leap: {{parameters: leaprc.gaff}}}}
    solvents:
        solv: {{nonbonded_method: NoCutoff}}
    systems:
        sys: {{receptor: rec, ligand: lig, solvent: solv}}
    protocols:{}
    """.format(examples_paths()['lysozyme'], standard_protocol)
    basic_script = yank_load(basic_script)

    bor = {'system': 'sys', 'protocol': 'absolute-binding', 'restraint': {
            'type': 'Boresch', 'restrained_receptor_atoms': [1335, 1339, 1397],
            'restrained_ligand_atoms': [2609, 2607, 2606], 'r_aA0': '0.35*nanometer',
            'K_r': '20.0*kilocalories_per_mole/angstrom**2'}}
    period_tor_bor = {**bor}
    period_tor_bor['restraint']['type'] = 'PeriodicTorsionBoresch'

    experiments = [
        {'system': 'sys', 'protocol': 'absolute-binding'},
        {'system': 'sys', 'protocol': 'absolute-binding', 'restraint': {'type': 'Harmonic'}},
        {'system': 'sys', 'protocol': 'absolute-binding', 'restraint': {
            'type': 'Harmonic', 'spring_constant': '8*kilojoule_per_mole/nanometers**2'}},
        {'system': 'sys', 'protocol': 'absolute-binding', 'restraint': {
            'type': 'FlatBottom', 'well_radius': '5.2*nanometers', 'restrained_receptor_atoms': 1644}},
        bor,
        period_tor_bor
    ]
    for experiment in experiments:
        modified_script = basic_script.copy()
        modified_script['experiments'] = experiment
        yield exp_builder.parse, modified_script


def test_validation_wrong_experiments():
    """YAML validation raises exception with wrong experiments specification."""
    exp_builder = ExperimentBuilder()
    basic_script = """
    ---
    molecules:
        rec: {{filepath: {}, leap: {{parameters: oldff/leaprc.ff14SB}}}}
        lig: {{name: lig, leap: {{parameters: leaprc.gaff}}}}
    solvents:
        solv: {{nonbonded_method: NoCutoff}}
    systems:
        sys: {{receptor: rec, ligand: lig, solvent: solv}}
    protocols:{}
    """.format(examples_paths()['lysozyme'], standard_protocol)
    basic_script = yank_load(basic_script)

    experiments = [
        {'system': 'unknownsys', 'protocol': 'absolute-binding'},
        {'system': 'sys', 'protocol': 'unknownprotocol'},
        {'system': 'sys'},
        {'protocol': 'absolute-binding'},

        # Restraint does not specify "type".
        {'system': 'sys', 'protocol': 'absolute-binding', 'restraint': {
            'spring_constant': '8*kilojoule_per_mole/nanometers**2'}},

        # Restraint has unknown constructor parameter.
        {'system': 'sys', 'protocol': 'absolute-binding', 'restraint': {
            'type': 'Harmonic', 'unknown': '3*meters'}},
    ]
    for experiment in experiments:
        modified_script = basic_script.copy()
        modified_script['experiments'] = experiment
        yield assert_raises, YamlParseError, exp_builder.parse, modified_script


# ==============================================================================
# Molecules pipeline
# ==============================================================================

def test_yaml_mol2_antechamber():
    """Test antechamber setup of molecule files."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_content = get_template_script(tmp_dir)
        exp_builder = ExperimentBuilder(yaml_content)
        exp_builder._db._setup_molecules('benzene')

        output_dir = exp_builder._db.get_molecule_dir('benzene')
        gaff_path = os.path.join(output_dir, 'benzene.gaff.mol2')
        frcmod_path = os.path.join(output_dir, 'benzene.frcmod')

        # Get last modified time
        last_touched_gaff = os.stat(gaff_path).st_mtime
        last_touched_frcmod = os.stat(frcmod_path).st_mtime

        # Check that output files have been created
        assert os.path.exists(gaff_path)
        assert os.path.exists(frcmod_path)
        assert os.path.getsize(gaff_path) > 0
        assert os.path.getsize(frcmod_path) > 0

        # Check that setup_molecules do not recreate molecule files
        time.sleep(0.5)  # st_mtime doesn't have much precision
        exp_builder._db._setup_molecules('benzene')
        assert last_touched_gaff == os.stat(gaff_path).st_mtime
        assert last_touched_frcmod == os.stat(frcmod_path).st_mtime


@unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
def test_setup_name_smiles_openeye_charges():
    """Setup molecule from name and SMILES with openeye charges and gaff."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        molecules_ids = ['toluene-smiles', 'p-xylene-name']
        yaml_content = get_template_script(tmp_dir, keep_openeye=True)
        exp_builder = ExperimentBuilder(yaml_content)
        exp_builder._db._setup_molecules(*molecules_ids)

        for mol in molecules_ids:
            output_dir = exp_builder._db.get_molecule_dir(mol)
            output_basepath = os.path.join(output_dir, mol)

            # Check that all the files have been created
            assert os.path.exists(output_basepath + '.mol2')
            assert os.path.exists(output_basepath + '.gaff.mol2')
            assert os.path.exists(output_basepath + '.frcmod')
            assert os.path.getsize(output_basepath + '.mol2') > 0
            assert os.path.getsize(output_basepath + '.gaff.mol2') > 0
            assert os.path.getsize(output_basepath + '.frcmod') > 0

            atoms_frame, _ = mdtraj.formats.mol2.mol2_to_dataframes(output_basepath + '.mol2')
            input_charges = atoms_frame['charge']
            atoms_frame, _ = mdtraj.formats.mol2.mol2_to_dataframes(output_basepath + '.gaff.mol2')
            output_charges = atoms_frame['charge']

            # With openeye:am1bcc charges, the final charges should be unaltered
            if mol == 'p-xylene-name':
                assert input_charges.equals(output_charges)
            else:  # With antechamber, sqm should alter the charges a little
                assert not input_charges.equals(output_charges)

        # Check that molecules are resumed correctly
        exp_builder = ExperimentBuilder(yaml_content)
        exp_builder._db._setup_molecules(*molecules_ids)


@unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
def test_clashing_atoms():
    """Check that clashing atoms are resolved."""
    benzene_path = examples_paths()['benzene']
    toluene_path = examples_paths()['toluene']
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_content = get_template_script(tmp_dir, keep_openeye=True)
        system_id = 'explicit-system'
        system_description = yaml_content['systems'][system_id]
        system_description['pack'] = True
        system_description['solvent'] = utils.CombinatorialLeaf(['vacuum', 'PME'])

        # Sanity check: at the beginning molecules clash
        toluene_pos = utils.get_oe_mol_positions(utils.load_oe_molecules(toluene_path, molecule_idx=0))
        benzene_pos = utils.get_oe_mol_positions(utils.load_oe_molecules(benzene_path, molecule_idx=0))
        assert pipeline.compute_min_dist(toluene_pos, benzene_pos) < pipeline.SetupDatabase.CLASH_THRESHOLD

        exp_builder = ExperimentBuilder(yaml_content)

        for sys_id in [system_id + '_vacuum', system_id + '_PME']:
            system_dir = os.path.dirname(
                exp_builder._db.get_system(sys_id)[0].position_path)

            # Get positions of molecules in the final system
            prmtop = openmm.app.AmberPrmtopFile(os.path.join(system_dir, 'complex.prmtop'))
            inpcrd = openmm.app.AmberInpcrdFile(os.path.join(system_dir, 'complex.inpcrd'))
            positions = inpcrd.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            topography = Topography(prmtop.topology, ligand_atoms='resname TOL')
            benzene_pos2 = positions.take(topography.receptor_atoms, axis=0)
            toluene_pos2 = positions.take(topography.ligand_atoms, axis=0)
            # atom_indices = pipeline.find_components(prmtop.createSystem(), prmtop.topology, 'resname TOL')
            # benzene_pos2 = positions.take(atom_indices['receptor'], axis=0)
            # toluene_pos2 = positions.take(atom_indices['ligand'], axis=0)

            # Test that clashes are resolved in the system
            min_dist, max_dist = pipeline.compute_min_max_dist(toluene_pos2, benzene_pos2)
            assert min_dist >= pipeline.SetupDatabase.CLASH_THRESHOLD

            # For solvent we check that molecule is within the box
            if sys_id == system_id + '_PME':
                assert max_dist <= exp_builder._db.solvents['PME']['clearance'].value_in_unit(unit.angstrom)


@unittest.skipIf(not moltools.schrodinger.is_schrodinger_suite_installed(),
                 "This test requires Schrodinger's suite")
def test_epik_enumeration():
    """Test epik protonation state enumeration."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_content = get_template_script(tmp_dir, keep_schrodinger=True)
        exp_builder = ExperimentBuilder(yaml_content)
        mol_ids = ['benzene-epik0', 'benzene-epikcustom']
        exp_builder._db._setup_molecules(*mol_ids)

        for mol_id in mol_ids:
            output_dir = exp_builder._db.get_molecule_dir(mol_id)
            output_basename = os.path.join(output_dir, mol_id + '-epik.')
            assert os.path.exists(output_basename + 'mol2')
            assert os.path.getsize(output_basename + 'mol2') > 0
            assert os.path.exists(output_basename + 'sdf')
            assert os.path.getsize(output_basename + 'sdf') > 0


def setup_molecule_output_check(exp_builder_db, mol_id, output_path):
    """
    Helper function to check molecules which have to go through the setup pipeline
    Accepts the experiment builder database, the mol_id, and the output_path
    Tries to setup the given mol_id and makes sure the output exists and is non-zero
    """
    exp_builder_db._setup_molecules(mol_id)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_strip_protons():
    """Test that protons are stripped correctly for tleap."""
    mol_id = 'Abl'
    abl_path = examples_paths()['abl']
    with mmtools.utils.temporary_directory() as tmp_dir:
        # Safety check: protein must have protons
        has_hydrogen = False
        with open(abl_path, 'r') as f:
            for line in f:
                if line[:6] == 'ATOM  ' and (line[12] == 'H' or line[13] == 'H'):
                    has_hydrogen = True
                    break
        assert has_hydrogen

        yaml_content = get_template_script(tmp_dir)
        exp_builder = ExperimentBuilder(yaml_content)
        output_dir = exp_builder._db.get_molecule_dir(mol_id)
        output_path = os.path.join(output_dir, 'Abl.pdb')

        # We haven't set the strip_protons options, so this shouldn't do anything
        exp_builder._db._setup_molecules(mol_id)
        assert not os.path.exists(output_path)

        # Now we set the strip_protons options and repeat
        exp_builder._db.molecules[mol_id]['strip_protons'] = True
        setup_molecule_output_check(exp_builder._db, mol_id, output_path)

        # The new pdb does not have hydrogen atoms
        has_hydrogen = False
        with open(output_path, 'r') as f:
            for line in f:
                if line[:6] == 'ATOM  ' and (line[12] == 'H' or line[13] == 'H'):
                    has_hydrogen = True
                    break
        assert not has_hydrogen


def test_pdbfixer_mutations():
    """Test that pdbfixer can apply mutations correctly."""
    mol_id = 'Abl'
    abl_path = examples_paths()['abl']
    with mmtools.utils.temporary_directory() as tmp_dir:
        # Safety check: protein must have WT residue: THR at residue 85 in chain A
        has_wt_residue = False
        with open(abl_path, 'r') as f:
            for line in f:
                if (line[:6] == 'ATOM  ') and (line[21] == 'A') and (int(line[22:26]) == 85) and (line[17:20]=='THR'):
                    has_wt_residue = True
                    break
        assert has_wt_residue

        yaml_content = get_template_script(tmp_dir)
        exp_builder = ExperimentBuilder(yaml_content)
        output_dir = exp_builder._db.get_molecule_dir(mol_id)
        output_path = os.path.join(output_dir, 'Abl.pdb')

        # We haven't set the strip_protons options, so this shouldn't do anything
        exp_builder._db._setup_molecules(mol_id)
        assert not os.path.exists(output_path)

        # Now we set the strip_protons options and repeat
        exp_builder._db.molecules[mol_id]['pdbfixer'] = {
            'apply_mutations' : {
                'chain_id' : 'A',
                'mutations': 'T85I',
            }
        }
        setup_molecule_output_check(exp_builder._db, mol_id, output_path)

        # Safety check: protein must have mutated residue: ILE at residue 85 in chain A
        has_mut_residue = False
        with open(output_path, 'r') as f:
            for line in f:
                if (line[:6] == 'ATOM  ') and (line[21] == 'A') and (int(line[22:26]) == 85) and (line[17:20]=='ILE'):
                    has_mut_residue = True
                    break
        assert has_mut_residue


@unittest.skipIf(not utils.is_modeller_installed(), "This test requires Salilab Modeller")
def test_modeller_mutations():
    """Test that modeller can apply mutations correctly."""
    mol_id = 'Abl'
    abl_path = examples_paths()['abl']
    with mmtools.utils.temporary_directory() as tmp_dir:
        # Safety check: protein must have WT residue: THR at residue 85 in chain A
        has_wt_residue = False
        with open(abl_path, 'r') as f:
            for line in f:
                if (line[:6] == 'ATOM  ') and (line[21] == 'A') and (int(line[22:26]) == 85) and (line[17:20]=='THR'):
                    has_wt_residue = True
                    break
        assert has_wt_residue

        yaml_content = get_template_script(tmp_dir)
        exp_builder = ExperimentBuilder(yaml_content)
        output_dir = exp_builder._db.get_molecule_dir(mol_id)
        output_path = os.path.join(output_dir, 'Abl.pdb')

        # We haven't set the strip_protons options, so this shouldn't do anything
        exp_builder._db._setup_molecules(mol_id)
        assert not os.path.exists(output_path)

        # Calling modeller with WT creates a file (although the protein is not mutated).
        exp_builder._db.molecules[mol_id]['modeller'] = {
            'apply_mutations': {
                'chain_id': 'A',
                'mutations': 'WT',
            }
        }
        setup_molecule_output_check(exp_builder._db, mol_id, output_path)
        os.remove(output_path)  # Remove file for next check.


        # Reinitialize exp_builder
        exp_builder = ExperimentBuilder(yaml_content)

        # Now we set the strip_protons options and repeat for the mutant case
        exp_builder._db.molecules[mol_id]['modeller'] = {
            'apply_mutations': {
                'chain_id': 'A',
                'mutations': 'T85I',
            }
        }
        setup_molecule_output_check(exp_builder._db, mol_id, output_path)

        # Safety check: protein must have mutated residue: ILE at residue 85 in chain A
        has_mut_residue = False
        with open(output_path, 'r') as f:
            for line in f:
                if (line[:6] == 'ATOM  ') and (line[21] == 'A') and (int(line[22:26]) == 85) and (line[17:20]=='ILE'):
                    has_mut_residue = True
                    break
        assert has_mut_residue

def test_pdbfixer_processing():
    """Test that PDB fixer correctly parses and sets up the molecules"""
    mol_id = 'Abl'
    pdb_fixer_modifications = [
        {'pdbfixer': {}},
        {'pdbfixer': {'add_missing_residues': True}},
        {'pdbfixer': {'add_missing_atoms': 'all', 'ph': '8.0'}},
        {'pdbfixer': {'remove_heterogens': 'all'}},
        {'pdbfixer': {'replace_nonstandard_residues': True}},
        {'pdbfixer': {'apply_mutations': {'chain_id': 'A', 'mutations': 'T85I'}}},
        {'pdbfixer': {'apply_mutations': {'chain_id': 'A', 'mutations': 'I8A/T9A'}}},
    ]
    for mod in pdb_fixer_modifications:
        with mmtools.utils.temporary_directory() as tmp_dir:
            yaml_content = get_template_script(tmp_dir)
            exp_builder = ExperimentBuilder(yaml_content)
            output_dir = exp_builder._db.get_molecule_dir(mol_id)
            output_path = os.path.join(output_dir, 'Abl.pdb')
            exp_builder._db.molecules[mol_id].update(mod)
            yield setup_molecule_output_check, exp_builder._db, mol_id, output_path


# ==============================================================================
# Combinatorial expansion
# ==============================================================================

class TestMultiMoleculeFiles(object):

    @classmethod
    def setup_class(cls):
        """Create a 2-frame PDB file in pdb_path. The second frame has same positions
        of the first one but with inversed z-coordinate."""
        # Creating a temporary directory and generating paths for output files
        cls.tmp_dir = tempfile.mkdtemp()
        cls.pdb_path = os.path.join(cls.tmp_dir, 'multipdb.pdb')
        cls.smiles_path = os.path.join(cls.tmp_dir, 'multismiles.smiles')
        cls.sdf_path = os.path.join(cls.tmp_dir, 'multisdf.sdf')
        cls.mol2_path = os.path.join(cls.tmp_dir, 'multimol2.mol2')

        # Rotation matrix to invert z-coordinate, i.e. flip molecule w.r.t. x-y plane
        rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        # Create 2-frame PDB file. First frame is lysozyme, second is lysozyme with inverted z
        lysozyme_path = examples_paths()['lysozyme']
        lysozyme = PDBFile(lysozyme_path)

        # Rotate positions to invert z for the second frame
        symmetric_pos = lysozyme.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        symmetric_pos = symmetric_pos.dot(rot) * unit.angstrom

        with open(cls.pdb_path, 'w') as f:
            PDBFile.writeHeader(lysozyme.topology, file=f)
            PDBFile.writeModel(lysozyme.topology, lysozyme.positions, file=f, modelIndex=0)
            PDBFile.writeModel(lysozyme.topology, symmetric_pos, file=f, modelIndex=1)

        # Create 2-molecule SMILES file
        with open(cls.smiles_path, 'w') as f:
            f.write('# comment\n')
            f.write('benzene,c1ccccc1\n')
            f.write('toluene,Cc1ccccc1\n')

        # Create 2-molecule sdf and mol2 with OpenEye
        if utils.is_openeye_installed():
            from openeye import oechem
            oe_benzene = utils.load_oe_molecules(examples_paths()['benzene'], molecule_idx=0)
            oe_benzene_pos = utils.get_oe_mol_positions(oe_benzene).dot(rot)
            oe_benzene.NewConf(oechem.OEFloatArray(oe_benzene_pos.flatten()))

            # Save 2-conformer benzene in sdf and mol2 format
            utils.write_oe_molecule(oe_benzene, cls.sdf_path)
            utils.write_oe_molecule(oe_benzene, cls.mol2_path, mol2_resname='MOL')

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmp_dir)

    @unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
    def test_expand_molecules(self):
        """Check that combinatorial molecules are handled correctly."""
        yaml_content = """
        ---
        molecules:
            rec:
                filepath: !Combinatorial [{}, {}]
                leap: {{parameters: oldff/leaprc.ff14SB}}
            lig:
                name: !Combinatorial [iupac1, iupac2]
                leap: {{parameters: leaprc.gaff}}
                epik:
                    select: !Combinatorial [0, 2]
            multi:
                filepath: {}
                leap: {{parameters: oldff/leaprc.ff14SB}}
                select: all
            smiles:
                filepath: {}
                leap: {{parameters: leaprc.gaff}}
                select: all
            sdf:
                filepath: {}
                leap: {{parameters: leaprc.gaff}}
                select: all
            mol2:
                filepath: {}
                leap: {{parameters: leaprc.gaff}}
                select: all
        solvents:
            solv1:
                nonbonded_method: NoCutoff
            solv2:
                nonbonded_method: PME
                nonbonded_cutoff: 1*nanometer
                clearance: 10*angstroms
        protocols:{}
        systems:
            sys:
                receptor: !Combinatorial [rec, multi]
                ligand: lig
                solvent: !Combinatorial [solv1, solv2]
        experiments:
            system: sys
            protocol: absolute-binding
        """.format(self.sdf_path, self.mol2_path, self.pdb_path,
                   self.smiles_path, self.sdf_path, self.mol2_path,
                   indent(indent(standard_protocol)))
        yaml_content = textwrap.dedent(yaml_content)

        expected_content = """
        ---
        molecules:
            rec_multisdf:
                filepath: {}
                leap: {{parameters: oldff/leaprc.ff14SB}}
            rec_multimol2:
                filepath: {}
                leap: {{parameters: oldff/leaprc.ff14SB}}
            lig_0_iupac1:
                name: iupac1
                leap: {{parameters: leaprc.gaff}}
                epik: {{select: 0}}
            lig_2_iupac1:
                name: iupac1
                leap: {{parameters: leaprc.gaff}}
                epik: {{select: 2}}
            lig_0_iupac2:
                name: iupac2
                leap: {{parameters: leaprc.gaff}}
                epik: {{select: 0}}
            lig_2_iupac2:
                name: iupac2
                leap: {{parameters: leaprc.gaff}}
                epik: {{select: 2}}
            multi_0:
                filepath: {}
                leap: {{parameters: oldff/leaprc.ff14SB}}
                select: 0
            multi_1:
                filepath: {}
                leap: {{parameters: oldff/leaprc.ff14SB}}
                select: 1
            smiles_0:
                filepath: {}
                leap: {{parameters: leaprc.gaff}}
                select: 0
            smiles_1:
                filepath: {}
                leap: {{parameters: leaprc.gaff}}
                select: 1
            sdf_0:
                filepath: {}
                leap: {{parameters: leaprc.gaff}}
                select: 0
            sdf_1:
                filepath: {}
                leap: {{parameters: leaprc.gaff}}
                select: 1
            mol2_0:
                filepath: {}
                leap: {{parameters: leaprc.gaff}}
                select: 0
            mol2_1:
                filepath: {}
                leap: {{parameters: leaprc.gaff}}
                select: 1
        solvents:
            solv1:
                nonbonded_method: NoCutoff
            solv2:
                nonbonded_method: PME
                nonbonded_cutoff: 1*nanometer
                clearance: 10*angstroms
        protocols:{}
        systems:
            sys:
                receptor: !Combinatorial [rec_multimol2, rec_multisdf, multi_0, multi_1]
                ligand: !Combinatorial [lig_0_iupac1, lig_0_iupac2, lig_2_iupac1, lig_2_iupac2]
                solvent: !Combinatorial [solv1, solv2]
        experiments:
            system: sys
            protocol: absolute-binding
        """.format(self.sdf_path, self.mol2_path, self.pdb_path, self.pdb_path,
                   self.smiles_path, self.smiles_path, self.sdf_path, self.sdf_path,
                   self.mol2_path, self.mol2_path, indent(standard_protocol))
        expected_content = textwrap.dedent(expected_content)

        raw = yank_load(yaml_content)
        expanded = ExperimentBuilder(yaml_content)._expand_molecules(raw)
        expected = yank_load(expected_content)
        assert expanded == expected, 'Expected:\n{}\n\nExpanded:\n{}'.format(
            expected['systems'], expanded['systems'])

    def test_select_pdb_conformation(self):
        """Check that frame selection in multi-model PDB files works."""
        with mmtools.utils.temporary_directory() as tmp_dir:
            yaml_content = """
            ---
            options:
                output_dir: {}
                setup_dir: .
            molecules:
                selected:
                    filepath: {}
                    leap: {{parameters: oldff/leaprc.ff14SB}}
                    select: 1
            """.format(tmp_dir, self.pdb_path)
            yaml_content = textwrap.dedent(yaml_content)
            exp_builder = ExperimentBuilder(yaml_content)

            # The molecule now is neither set up nor processed
            is_setup, is_processed = exp_builder._db.is_molecule_setup('selected')
            assert is_setup is False
            assert is_processed is False

            # The setup of the molecule must isolate the frame in a single-frame PDB
            exp_builder._db._setup_molecules('selected')
            selected_pdb_path = os.path.join(tmp_dir, pipeline.SetupDatabase.MOLECULES_DIR,
                                             'selected', 'selected.pdb')
            assert os.path.exists(os.path.join(selected_pdb_path))
            assert os.path.getsize(os.path.join(selected_pdb_path)) > 0

            # The positions must be the ones of the second frame
            selected_pdb = PDBFile(selected_pdb_path)
            selected_pos = selected_pdb.getPositions(asNumpy=True)
            second_pos = PDBFile(self.pdb_path).getPositions(asNumpy=True, frame=1)
            assert selected_pdb.getNumFrames() == 1
            assert (selected_pos == second_pos).all()

            # The description of the molecule is now updated
            assert os.path.normpath(exp_builder._db.molecules['selected']['filepath']) == selected_pdb_path

            # The molecule now both set up and processed
            is_setup, is_processed = exp_builder._db.is_molecule_setup('selected')
            assert is_setup is True
            assert is_processed is True

            # A new instance of ExperimentBuilder is able to resume with correct molecule
            exp_builder = ExperimentBuilder(yaml_content)
            is_setup, is_processed = exp_builder._db.is_molecule_setup('selected')
            assert is_setup is True
            assert is_processed is True

    @unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
    def test_setup_smiles(self):
        """Check that setup molecule from SMILES files works."""
        from openeye.oechem import OEMolToSmiles

        with mmtools.utils.temporary_directory() as tmp_dir:
            yaml_content = """
            ---
            options:
                output_dir: {}
                setup_dir: .
            molecules:
                take-first:
                    filepath: {}
                    antechamber: {{charge_method: bcc}}
                    leap: {{parameters: leaprc.gaff}}
                select-second:
                    filepath: {}
                    antechamber: {{charge_method: bcc}}
                    leap: {{parameters: leaprc.gaff}}
                    select: 1
            """.format(tmp_dir, self.smiles_path, self.smiles_path)
            yaml_content = textwrap.dedent(yaml_content)
            exp_builder = ExperimentBuilder(yaml_content)

            for i, mol_id in enumerate(['take-first', 'select-second']):
                # The molecule now is neither set up nor processed
                is_setup, is_processed = exp_builder._db.is_molecule_setup(mol_id)
                assert is_setup is False
                assert is_processed is False

                # The single SMILES has been converted to mol2 file
                exp_builder._db._setup_molecules(mol_id)
                mol2_path = os.path.join(tmp_dir, pipeline.SetupDatabase.MOLECULES_DIR, mol_id, mol_id + '.mol2')
                assert os.path.exists(os.path.join(mol2_path))
                assert os.path.getsize(os.path.join(mol2_path)) > 0

                # The mol2 represents the right molecule
                csv_smiles_str = pipeline.read_csv_lines(self.smiles_path, lines=i).strip().split(',')[1]
                mol2_smiles_str = OEMolToSmiles(utils.load_oe_molecules(mol2_path, molecule_idx=0))
                assert mol2_smiles_str == csv_smiles_str

                # The molecule now both set up and processed
                is_setup, is_processed = exp_builder._db.is_molecule_setup(mol_id)
                assert is_setup is True
                assert is_processed is True

                # A new instance of ExperimentBuilder is able to resume with correct molecule
                exp_builder = ExperimentBuilder(yaml_content)
                is_setup, is_processed = exp_builder._db.is_molecule_setup(mol_id)
                assert is_setup is True
                assert is_processed is True

    @unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
    def test_select_sdf_mol2(self):
        """Check that selection in sdf and mol2 files works."""
        with mmtools.utils.temporary_directory() as tmp_dir:
            yaml_content = """
            ---
            options:
                output_dir: {}
                setup_dir: .
            molecules:
                sdf_0:
                    filepath: {}
                    antechamber: {{charge_method: bcc}}
                    leap: {{parameters: leaprc.gaff}}
                    select: 0
                sdf_1:
                    filepath: {}
                    antechamber: {{charge_method: bcc}}
                    leap: {{parameters: leaprc.gaff}}
                    select: 1
                mol2_0:
                    filepath: {}
                    antechamber: {{charge_method: bcc}}
                    leap: {{parameters: leaprc.gaff}}
                    select: 0
                mol2_1:
                    filepath: {}
                    antechamber: {{charge_method: bcc}}
                    leap: {{parameters: leaprc.gaff}}
                    select: 1
            """.format(tmp_dir, self.sdf_path, self.sdf_path, self.mol2_path, self.mol2_path)
            yaml_content = textwrap.dedent(yaml_content)
            exp_builder = ExperimentBuilder(yaml_content)

            for extension in ['sdf', 'mol2']:
                multi_path = getattr(self, extension + '_path')
                for model_idx in [0, 1]:
                    mol_id = extension + '_' + str(model_idx)

                    # The molecule now is neither set up nor processed
                    is_setup, is_processed = exp_builder._db.is_molecule_setup(mol_id)
                    assert is_setup is False
                    assert is_processed is False

                    exp_builder._db._setup_molecules(mol_id)

                    # The setup of the molecule must isolate the frame in a single-frame PDB
                    single_mol_path = os.path.join(tmp_dir, pipeline.SetupDatabase.MOLECULES_DIR,
                                                   mol_id, mol_id + '.' + extension)
                    assert os.path.exists(os.path.join(single_mol_path))
                    assert os.path.getsize(os.path.join(single_mol_path)) > 0
                    if extension == 'mol2':
                        # OpenEye loses the resname when writing a mol2 file.
                        mol2_file = utils.Mol2File(single_mol_path)
                        assert len(list(mol2_file.resnames)) == 1
                        assert mol2_file.resname != '<0>'

                    # sdf files must be converted to mol2 to be fed to antechamber
                    if extension == 'sdf':
                        single_mol_path = os.path.join(tmp_dir, pipeline.SetupDatabase.MOLECULES_DIR,
                                                       mol_id, mol_id + '.mol2')
                        assert os.path.exists(os.path.join(single_mol_path))
                        assert os.path.getsize(os.path.join(single_mol_path)) > 0

                    # Check antechamber parametrization
                    single_mol_path = os.path.join(tmp_dir, pipeline.SetupDatabase.MOLECULES_DIR,
                                                   mol_id, mol_id + '.gaff.mol2')
                    assert os.path.exists(os.path.join(single_mol_path))
                    assert os.path.getsize(os.path.join(single_mol_path)) > 0

                    # The positions must be approximately correct (antechamber move the molecule)
                    selected_oe_mol = utils.load_oe_molecules(single_mol_path, molecule_idx=0)
                    selected_pos = utils.get_oe_mol_positions(selected_oe_mol)
                    second_oe_mol = utils.load_oe_molecules(multi_path, molecule_idx=model_idx)
                    second_pos = utils.get_oe_mol_positions(second_oe_mol)
                    assert selected_oe_mol.NumConfs() == 1
                    assert np.allclose(selected_pos, second_pos, atol=1e-1)

                    # The molecule now both set up and processed
                    is_setup, is_processed = exp_builder._db.is_molecule_setup(mol_id)
                    assert is_setup is True
                    assert is_processed is True

                    # A new instance of ExperimentBuilder is able to resume with correct molecule
                    exp_builder = ExperimentBuilder(yaml_content)
                    is_setup, is_processed = exp_builder._db.is_molecule_setup(mol_id)
                    assert is_setup is True
                    assert is_processed is True


def test_system_expansion():
    """Combinatorial systems are correctly expanded."""
    # We need 2 combinatorial systems
    template_script = get_template_script()
    template_system = template_script['systems']['implicit-system']
    del template_system['leap']
    template_script['systems'] = {'system1': template_system.copy(),
                                  'system2': template_system.copy()}
    template_script['systems']['system1']['receptor'] = utils.CombinatorialLeaf(['Abl', 'T4Lysozyme'])
    template_script['systems']['system2']['ligand'] = utils.CombinatorialLeaf(['p-xylene', 'toluene'])
    template_script['experiments']['system'] = utils.CombinatorialLeaf(['system1', 'system2'])

    # Expected expanded script
    expected_script = yank_load("""
    systems:
        system1_Abl: {receptor: Abl, ligand: p-xylene, solvent: GBSA-OBC2}
        system1_T4Lysozyme: {receptor: T4Lysozyme, ligand: p-xylene, solvent: GBSA-OBC2}
        system2_pxylene: {receptor: T4Lysozyme, ligand: p-xylene, solvent: GBSA-OBC2}
        system2_toluene: {receptor: T4Lysozyme, ligand: toluene, solvent: GBSA-OBC2}
    experiments:
        system: !Combinatorial ['system1_Abl', 'system1_T4Lysozyme', 'system2_pxylene', 'system2_toluene']
        protocol: absolute-binding
    """)
    expanded_script = template_script.copy()
    expanded_script['systems'] = expected_script['systems']
    expanded_script['experiments'] = expected_script['experiments']

    assert ExperimentBuilder(template_script)._expand_systems(template_script) == expanded_script


def test_exp_sequence():
    """Test all experiments in a sequence are parsed."""
    yaml_content = """
    ---
    molecules:
        rec:
            filepath: {}
            leap: {{parameters: oldff/leaprc.ff14SB}}
        lig:
            name: lig
            leap: {{parameters: leaprc.gaff}}
    solvents:
        solv1:
            nonbonded_method: NoCutoff
        solv2:
            nonbonded_method: PME
            nonbonded_cutoff: 1*nanometer
            clearance: 10*angstroms
    protocols:{}
    systems:
        system1:
            receptor: rec
            ligand: lig
            solvent: !Combinatorial [solv1, solv2]
        system2:
            receptor: rec
            ligand: lig
            solvent: solv1
    experiment1:
        system: system1
        protocol: absolute-binding
    experiment2:
        system: system2
        protocol: absolute-binding
    experiments: [experiment1, experiment2]
    """.format(examples_paths()['lysozyme'], standard_protocol)
    exp_builder = ExperimentBuilder(textwrap.dedent(yaml_content))
    assert len(exp_builder._experiments) == 2


# ==============================================================================
# Systems pipeline
# ==============================================================================

def test_setup_implicit_system_leap():
    """Create prmtop and inpcrd for implicit solvent protein-ligand system."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_content = get_template_script(tmp_dir)
        exp_builder = ExperimentBuilder(yaml_content)

        output_dir = os.path.dirname(
            exp_builder._db.get_system('implicit-system')[0].position_path)
        last_modified_path = os.path.join(output_dir, 'complex.prmtop')
        last_modified = os.stat(last_modified_path).st_mtime

        # Test that output files exist and there is no water
        for phase in ['complex', 'solvent']:
            found_resnames = set()
            pdb_path = os.path.join(output_dir, phase + '.pdb')
            prmtop_path = os.path.join(output_dir, phase + '.prmtop')
            inpcrd_path = os.path.join(output_dir, phase + '.inpcrd')

            with open(pdb_path, 'r') as f:
                for line in f:
                    if len(line) > 10 and line[:5] != 'CRYST':
                        found_resnames.add(line[17:20])

            assert os.path.exists(prmtop_path)
            assert os.path.exists(inpcrd_path)
            assert os.path.getsize(prmtop_path) > 0
            assert os.path.getsize(inpcrd_path) > 0
            assert 'MOL' in found_resnames
            assert 'WAT' not in found_resnames

        # Test that another call do not regenerate the system
        time.sleep(0.5)  # st_mtime doesn't have much precision
        exp_builder._db.get_system('implicit-system')
        assert last_modified == os.stat(last_modified_path).st_mtime


def test_setup_explicit_system_leap():
    """Create prmtop and inpcrd protein-ligand system in explicit solvent."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_content = get_template_script(tmp_dir)
        exp_builder = ExperimentBuilder(yaml_content)

        output_dir = os.path.dirname(
            exp_builder._db.get_system('explicit-system')[0].position_path)

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
                    if len(line) > 10 and line[:5] != 'CRYST':
                        found_resnames .add(line[17:20])

            assert os.path.exists(prmtop_path)
            assert os.path.exists(inpcrd_path)
            assert os.path.getsize(prmtop_path) > 0
            assert os.path.getsize(inpcrd_path) > 0
            assert found_resnames == expected_resnames[phase]


def test_neutralize_system():
    """Test whether the system charge is neutralized correctly."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_content = get_template_script(tmp_dir)
        yaml_content['systems']['explicit-system']['receptor'] = 'T4Lysozyme'
        yaml_content['systems']['explicit-system']['ligand'] = 'p-xylene'
        exp_builder = ExperimentBuilder(yaml_content)

        output_dir = os.path.dirname(
            exp_builder._db.get_system('explicit-system')[0].position_path)

        # Test that output file exists and that there are ions
        found_resnames = set()
        with open(os.path.join(output_dir, 'complex.pdb'), 'r') as f:
            for line in f:
                if len(line) > 10 and line[:5] != 'CRYST':
                    found_resnames.add(line[17:20])
        assert set(['MOL', 'WAT', 'Cl-']) <= found_resnames

        # Check that parameter files exist
        prmtop_path = os.path.join(output_dir, 'complex.prmtop')
        inpcrd_path = os.path.join(output_dir, 'complex.inpcrd')
        assert os.path.exists(prmtop_path)
        assert os.path.exists(inpcrd_path)


def get_number_of_ions(exp_builder, phase, system_id):
    """Return number of ions in the system.

    Parameters
    ----------
    exp_builder : ExperimentBuilder
        The experiment builder.
    phase : str
        complex, solvent, solvent1, solvent2.
    system_id : str
        The ID of the system.

    Returns
    -------
    n_pos_ions : int
        Number of positive ions in the system.
    n_neg_ions : int
        Number of negative ions in the system.
    n_ionic_strength_ions : int
        Expected number of ions needed to reach the desired ionic strength.

    """
    # Read in output pdb file to read ionic strength.
    if phase == 'complex' or phase == 'solvent1':
        phase_id = 0
    else:
        phase_id = 1
    system_filepath = exp_builder._db.get_system_files_paths(system_id)[phase_id].position_path
    system_filepath = os.path.splitext(system_filepath)[0] + '.pdb'
    system_traj = mdtraj.load(system_filepath)

    # Count number of waters and ions.
    n_waters = 0
    n_pos_ions = 0
    n_neg_ions = 0
    for res in system_traj.topology.residues:
        if res.is_water:
            n_waters += 1
        elif '+' in res.name:
            n_pos_ions += 1
        elif '-' in res.name:
            n_neg_ions += 1

    # Verify that number of ions roughly models the expected ionic strength.
    try:
        solvent_id = exp_builder._db.systems[system_id]['solvent']
    except KeyError:
        solvent_id = exp_builder._db.systems[system_id][phase]  # solvent1 or solvent2
    ionic_strength = exp_builder._db.solvents[solvent_id]['ionic_strength']
    n_ionic_strength_ions = int(np.round(n_waters * ionic_strength / (55.41*unit.molar)))

    return n_pos_ions, n_neg_ions, n_ionic_strength_ions


@unittest.skipIf(not utils.is_openeye_installed(), "This test requires OpenEye toolkit")
def test_charged_ligand():
    """Check that there are alchemical counterions for charged ligands."""
    imatinib_path = examples_paths()['imatinib']
    with mmtools.utils.temporary_directory() as tmp_dir:
        # receptors = {'Asp': -1, 'Abl': -8}  # receptor name -> net charge
        # Only run `Asp` on CI as Abl can be very slow
        receptors = {'Asp': -1}  # receptor name -> net charge
        solvent_names = ['PME', 'PMEionic']
        updates = yank_load("""
        molecules:
            Asp:
                name: "(3S)-3-amino-4-hydroxy-4-oxo-butanoate"
                openeye: {{quacpac: am1-bcc}}
                antechamber: {{charge_method: null}}
            imatinib:
                filepath: {}
                openeye: {{quacpac: am1-bcc}}
                antechamber: {{charge_method: null}}
        explicit-system:
            receptor: !Combinatorial {}
            ligand: imatinib
            pack: yes
            solvent: !Combinatorial {}
        """.format(imatinib_path, list(receptors.keys()), solvent_names))
        yaml_content = get_template_script(tmp_dir, keep_openeye=True)
        yaml_content['options']['resume_setup'] = True
        yaml_content['molecules'].update(updates['molecules'])
        yaml_content['solvents']['PMEionic'] = copy.deepcopy(yaml_content['solvents']['PME'])
        yaml_content['solvents']['PMEionic']['ionic_strength'] = '10*millimolar'
        yaml_content['systems']['explicit-system'].update(updates['explicit-system'])
        exp_builder = ExperimentBuilder(yaml_content)

        for receptor in receptors:

            for solvent_name in solvent_names:
                system_id = 'explicit-system_{}_{}'.format(receptor, solvent_name)
                system_files_paths = exp_builder._db.get_system(system_id)
                for i, phase_name in enumerate(['complex', 'solvent']):
                    inpcrd_file_path = system_files_paths[i].position_path
                    prmtop_file_path = system_files_paths[i].parameters_path

                    system, topology, _ = pipeline.read_system_files(
                        inpcrd_file_path, prmtop_file_path, {'nonbondedMethod': openmm.app.PME})

                    # Identify components.
                    if phase_name == 'complex':
                        alchemical_region = 'ligand_atoms'
                        topography = Topography(topology, ligand_atoms='resname MOL')

                        # Safety check: receptor must be negatively charged as expected
                        receptor_net_charge = pipeline.compute_net_charge(system,
                                                                          topography.receptor_atoms)
                        assert receptor_net_charge == receptors[receptor]
                    else:
                        alchemical_region = 'solute_atoms'
                        topography = Topography(topology)

                    # There is a single ligand/solute counterion.
                    ligand_counterions = pipeline.find_alchemical_counterions(system, topography,
                                                                              alchemical_region)
                    assert len(ligand_counterions) == 1
                    ion_idx = ligand_counterions[0]
                    ion_atom = next(itertools.islice(topology.atoms(), ion_idx, None))
                    assert '-' in ion_atom.residue.name

                    # In complex, there should be both ions even if the system is globally
                    # neutral (e.g. asp lys system), because of the alchemical ion.
                    n_pos_ions, n_neg_ions, n_ionic_strength_ions = get_number_of_ions(
                        exp_builder, phase=phase_name, system_id=system_id)
                    print(system_id, phase_name, n_ionic_strength_ions, n_pos_ions, n_neg_ions)

                    # Check correct number of ions.
                    if phase_name == 'complex':
                        n_neutralization_ions = abs(receptors[receptor])
                        if n_ionic_strength_ions > 0:
                            n_neutralization_ions -= 1  # we don't add an extra anion to alchemically modify
                        assert n_pos_ions == n_neutralization_ions + n_ionic_strength_ions
                        assert n_neg_ions == max(1, n_ionic_strength_ions)
                    else:
                        assert n_pos_ions == n_ionic_strength_ions
                        assert n_neg_ions == 1 + n_ionic_strength_ions


def test_ionic_strength():
    """The correct number of ions is added to achieve the requested ionic strength."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_script = get_template_script(tmp_dir)
        assert yaml_script['systems']['hydration-system']['solute'] == 'toluene'
        assert yaml_script['systems']['hydration-system']['solvent1'] == 'PME'

        # Set ionic strength.
        yaml_script['solvents']['PME']['ionic_strength'] = '200*millimolar'

        # Set up toluene in explicit solvent.
        exp_builder = ExperimentBuilder(yaml_script)
        exp_builder._db.get_system('hydration-system')

        n_pos_ions, n_neg_ions, expected_n_ions = get_number_of_ions(exp_builder, phase='solvent1',
                                                                     system_id='hydration-system')

        assert expected_n_ions > 0  # Otherwise this test doesn't make sense.
        assert n_pos_ions == expected_n_ions, '{}, {}'.format(n_pos_ions, expected_n_ions)
        assert n_neg_ions == expected_n_ions, '{}, {}'.format(n_neg_ions, expected_n_ions)


def test_setup_explicit_solvation_system():
    """Create prmtop and inpcrd files for solvation free energy in explicit solvent."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_script = get_template_script(tmp_dir)
        del yaml_script['experiments']
        exp_builder = ExperimentBuilder(yaml_script)
        output_dir = os.path.dirname(
            exp_builder._db.get_system('hydration-system')[0].position_path)

        # Test that output file exists and that it has correct components
        expected_resnames = {'solvent1': set(['TOL', 'WAT']), 'solvent2': set(['TOL'])}
        for phase in expected_resnames:
            found_resnames = set()
            pdb_path = os.path.join(output_dir, phase + '.pdb')
            prmtop_path = os.path.join(output_dir, phase + '.prmtop')
            inpcrd_path = os.path.join(output_dir, phase + '.inpcrd')

            with open(pdb_path, 'r') as f:
                for line in f:
                    if len(line) > 10 and line[:5] != 'CRYST':
                        found_resnames.add(line[17:20])

            assert os.path.exists(prmtop_path)
            assert os.path.exists(inpcrd_path)
            assert os.path.getsize(prmtop_path) > 0
            assert os.path.getsize(inpcrd_path) > 0
            assert found_resnames == expected_resnames[phase]


def test_setup_solvent_models():
    """Test the solvation with different solvent models works."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        template_script = get_template_script(tmp_dir)

        # Setup solvation system and reduce clearance to make test faster.
        template_script['systems']['hydration-system']['solvent1'] = 'PME'
        template_script['solvents']['PME']['clearance'] = '3.0 * angstrom'
        del template_script['experiments']

        # Test solvent models.
        for solvent_model in ['tip3p', 'tip4pew', 'tip3pfb', 'tip5p']:
            yaml_script = copy.deepcopy(template_script)
            yaml_script['solvents']['PME']['solvent_model'] = solvent_model
            if solvent_model == 'tip3p' or solvent_model == 'tip4pew':
                solvent_parameters = ['leaprc.water.' + solvent_model]
            else:
                solvent_parameters = ['leaprc.water.tip3p', 'frcmod.' + solvent_model]
            yaml_script['solvents']['PME']['leap']['parameters'] = solvent_parameters
            yaml_script['options']['setup_dir'] = solvent_model
            exp_builder = ExperimentBuilder(yaml_script)

            # Infer number of expected atoms per water molecule from model.
            expected_water_n_atoms = int(list(filter(str.isdigit, solvent_model))[0])

            # Setup the system and check that water residues have expected number of particles.
            prmtop_filepath = exp_builder._db.get_system('hydration-system')[0].parameters_path
            topology = mdtraj.load_prmtop(prmtop_filepath)
            yield assert_equal, topology.residue(1).n_atoms, expected_water_n_atoms


def test_setup_multiple_parameters_system():
    """Set up system with molecule that needs many parameter files."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_script = get_template_script(tmp_dir)

        # Force antechamber parametrization of benzene to output frcmod file
        exp_builder = ExperimentBuilder(yaml_script)
        exp_builder._db._setup_molecules('benzene')
        benzene_dir = exp_builder._db.get_molecule_dir('benzene')
        frcmod_path = os.path.join(benzene_dir, 'benzene.frcmod')
        benzene_path = os.path.join(benzene_dir, 'benzene.gaff.mol2')

        # Redefine benzene to use leaprc.gaff and benzene.frcmod
        # and set up system for hydration free energy calculation
        yaml_script['molecules'] = {
            'benzene-frcmod': {'filepath': benzene_path,
                               'leap': {'parameters': ['leaprc.gaff', frcmod_path]}}}
        yaml_script['systems'] = {
            'system':
                {'solute': 'benzene-frcmod', 'solvent1': 'PME', 'solvent2': 'vacuum',
                 'leap': {'parameters': 'oldff/leaprc.ff14SB'}}
        }
        del yaml_script['experiments']

        exp_builder = ExperimentBuilder(yaml_script)
        system_files_path = exp_builder._db.get_system('system')

        # Check that output exist:
        for phase in system_files_path:
            assert os.path.exists(phase.parameters_path)
            assert os.path.exists(phase.position_path)
            assert os.path.getsize(phase.parameters_path) > 0
            assert os.path.getsize(phase.position_path) > 0


# ==============================================================================
# Platform configuration tests
# ==============================================================================

def test_platform_configuration():
    """Test that the precision for platform is configured correctly."""
    available_platforms = [openmm.Platform.getPlatform(i).getName()
                           for i in range(openmm.Platform.getNumPlatforms())]

    for platform_name in available_platforms:
        exp_builder = ExperimentBuilder(script='options: {}')

        # Reference and CPU platform support only one precision model
        if platform_name == 'Reference':
            assert_raises(RuntimeError, exp_builder._configure_platform, platform_name, 'mixed')
            continue
        elif platform_name == 'CPU':
            assert_raises(RuntimeError, exp_builder._configure_platform, platform_name, 'double')
            continue

        # Check that precision is set as expected
        for precision in ['mixed', 'double', 'single']:
            if platform_name == 'CUDA':
                platform = exp_builder._configure_platform(platform_name=platform_name,
                                                            platform_precision=precision)
                assert platform.getPropertyDefaultValue('CudaPrecision') == precision
                assert platform.getPropertyDefaultValue('DeterministicForces') == 'true'
            elif platform_name == 'OpenCL':
                if ExperimentBuilder._opencl_device_support_precision(precision):
                    platform = exp_builder._configure_platform(platform_name=platform_name,
                                                                platform_precision=precision)
                    assert platform.getPropertyDefaultValue('OpenCLPrecision') == precision
                else:
                    assert_raises(RuntimeError, exp_builder._configure_platform, platform_name, precision)


def test_default_platform_precision():
    """Test that the precision for platform is set to mixed by default."""
    available_platforms = [openmm.Platform.getPlatform(i).getName()
                           for i in range(openmm.Platform.getNumPlatforms())]

    # Determine whether this device OpenCL platform supports double precision
    if 'OpenCL' in available_platforms:
        opencl_support_double = ExperimentBuilder._opencl_device_support_precision('double')

    for platform_name in available_platforms:
        # Reference and CPU platform support only one precision model so we don't
        # explicitly test them. We still call _configure_platform to be sure that
        # precision 'auto' works
        exp_builder = ExperimentBuilder(script='options: {}')
        platform = exp_builder._configure_platform(platform_name=platform_name,
                                                    platform_precision='auto')
        if platform_name == 'CUDA':
            assert platform.getPropertyDefaultValue('CudaPrecision') == 'mixed'
        elif platform_name == 'OpenCL':
            if opencl_support_double:
                assert platform.getPropertyDefaultValue('OpenCLPrecision') == 'mixed'
            else:
                assert platform.getPropertyDefaultValue('OpenCLPrecision') == 'single'


# ==============================================================================
# Experiment building
# ==============================================================================

class TestExperimentBuilding(object):
    """Test that options are passed correctly from YAML to the built objects."""

    @classmethod
    def get_implicit_template_script(cls, output_dir):
        """Return the template script with only an implicit system."""
        template_script = get_template_script(output_dir)

        # Remove systems we don't need to setup.
        del template_script['systems']['explicit-system']
        del template_script['systems']['hydration-system']
        template_script['experiments']['system'] = 'implicit-system'
        return template_script

    def check_constructor(self, yaml_script, constructor_description, object_name,
                          complex_phase_only=False, special_check_func=None):
        exp_builder = ExperimentBuilder(script=yaml_script)
        for experiment in exp_builder.build_experiments():
            phases = experiment.phases
            if complex_phase_only:
                phases = [phases[0]]
            for phase in phases:
                for k, v in constructor_description.items():
                    # Convert constructor strings to quantities if necessary.
                    try:
                        v = utils.quantity_from_string(v)
                    except:
                        pass
                    # Obtain instantiated object.
                    object_instance = phase
                    for name in object_name.split('.'):
                        object_instance = getattr(object_instance, name)
                    # Check class and attributes.
                    if k == 'type':
                        assert object_instance.__class__.__name__ == v
                    else:
                        assert getattr(object_instance, k) == v
                if special_check_func is not None:
                    special_check_func(phase, constructor_description)

    def test_alchemical_phase_factory_building(self):
        """Test that options are passed to AlchemicalPhaseFactory correctly."""
        with mmtools.utils.temporary_directory() as tmp_dir:
            template_script = self.get_implicit_template_script(tmp_dir)

            # AbsoluteAlchemicalFactory options.
            template_script['options']['alchemical_pme_treatment'] = 'exact'

            # Test that options are passed to AlchemicalPhaseFactory correctly.
            exp_builder = ExperimentBuilder(script=template_script)
            for experiment in exp_builder.build_experiments():
                for phase_factory in experiment.phases:
                    assert phase_factory.alchemical_factory.alchemical_pme_treatment == 'exact'
                    # Overwrite AbsoluteAlchemicalFactory default for disable_alchemical_dispersion_correction.
                    assert phase_factory.alchemical_factory.disable_alchemical_dispersion_correction == True

    def test_restraint_building(self):
        """Test that experiment restraints are built correctly."""
        with mmtools.utils.temporary_directory() as tmp_dir:
            template_script = self.get_implicit_template_script(tmp_dir)

            # Restraint options.
            template_script['experiments']['restraint'] = {
                'type': 'Harmonic',
                'restrained_receptor_atoms': [10, 11, 12],
                'restrained_ligand_atoms': 'resname MOL',
                'spring_constant': '8*kilojoule_per_mole/nanometers**2'
            }

            # Test that options are passed to the restraint correctly.
            constructor_description = template_script['experiments']['restraint']
            self.check_constructor(template_script, constructor_description,
                                   object_name='restraint', complex_phase_only=True)

    def test_sampler_building(self):
        """Test that the experiment sampler is built correctly."""
        with mmtools.utils.temporary_directory() as tmp_dir:
            template_script = self.get_implicit_template_script(tmp_dir)
            template_script['options']['resume_setup'] = True
            default_number_of_iterations = template_script['options']['default_number_of_iterations']

            # Add tested samplers.
            template_script['samplers'] = {
                'my-sampler1': {
                    'type': 'ReplicaExchangeSampler',
                    'number_of_iterations': 9,
                    'replica_mixing_scheme': 'swap-neighbors',
                },
                'my-sampler2': {
                    'type': 'MultiStateSampler',
                    'locality': 5
                }
            }

            def check_default_number_of_iterations(phase, sampler_description):
                if 'number_of_iterations' not in sampler_description:
                    assert phase.sampler.number_of_iterations == default_number_of_iterations

            # Test that options are passed to the sampler correctly.
            for sampler_id, sampler_description in template_script['samplers'].items():
                template_script['experiments']['sampler'] = sampler_id
                constructor_description = template_script['samplers'][sampler_id]
                yield (self.check_constructor, template_script, constructor_description,
                       'sampler', None, check_default_number_of_iterations)

    def test_mcmc_move_building(self):
        """Test that the experiment MCMCMoves are built correctly."""
        with mmtools.utils.temporary_directory() as tmp_dir:
            template_script = self.get_implicit_template_script(tmp_dir)
            template_script['options']['resume_setup'] = True
            template_script['experiments']['sampler'] = 'repex'

            print(template_script['samplers'])

            # Add tested samplers.
            template_script['mcmc_moves'] = {
                'my-move1': {
                    'type': 'LangevinSplittingDynamicsMove',
                    'reassign_velocities': False,
                    'splitting': 'RVOVR',
                    'n_steps': 10,
                    'timestep': '2.0*femtosecond'
                },
                'my-move2': {'type': 'SequenceMove', 'move_list': [
                    {'type': 'MCDisplacementMove', 'displacement_sigma': '5.0*nanometers'},
                    {'type': 'LangevinDynamicsMove'}
                ]}
            }

            # Test default MCMCMove.
            exp_builder = ExperimentBuilder(script=template_script)
            for experiment in exp_builder.build_experiments():
                for phase in experiment.phases:
                    mcmc_move = phase.sampler.mcmc_moves
                    if len(phase.topography.ligand_atoms) > 0:
                        assert type(mcmc_move) is mmtools.mcmc.SequenceMove
                        assert len(mcmc_move.move_list) == 3
                        assert type(mcmc_move.move_list[0]) is mmtools.mcmc.MCDisplacementMove
                        assert type(mcmc_move.move_list[1]) is mmtools.mcmc.MCRotationMove
                        assert mcmc_move.move_list[0].atom_subset == phase.topography.ligand_atoms
                        langevin_move = mcmc_move.move_list[2]
                    else:
                        langevin_move = mcmc_move
                    # Check default parameters LangevinMove
                    assert type(langevin_move) is mmtools.mcmc.LangevinSplittingDynamicsMove
                    assert langevin_move.timestep == exp_builder._options['default_timestep']
                    assert langevin_move.n_steps == exp_builder._options['default_nsteps_per_iteration']

            # Test that custom MCMCMoves are built correctly.
            template_script['samplers']['repex']['mcmc_moves'] = 'my-move1'
            constructor_description = template_script['mcmc_moves']['my-move1']
            self.check_constructor(template_script, constructor_description,
                                   object_name='sampler.mcmc_moves')

            template_script['samplers']['repex']['mcmc_moves'] = 'my-move2'
            exp_builder = ExperimentBuilder(script=template_script)
            for experiment in exp_builder.build_experiments():
                for phase in experiment.phases:
                    mcmc_move = phase.sampler.mcmc_moves
                    assert type(mcmc_move) is mmtools.mcmc.SequenceMove
                    assert len(mcmc_move.move_list) == 2
                    assert type(mcmc_move.move_list[0]) is mmtools.mcmc.MCDisplacementMove
                    assert type(mcmc_move.move_list[1]) is mmtools.mcmc.LangevinDynamicsMove
                    assert mcmc_move.move_list[0].displacement_sigma == 5.0*unit.nanometers


# ==============================================================================
# Experiment execution
# ==============================================================================

def test_expand_experiments():
    """Test that job_id and n_jobs limit the number of experiments run."""
    template_script = get_template_script()
    experiment_systems = utils.CombinatorialLeaf(['explicit-system', 'implicit-system', 'hydration-system'])
    template_script['experiments']['system'] = experiment_systems

    exp_builder = ExperimentBuilder(script=template_script, job_id=1, n_jobs=2)
    experiments = list(exp_builder._expand_experiments())
    assert len(experiments) == 2

    exp_builder = ExperimentBuilder(script=template_script, job_id=2, n_jobs=2)
    experiments = list(exp_builder._expand_experiments())
    assert len(experiments) == 1


def test_yaml_creation():
    """Test the content of generated single experiment YAML files."""
    ligand_path = examples_paths()['p-xylene']
    toluene_path = examples_paths()['toluene']
    with mmtools.utils.temporary_directory() as tmp_dir:
        molecules = """
            T4lysozyme:
                filepath: {}
                leap: {{parameters: oldff/leaprc.ff14SB}}""".format(examples_paths()['lysozyme'])
        solvent = """
            vacuum:
                nonbonded_method: NoCutoff"""
        protocol = indent(standard_protocol)
        system = """
            system:
                ligand: p-xylene
                receptor: T4lysozyme
                solvent: vacuum"""
        experiment = """
            protocol: absolute-binding
            system: system"""

        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:{}
            p-xylene:
                filepath: {}
                antechamber: {{charge_method: bcc}}
                leap: {{parameters: leaprc.gaff}}
            benzene:
                filepath: {}
                antechamber: {{charge_method: bcc}}
                leap: {{parameters: leaprc.gaff}}
        solvents:{}
            GBSA-OBC2:
                nonbonded_method: NoCutoff
                implicit_solvent: OBC2
        systems:{}
        protocols:{}
        experiments:{}
        """.format(os.path.relpath(tmp_dir), molecules,
                   os.path.relpath(ligand_path), toluene_path,
                   solvent, system, protocol, experiment)

        # We need to check whether the relative paths to the output directory and
        # for p-xylene are handled correctly while absolute paths (T4lysozyme) are
        # left untouched
        expected_yaml_content = textwrap.dedent("""
        ---
        version: '{}'
        options:
            experiments_dir: .
            output_dir: .
        molecules:{}
            p-xylene:
                filepath: {}
                antechamber: {{charge_method: bcc}}
                leap: {{parameters: leaprc.gaff}}
        solvents:{}
        systems:{}
        protocols:{}
        experiments:{}
        """.format(HIGHEST_VERSION, molecules, os.path.relpath(ligand_path, tmp_dir),
                   solvent, system, protocol, experiment))
        expected_yaml_content = expected_yaml_content[1:]  # remove first '\n'

        exp_builder = ExperimentBuilder(textwrap.dedent(yaml_content))

        # during setup we can modify molecule's fields, so we need
        # to check that it doesn't affect the YAML file exported
        experiment_dict = yaml.load(experiment, Loader=yaml.FullLoader)
        exp_builder._db.get_system(experiment_dict['system'])

        generated_yaml_path = os.path.join(tmp_dir, 'experiment.yaml')
        exp_builder._generate_yaml(experiment_dict, generated_yaml_path)
        with open(generated_yaml_path, 'r') as f:
            assert yaml.load(f, Loader=yaml.FullLoader) == yank_load(expected_yaml_content)


def test_yaml_extension():
    """Test that extending a yaml content with additional data produces the correct fusion"""
    ligand_path = examples_paths()['p-xylene']
    toluene_path = examples_paths()['toluene']
    with mmtools.utils.temporary_directory() as tmp_dir:
        molecules = """
            T4lysozyme:
                filepath: {}
                leap: {{parameters: oldff/leaprc.ff14SB}}""".format(examples_paths()['lysozyme'])
        solvent = """
            vacuum:
                nonbonded_method: NoCutoff"""
        protocol = indent(standard_protocol)
        system = """
            system:
                ligand: p-xylene
                receptor: T4lysozyme
                solvent: vacuum"""
        experiment = """
            protocol: absolute-binding
            system: system"""
        num_iterations = 5
        replacement_solvent = "HTC"


        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:{}
            p-xylene:
                filepath: {}
                antechamber: {{charge_method: bcc}}
                leap: {{parameters: leaprc.gaff}}
            benzene:
                filepath: {}
                antechamber: {{charge_method: bcc}}
                leap: {{parameters: leaprc.gaff}}
        solvents:{}
            GBSA-OBC2:
                nonbonded_method: NoCutoff
                implicit_solvent: OBC2
        systems:{}
        protocols:{}
        experiments:{}
        """.format(os.path.relpath(tmp_dir), molecules,
                   os.path.relpath(ligand_path), toluene_path,
                   solvent, system, protocol, experiment)

        yaml_extension = """
        options:
            default_number_of_iterations: {}
        solvents:
            GBSA-OBC2:
                implicit_solvent: HCT
        """.format(num_iterations, replacement_solvent)

        # We need to check whether the relative paths to the output directory and
        # for p-xylene are handled correctly while absolute paths (T4lysozyme) are
        # left untouched
        expected_yaml_content = textwrap.dedent("""
        ---
        version: '{}'
        options:
            experiments_dir: .
            output_dir: .
            default_number_of_iterations: {}
        molecules:{}
            p-xylene:
                filepath: {}
                antechamber: {{charge_method: bcc}}
                leap: {{parameters: leaprc.gaff}}
        solvents:{}
        systems:{}
        protocols:{}
        experiments:{}
        """.format(HIGHEST_VERSION, num_iterations, molecules, os.path.relpath(ligand_path, tmp_dir),
                   solvent, system, protocol, experiment))
        expected_yaml_content = expected_yaml_content[1:]  # remove first '\n'
        exp_builder = ExperimentBuilder(textwrap.dedent(yaml_content))
        exp_builder.update_yaml(yaml_extension)
        # during setup we can modify molecule's fields, so we need
        # to check that it doesn't affect the YAML file exported
        experiment_dict = yaml.load(experiment, Loader=yaml.FullLoader)
        exp_builder._db.get_system(experiment_dict['system'])
        generated_yaml_path = os.path.join(tmp_dir, 'experiment.yaml')
        exp_builder._generate_yaml(experiment_dict, generated_yaml_path)
        with open(generated_yaml_path, 'r') as f:
            assert yaml.load(f, Loader=yaml.FullLoader) == yank_load(expected_yaml_content)


@attr('slow')  # Skip on Travis-CI
def test_run_experiment_from_amber_files():
    """Test experiment run from prmtop/inpcrd files."""
    complex_path = examples_paths()['bentol-complex']
    solvent_path = examples_paths()['bentol-solvent']
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_script = get_template_script(tmp_dir)
        yaml_script['options']['anisotropic_dispersion_cutoff'] = None
        del yaml_script['molecules']  # we shouldn't need any molecule
        del yaml_script['solvents']['PME']['clearance']  # we shouldn't need this
        yaml_script['systems'] = {'explicit-system':
                {'phase1_path': complex_path, 'phase2_path': solvent_path,
                 'ligand_dsl': 'resname TOL', 'solvent': 'PME'}}

        exp_builder = ExperimentBuilder(yaml_script)
        exp_builder._check_resume()  # check_resume should not raise exceptions
        exp_builder.run_experiments()

        # The experiments folders are correctly named and positioned
        output_dir = exp_builder._get_experiment_dir('')
        assert os.path.isdir(output_dir)
        assert os.path.isfile(os.path.join(output_dir, 'complex.nc'))
        assert os.path.isfile(os.path.join(output_dir, 'solvent.nc'))
        assert os.path.isfile(os.path.join(output_dir, 'experiments.yaml'))
        assert os.path.isfile(os.path.join(output_dir, 'experiments.log'))

        # Analysis script is correct
        analysis_script_path = os.path.join(output_dir, 'analysis.yaml')
        with open(analysis_script_path, 'r') as f:
            assert yaml.load(f, Loader=yaml.FullLoader) == [['complex', 1], ['solvent', -1]]


@attr('slow')  # Skip on Travis-CI
def test_run_experiment_from_gromacs_files():
    """Test experiment run from top/gro files."""
    complex_path = examples_paths()['pxylene-complex']
    solvent_path = examples_paths()['pxylene-solvent']
    include_path = examples_paths()['pxylene-gro-include']
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_script = get_template_script(tmp_dir)
        yaml_script['options']['anisotropic_dispersion_cutoff'] = None
        del yaml_script['molecules']  # we shouldn't need any molecule
        yaml_script['systems'] = {'explicit-system':
                {'phase1_path': complex_path, 'phase2_path': solvent_path,
                 'ligand_dsl': 'resname "p-xylene"', 'solvent': 'PME',
                 'gromacs_include_dir': include_path}}
        yaml_script['experiments']['system'] = 'explicit-system'

        exp_builder = ExperimentBuilder(yaml_script)
        exp_builder._check_resume()  # check_resume should not raise exceptions
        exp_builder.run_experiments()

        # The experiments folders are correctly named and positioned
        output_dir = exp_builder._get_experiment_dir('')
        assert os.path.isdir(output_dir)
        assert os.path.isfile(os.path.join(output_dir, 'complex.nc'))
        assert os.path.isfile(os.path.join(output_dir, 'solvent.nc'))
        assert os.path.isfile(os.path.join(output_dir, 'experiments.yaml'))
        assert os.path.isfile(os.path.join(output_dir, 'experiments.log'))

        # Analysis script is correct
        analysis_script_path = os.path.join(output_dir, 'analysis.yaml')
        with open(analysis_script_path, 'r') as f:
            assert yaml.load(f, Loader=yaml.FullLoader) == [['complex', 1], ['solvent', -1]]


@attr('slow')  # Skip on Travis-CI
def test_run_experiment_from_xml_files():
    """Test hydration experiment run from pdb/xml files."""
    solvent_path = examples_paths()['toluene-solvent']
    vacuum_path = examples_paths()['toluene-vacuum']
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_script = get_template_script(tmp_dir)
        del yaml_script['molecules']  # we shouldn't need any molecule
        yaml_script['systems'] = {'explicit-system':
                {'phase1_path': solvent_path, 'phase2_path': vacuum_path,
                 'solvent_dsl': 'not resname TOL'}}

        exp_builder = ExperimentBuilder(yaml_script)
        exp_builder._check_resume()  # check_resume should not raise exceptions
        exp_builder.run_experiments()

        # The experiments folders are correctly named and positioned
        output_dir = exp_builder._get_experiment_dir('')
        assert os.path.isdir(output_dir)
        assert os.path.isfile(os.path.join(output_dir, 'complex.nc'))
        assert os.path.isfile(os.path.join(output_dir, 'solvent.nc'))
        assert os.path.isfile(os.path.join(output_dir, 'experiments.yaml'))
        assert os.path.isfile(os.path.join(output_dir, 'experiments.log'))

        # Analysis script is correct
        analysis_script_path = os.path.join(output_dir, 'analysis.yaml')
        with open(analysis_script_path, 'r') as f:
            assert yaml.load(f, Loader=yaml.FullLoader) == [['complex', 1], ['solvent', -1]]


@attr('slow')  # Skip on Travis-CI
def test_run_experiment():
    """Test experiment run and resuming."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            resume_setup: no
            resume_simulation: no
            default_number_of_iterations: 0
            output_dir: {}
            setup_dir: ''
            experiments_dir: ''
            minimize: no
            annihilate_sterics: yes
        molecules:
            T4lysozyme:
                filepath: {}
                leap: {{parameters: oldff/leaprc.ff14SB}}
                select: 0
            p-xylene:
                filepath: {}
                antechamber: {{charge_method: bcc}}
                leap: {{parameters: leaprc.gaff}}
        solvents:
            vacuum:
                nonbonded_method: NoCutoff
            GBSA-OBC2:
                nonbonded_method: NoCutoff
                implicit_solvent: OBC2
        protocols:{}
        systems:
            system:
                receptor: T4lysozyme
                ligand: p-xylene
                solvent: !Combinatorial [vacuum, GBSA-OBC2]
        experiments:
            system: system
            protocol: absolute-binding
            restraint:
                type: FlatBottom
                spring_constant: 0.6*kilocalorie_per_mole/angstroms**2
                well_radius: 5.2*nanometers
                restrained_receptor_atoms: 1644
                restrained_ligand_atoms: 2609
            options:
                temperature: 302.0*kelvin
        """.format(tmp_dir, examples_paths()['lysozyme'], examples_paths()['p-xylene'],
                   indent(standard_protocol))

        exp_builder = ExperimentBuilder(textwrap.dedent(yaml_content))

        # Now check_setup_resume should not raise exceptions
        exp_builder._check_resume()

        # We setup a molecule and with resume_setup: now we can't do the experiment
        err_msg = ''
        exp_builder._options['resume_setup'] = False
        exp_builder._db._setup_molecules('p-xylene')
        try:
            exp_builder.run_experiments()
        except YamlParseError as e:
            err_msg = str(e)
        assert 'molecule' in err_msg

        # Same thing with a system
        err_msg = ''
        system_dir = os.path.dirname(
            exp_builder._db.get_system('system_GBSAOBC2')[0].position_path)
        try:
            exp_builder.run_experiments()
        except YamlParseError as e:
            err_msg = str(e)
        assert 'system' in err_msg

        # Now we set resume_setup to True and things work
        exp_builder._options['resume_setup'] = True
        ligand_dir = exp_builder._db.get_molecule_dir('p-xylene')
        frcmod_file = os.path.join(ligand_dir, 'p-xylene.frcmod')
        prmtop_file = os.path.join(system_dir, 'complex.prmtop')
        molecule_last_touched = os.stat(frcmod_file).st_mtime
        system_last_touched = os.stat(prmtop_file).st_mtime
        exp_builder.run_experiments()

        # Neither the system nor the molecule has been processed again
        assert molecule_last_touched == os.stat(frcmod_file).st_mtime
        assert system_last_touched == os.stat(prmtop_file).st_mtime

        # The experiments folders are correctly named and positioned
        for exp_name in ['systemvacuum', 'systemGBSAOBC2']:
            # The output directory must be the one in the experiment section
            output_dir = os.path.join(tmp_dir, exp_name)
            assert os.path.isdir(output_dir)
            assert os.path.isfile(os.path.join(output_dir, 'complex.nc'))
            assert os.path.isfile(os.path.join(output_dir, 'solvent.nc'))
            assert os.path.isfile(os.path.join(output_dir, exp_name + '.yaml'))
            assert os.path.isfile(os.path.join(output_dir, exp_name + '.log'))

            # Analysis script is correct
            analysis_script_path = os.path.join(output_dir, 'analysis.yaml')
            with open(analysis_script_path, 'r') as f:
                assert yaml.load(f, Loader=yaml.FullLoader) == [['complex', 1], ['solvent', -1]]

        # Now we can't run the experiment again with resume_simulation: no
        exp_builder._options['resume_simulation'] = False
        try:
            exp_builder.run_experiments()
        except YamlParseError as e:
            err_msg = str(e)
        assert 'experiment' in err_msg

        # We set resume_simulation: yes and now things work
        exp_builder._options['resume_simulation'] = True
        exp_builder.run_experiments()


def solvation_stock(tmp_dir, overwrite_options=None):
    """Stock actions to take for a solvation run"""
    yaml_script = get_template_script(tmp_dir)
    yaml_script['experiments']['system'] = 'hydration-system'
    yaml_script['experiments']['protocol'] = 'hydration-protocol'
    # Pop out all non-hydration system items for setup speed
    molecule_poppers = []
    for molecule in yaml_script['molecules'].keys():
        if molecule not in yaml_script['systems']['hydration-system'].values():
            molecule_poppers.append(molecule)
    for molecule in molecule_poppers:
        yaml_script['molecules'].pop(molecule, None)
    system_poppers = []
    for system in yaml_script['systems'].keys():
        if system != 'hydration-system':
            system_poppers.append(system)
    for system in system_poppers:
        yaml_script['systems'].pop(system, None)

    if overwrite_options is not None:
        yaml_script = utils.update_nested_dict(yaml_script, overwrite_options)

    exp_builder = ExperimentBuilder(yaml_script)
    exp_builder._check_resume()  # check_resume should not raise exceptions
    exp_builder.run_experiments()
    return yaml_script, exp_builder


def test_run_solvation_experiment():
    """Test solvation free energy experiment run."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        _, exp_builder = solvation_stock(tmp_dir)
        # The experiments folders are correctly named and positioned
        output_dir = exp_builder._get_experiment_dir('')

        assert os.path.isdir(output_dir)
        for solvent in ['solvent1.nc', 'solvent2.nc']:
            solvent_path = os.path.join(output_dir, solvent)
            reporter = mmtools.multistate.MultiStateReporter(solvent_path, open_mode=None)
            assert reporter.storage_exists()
            del reporter
        assert os.path.isfile(os.path.join(output_dir, 'experiments.yaml'))
        assert os.path.isfile(os.path.join(output_dir, 'experiments.log'))

        # Analysis script is correct
        analysis_script_path = os.path.join(output_dir, 'analysis.yaml')
        with open(analysis_script_path, 'r') as f:
            assert yaml.load(f, Loader=yaml.FullLoader) == [['solvent1', 1], ['solvent2', -1]]


class TestTrailblazeAlchemicalPath:
    """Test suite for the automatic discretization of the alchemical path."""

    def _get_harmonic_oscillator_script(self, tmp_dir, alchemical_path='auto', trailblazer_options=None):
        # Setup only 1 hydration free energy system in implicit solvent and vacuum.
        yaml_script = get_template_script(tmp_dir, systems=['hydration-system'])
        yaml_script['systems']['hydration-system']['solvent1'] = 'GBSA-OBC2'
        yaml_script['experiments']['system'] = 'hydration-system'
        yaml_script['experiments']['protocol'] = 'hydration-protocol'

        # We run trailblaze only for the phase in vacuum.
        yaml_script['protocols']['hydration-protocol']['solvent2']['alchemical_path'] = alchemical_path
        if trailblazer_options is not None:
            yaml_script['protocols']['hydration-protocol']['solvent2']['trailblazer_options'] = trailblazer_options

        # Make the generation of the trailblaze samples inexpensive.
        yaml_script['mcmc_moves']['single']['n_steps'] = 1
        yaml_script['mcmc_moves']['single']['timestep'] = '0.5*femtosecond'
        yaml_script['samplers']['repex']['mcmc_moves'] = 'single'
        yaml_script['experiments']['sampler'] = 'repex'
        yaml_script['options']['platform'] = 'CPU'

        return yaml_script

    def test_auto_alchemical_path(self):
        """Test automatic alchemical path found by thermodynamic trailblazing when the option 'auto' is set."""
        with mmtools.utils.temporary_directory() as tmp_dir:
            # Setup only 1 hydration free energy system in implicit solvent and vacuum.
            yaml_script = self._get_harmonic_oscillator_script(
                tmp_dir, alchemical_path='auto',
                trailblazer_options={'n_equilibration_iterations': 0}
            )
            yaml_script['options']['resume_setup'] = False
            yaml_script['options']['resume_simulation'] = False
            exp_builder = ExperimentBuilder(yaml_script)

            # ExperimentBuilder._get_experiment_protocol handles dummy protocols.
            experiment_path, experiment_description = next(exp_builder._expand_experiments())
            with assert_raises(FileNotFoundError):
                exp_builder._get_experiment_protocol(experiment_path, experiment_description)
            dummy_protocol = exp_builder._get_experiment_protocol(experiment_path, experiment_description,
                                                                  use_dummy_protocol=True)
            assert dummy_protocol['solvent2']['alchemical_path'] == {}  # This is the dummy protocol.

            # check_resume should not raise exceptions at this point.
            exp_builder._check_resume()

            # Building the experiment should generate the alchemical path.
            for experiment in exp_builder.build_experiments():
                pass

            # The experiment has the correct path. Only the path of solvent2 has been generated.
            expected_generated_protocol = {
                'lambda_electrostatics': [1.0, 0.0],
                'lambda_sterics': [1.0, 1.0]
            }
            assert experiment.phases[0].protocol == yaml_script['protocols']['hydration-protocol']['solvent1']['alchemical_path']
            assert experiment.phases[1].protocol == expected_generated_protocol

            # YANK takes advantage of the samples generated during the trailblaze.
            assert isinstance(experiment.phases[0].sampler_states, mmtools.states.SamplerState)
            assert len(experiment.phases[1].sampler_states) == 2

            # Resuming fails at this point because we have
            # generated the YAML file containing the protocol.
            with assert_raises(YamlParseError):
                next(exp_builder.build_experiments())

            # When resuming, ExperimentBuilder should recycle the path from the previous run.
            generated_yaml_script_path = exp_builder._get_generated_yaml_script_path('')
            last_touched_yaml = os.stat(generated_yaml_script_path).st_mtime
            exp_builder._options['resume_setup'] = True
            exp_builder._options['resume_simulation'] = True
            exp_builder.run_experiments()
            assert last_touched_yaml == os.stat(generated_yaml_script_path).st_mtime

    def test_alchemical_functions_path(self):
        """Test automatic alchemical path found from alchemical functions."""
        with mmtools.utils.temporary_directory() as tmp_dir:
            # Setup only 1 hydration free energy system in implicit solvent and vacuum.
            yaml_script = self._get_harmonic_oscillator_script(
                tmp_dir,
                alchemical_path={'lambda_electrostatics': 'lambda',
                                 'lambda_sterics': 'lambda',
                                 'lambda': [0.0, 1.0]},
                trailblazer_options={'function_variable_name': 'lambda',
                                     'n_equilibration_iterations': 0}
            )
            exp_builder = ExperimentBuilder(yaml_script)

            # Building the experiment should generate the alchemical path.
            for experiment in exp_builder.build_experiments():
                pass

            # The experiment has the correct path. Only the path of solvent2 has been generated.
            expected_generated_protocol = {
                'lambda_electrostatics': [0.0, 1.0],
                'lambda_sterics': [0.0, 1.0]
            }
            assert experiment.phases[0].protocol == yaml_script['protocols']['hydration-protocol']['solvent1']['alchemical_path']
            assert experiment.phases[1].protocol == expected_generated_protocol, experiment.phases[1].protocol

            # YANK takes advantage of the samples generated during the trailblaze.
            assert isinstance(experiment.phases[0].sampler_states, mmtools.states.SamplerState)
            assert len(experiment.phases[1].sampler_states) == 2


def test_experiment_nan():
    """Test that eventual NaN's are handled and that experiment is signal as completed."""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_script = get_functionality_script(output_directory=tmp_dir, experiment_repeats=0, number_nan_repeats=1)
        exp_builder = ExperimentBuilder(script=yaml_script)
        with moltools.utils.temporary_cd(exp_builder._script_dir):
            exp_builder._check_resume()
            exp_builder._setup_experiments()
            exp_builder._generate_experiments_protocols()
            for experiment in exp_builder._expand_experiments():
                is_completed = exp_builder._run_experiment(experiment)
                assert is_completed


def test_multi_experiment_nan():
    """Test that no one experiment going NaN crashes the simulation"""
    with mmtools.utils.temporary_directory() as tmp_dir:
        yaml_script = get_functionality_script(output_directory=tmp_dir,
                                               number_of_iter=2,
                                               experiment_repeats=2,
                                               number_nan_repeats=2)
        exp_builder = ExperimentBuilder(yaml_script)
        # This should run correctly and not raise errors
        exp_builder.run_experiments()


if __name__ == '__main__':
    test_run_solvation_experiment()
