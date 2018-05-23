#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Experiment
==========

Tools to build Yank experiments from a YAML configuration file.

This is not something that should be normally invoked by the user, and instead
created by going through the Command Line Interface with the ``yank script`` command.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import collections
import copy
import logging
import os

import numpy as np
import cerberus
import cerberus.errors
import openmmtools as mmtools
import openmoltools as moltools
import yaml
from simtk import unit, openmm
from simtk.openmm.app import PDBFile, AmberPrmtopFile

from . import utils, pipeline, mpi, restraints, schema, multistate
from .yank import AlchemicalPhase, Topography

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

HIGHEST_VERSION = '1.3'  # highest version of YAML syntax


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_openmm_nonbonded_methods_strings():
    """
    Get the list of valid OpenMM Nonbonded methods YANK can process

    Returns
    -------
    valid_methods : list of str

    """
    return ['NoCutoff', 'CutoffPeriodic', 'CutoffNonPeriodic', 'Ewald', 'PME']


def get_openmm_implicit_nb_method_strings():
    """
    Get the subset of nonbonded method strings which work for implicit solvent

    Returns
    -------
    valid_methods : list of str
    """
    return get_openmm_nonbonded_methods_strings()[:1]


def get_openmm_explicit_nb_method_strings():
    """
    Get the subset of nonbonded method strings which work for explicit solvent

    Returns
    -------
    valid_methods : list of str
    """
    return get_openmm_nonbonded_methods_strings()[1:]


def to_openmm_app(input_string):
    """
    Converter function to be used with :func:`yank.utils.validate_parameters`.

    Parameters
    ----------
    input_string : str
        Method name of openmm.app to fetch

    Returns
    -------
    method : Method of openmm.app
        Returns openmm.app.{input_string}
    """
    return getattr(openmm.app, input_string)


def _is_phase_completed(status, number_of_iterations):
    """Check if the stored simulation is completed.

    When the simulation is resumed, the number of iterations to run
    in the YAML script could be updated, so we can't rely entirely on
    the is_completed field in the ReplicaExchange.Status object.

    Parameters
    ----------
    status : namedtuple
        The status object returned by ``yank.AlchemicalPhase``.
    number_of_iterations : int
        The total number of iterations that the simulation must perform.

    """
    # TODO allow users to change online analysis options on resuming.
    if status.target_error is None and status.iteration < number_of_iterations:
        is_completed = False
    else:
        is_completed = status.is_completed
    return is_completed


# ==============================================================================
# UTILITY CLASSES
# ==============================================================================

class YamlParseError(Exception):
    """Represent errors occurring during parsing of Yank YAML file."""
    def __init__(self, message):
        super(YamlParseError, self).__init__(message)
        logger.error(message)


class YankLoader(yaml.Loader):
    """PyYAML Loader that recognized !Combinatorial nodes and load OrderedDicts."""
    def __init__(self, *args, **kwargs):
        super(YankLoader, self).__init__(*args, **kwargs)
        self.add_constructor(u'!Combinatorial', self.combinatorial_constructor)
        self.add_constructor(u'!Ordered', self.ordered_constructor)

    @staticmethod
    def combinatorial_constructor(loader, node):
        """Constructor for YAML !Combinatorial entries."""
        return utils.CombinatorialLeaf(loader.construct_sequence(node))

    @staticmethod
    def ordered_constructor(loader, node):
        """Constructor for YAML !Ordered tag."""
        loader.flatten_mapping(node)
        return collections.OrderedDict(loader.construct_pairs(node))


class YankDumper(yaml.Dumper):
    """PyYAML Dumper that always return sequences in flow style and maps in block style."""
    def __init__(self, *args, **kwargs):
        super(YankDumper, self).__init__(*args, **kwargs)
        self.add_representer(utils.CombinatorialLeaf, self.combinatorial_representer)
        self.add_representer(collections.OrderedDict, self.ordered_representer)

    def represent_sequence(self, tag, sequence, flow_style=None):
        return yaml.Dumper.represent_sequence(self, tag, sequence, flow_style=True)

    def represent_mapping(self, tag, mapping, flow_style=None):
        return yaml.Dumper.represent_mapping(self, tag, mapping, flow_style=False)

    @staticmethod
    def combinatorial_representer(dumper, data):
        """YAML representer CombinatorialLeaf nodes."""
        return dumper.represent_sequence(u'!Combinatorial', data)

    @staticmethod
    def ordered_representer(dumper, data):
        """YAML representer OrderedDict nodes."""
        return dumper.represent_mapping(u'!Ordered', data)


# ==============================================================================
# BUILDER CLASS
# ==============================================================================

class AlchemicalPhaseFactory(object):
    """
    YANK simulation phase entirely contained as one object.

    Creates a full phase to simulate, expects Replica Exchange simulation for now.

    Parameters
    ----------
    sampler : yank.multistate.MultiStateSampler
        Sampler which will carry out the simulation
    thermodynamic_state : openmmtools.states.ThermodynamicState
        Reference thermodynamic state without any alchemical modifications
    sampler_states : openmmtools.states.SamplerState
        Sampler state to initialize from, including the positions of the atoms
    topography : yank.yank.Topography
        Topography defining the ligand atoms and other atoms
    protocol : dict of lists
        Alchemical protocol to create states from.

        Format should be ``{parameter_name : parameter_values}``

        where ``parameter_name`` is the name of the specific alchemical
        parameter (e.g. ``lambda_sterics``), and ``parameter_values`` is a list of values for that parameter where each
        entry is one state.

        Each of the ``parameter_values`` lists for every ``parameter_name`` should be the same length.

    storage : yank.multistate.MultiStateReporter or str
        Reporter object to use, or file path to create the reporter at
        Will be a :class:`yank.multistate.MultiStateReporter` internally if str is given
    restraint : yank.restraint.ReceptorLigandRestraint or None, Optional, Default: None
        Optional restraint to apply to the system
    alchemical_regions : openmmtools.alchemy.AlchemicalRegion or None, Optional, Default: None
        Alchemical regions which define which atoms to modify.
    alchemical_factory : openmmtools.alchemy.AbsoluteAlchemicalFactory, Optional, Default: None
        Alchemical factory with which to create the alchemical system with if you don't want to use all the previously
        defined options.
        This is passed on to :class:`yank.yank.AlchemicalPhase`
    metadata : dict
        Additional metdata to pass on to :class:`yank.yank.AlchemicalPhase`
    options : dict
        Additional options to setup the rest of the process.
        See the DEFAULT_OPTIONS for this class in the source code or look at the *options* header for the YAML options.
    """

    DEFAULT_OPTIONS = {
        'anisotropic_dispersion_cutoff': 'auto',
        'minimize': True,
        'minimize_tolerance': 1.0 * unit.kilojoules_per_mole/unit.nanometers,
        'minimize_max_iterations': 1000,
        'randomize_ligand': False,
        'randomize_ligand_sigma_multiplier': 2.0,
        'randomize_ligand_close_cutoff': 1.5 * unit.angstrom,
        'number_of_equilibration_iterations': 0,
        'equilibration_timestep': 1.0 * unit.femtosecond,
        'checkpoint_interval': 200,
        'store_solute_trajectory': True
    }

    def __init__(self, sampler, thermodynamic_state, sampler_states, topography,
                 protocol, storage, restraint=None, alchemical_regions=None,
                 alchemical_factory=None, metadata=None, **options):
        self.sampler = sampler
        self.thermodynamic_state = thermodynamic_state
        self.sampler_states = sampler_states
        self.topography = topography
        self.protocol = protocol
        self.storage = storage
        self.restraint = restraint
        self.alchemical_regions = alchemical_regions
        self.alchemical_factory = alchemical_factory
        self.metadata = metadata
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options.update(options)

    def create_alchemical_phase(self):
        """
        Create the alchemical phase based on all the options

        This only creates it, but does nothing else to prepare for simulations. The ``initialize_alchemical_phase``
        will actually minimize, randomize ligand, and/or equilibrate if requested.

        Returns
        -------
        alchemical_phase : yank.yank.AlchemicalPhase

        See Also
        --------
        initialize_alchemical_phase
        """
        alchemical_phase = AlchemicalPhase(self.sampler)
        create_kwargs = self.__dict__.copy()
        create_kwargs.pop('options')
        create_kwargs.pop('sampler')

        # Create a reporter if this is only a path.
        if isinstance(self.storage, str):
            checkpoint_interval = self.options['checkpoint_interval']
            # Get the solute atoms
            if self.options['store_solute_trajectory']:
                # "Solute" is basically just not water. Includes all non-water atoms and ions
                # Topography ensures the union of solute_atoms and ions_atoms is a null set
                solute_atoms = self.topography.solute_atoms + self.topography.ions_atoms
                if checkpoint_interval == 1:
                    logger.warning("WARNING! You have specified both a solute-only trajectory AND a checkpoint "
                                   "interval of 1! You are about write the trajectory of the solute twice!\n"
                                   "This can be okay if you are running explicit solvent and want faster retrieval "
                                   "of the solute atoms, but in implicit solvent, this is redundant.")
            else:
                solute_atoms = ()
            # We don't allow checkpoint file overwriting in YAML file
            reporter = multistate.MultiStateReporter(self.storage, checkpoint_interval=checkpoint_interval,
                                                     analysis_particle_indices=solute_atoms)
            create_kwargs['storage'] = reporter
            self.storage = reporter

        dispersion_cutoff = self.options['anisotropic_dispersion_cutoff']  # This will be None or an option
        alchemical_phase.create(anisotropic_dispersion_cutoff=dispersion_cutoff,
                                **create_kwargs)
        return alchemical_phase

    def initialize_alchemical_phase(self):
        """
        Create and set all the initial options for the alchemical phase

        This minimizes, randomizes_ligand, and equilibrates the alchemical_phase on top of creating it, if the various
        options are set

        Returns
        -------
        alchemical_phase : yank.yank.AlchemicalPhase
        """
        alchemical_phase = self.create_alchemical_phase()

        # Minimize if requested.
        if self.options['minimize']:
            tolerance = self.options['minimize_tolerance']
            max_iterations = self.options['minimize_max_iterations']
            alchemical_phase.minimize(tolerance=tolerance, max_iterations=max_iterations)

        # Randomize ligand if requested.
        if self.options['randomize_ligand']:
            sigma_multiplier = self.options['randomize_ligand_sigma_multiplier']
            close_cutoff = self.options['randomize_ligand_close_cutoff']
            alchemical_phase.randomize_ligand(sigma_multiplier=sigma_multiplier,
                                              close_cutoff=close_cutoff)

        # Equilibrate if requested.
        if self.options['number_of_equilibration_iterations'] > 0:
            n_iterations = self.options['number_of_equilibration_iterations']
            # Get main propagation move. If this is a sequence, find first IntegratorMove.
            mcmc_move = self.sampler.mcmc_moves[0]
            try:
                integrator_moves = [move for move in mcmc_move.move_list
                                    if isinstance(move, mmtools.mcmc.BaseIntegratorMove)]
                mcmc_move = copy.deepcopy(integrator_moves[0])
            except AttributeError:
                mcmc_move = copy.deepcopy(mcmc_move)
            logger.debug('Using {} for equilibration.'.format(mcmc_move))

            # Fix move parameters for equilibration.
            move_parameters = dict(
                n_steps=500,
                n_restart_attempts=6,
                timestep=self.options['equilibration_timestep'],
                collision_rate=90.0/unit.picosecond,
                measure_shadow_work=False,
                measure_heat=False
            )
            for parameter_name, parameter_value in move_parameters.items():
                if hasattr(mcmc_move, parameter_name):
                    setattr(mcmc_move, parameter_name, parameter_value)

            # Run equilibration.
            alchemical_phase.equilibrate(n_iterations, mcmc_moves=mcmc_move)

        return alchemical_phase


class Experiment(object):
    """
    An experiment built by :class:`ExperimentBuilder`.

    This is a completely defined experiment with all parameters and settings ready to go.

    It is highly recommended to **NOT** use this class directly, and instead rely on the :class:`ExperimentBuilder`
    class to parse all options, configure all phases, properly set up the experiments, and even run them.

    These experiments are frequently created with the :func:`ExperimentBuilder.build_experiments` method.

    Parameters
    ----------
    phases : list of yank.yank.AlchemicalPhases
        Phases to run for the experiment
    number_of_iterations : int or infinity
        Total number of iterations each phase will be run for. Both
        ``float('inf')`` and ``numpy.inf`` are accepted for infinity.
    switch_phase_interval : int
        Number of iterations each phase will be run before the cycling to the next phase

    Attributes
    ----------
    iteration

    See Also
    --------
    ExperimentBuilder

    """
    def __init__(self, phases, number_of_iterations, switch_phase_interval):
        self.phases = phases
        self.number_of_iterations = number_of_iterations
        self.switch_phase_interval = switch_phase_interval
        self._phases_last_iterations = [None, None]
        self._are_phases_completed = [False, False]

    @property
    def iteration(self):
        """pair of int, Current total number of iterations which have been run for each phase."""
        if None in self._phases_last_iterations:
            return 0, 0
        return self._phases_last_iterations

    @property
    def is_completed(self):
        return all(self._are_phases_completed)

    def run(self, n_iterations=None):
        """
        Run the experiment.

        Runs until either the maximum number of iterations have been reached or the sampler
        for that phase reports its own completion (e.g. online analysis)

        Parameters
        ----------
        n_iterations : int or None, Optional, Default: None
            Optional parameter to run for a finite number of iterations instead of up to the maximum number of
            iterations.

        """
        # Handle default argument.
        if n_iterations is None:
            n_iterations = self.number_of_iterations

        # Handle case in which we don't alternate between phases.
        if self.switch_phase_interval <= 0:
            switch_phase_interval = self.number_of_iterations
        else:
            switch_phase_interval = self.switch_phase_interval

        # Count down the iterations to run.
        iterations_left = [None, None]
        while iterations_left != [0, 0]:

            # Alternate phases every switch_phase_interval iterations.
            for phase_id, phase in enumerate(self.phases):
                # Phases may get out of sync if the user delete the storage
                # file of only one phase and restart. Here we check that the
                # phase still has iterations to run before creating it.
                if self._are_phases_completed[phase_id]:
                    iterations_left[phase_id] = 0
                    continue

                # If this is a new simulation, initialize alchemical phase.
                if isinstance(phase, AlchemicalPhaseFactory):
                    alchemical_phase = phase.initialize_alchemical_phase()
                    self.phases[phase_id] = phase.storage  # Should automatically be a Reporter class
                else:  # Resume previously created simulation.
                    # Check the status before loading the full alchemical phase object.
                    status = AlchemicalPhase.read_status(phase)
                    if _is_phase_completed(status, self.number_of_iterations):
                        self._are_phases_completed[phase_id] = True
                        iterations_left[phase_id] = 0
                        continue
                    alchemical_phase = AlchemicalPhase.from_storage(phase)

                # TODO allow users to change online analysis options on resuming.
                # Update total number of iterations. This may write the new number
                # of iterations in the storage file so we do it only if necessary.
                if alchemical_phase.number_of_iterations != self.number_of_iterations:
                    alchemical_phase.number_of_iterations = self.number_of_iterations

                # Determine number of iterations to run in this function call.
                if iterations_left[phase_id] is None:
                    total_iterations_left = self.number_of_iterations - alchemical_phase.iteration
                    iterations_left[phase_id] = min(n_iterations, total_iterations_left)

                # Run simulation for iterations_left or until we have to switch phase.
                iterations_to_run = min(iterations_left[phase_id], switch_phase_interval)
                try:
                    alchemical_phase.run(n_iterations=iterations_to_run)
                except multistate.SimulationNaNError:
                    # Simulation has NaN'd, this experiment is done, flag phases as done and send error up stack
                    self._are_phases_completed = [True] * len(self._are_phases_completed)
                    raise

                # Check if the phase has converged.
                self._are_phases_completed[phase_id] = alchemical_phase.is_completed

                # Update phase iteration info. iterations_to_run may be infinity
                # if number_of_iterations is.
                if iterations_to_run == float('inf') and self._are_phases_completed[phase_id]:
                    iterations_left[phase_id] = 0
                else:
                    iterations_left[phase_id] -= iterations_to_run
                self._phases_last_iterations[phase_id] = alchemical_phase.iteration

                # Delete alchemical phase and prepare switching.
                del alchemical_phase


class ExperimentBuilder(object):
    """Parse YAML configuration file and build the experiment.

    The relative paths indicated in the script are assumed to be relative to
    the script directory. However, if ExperimentBuilder is initiated with a string
    rather than a file path, the paths will be relative to the user's working
    directory.

    The class firstly perform a dry run to check if this is going to overwrite
    some files and raises an exception if it finds already existing output folders
    unless the options resume_setup or resume_simulation are True.

    Parameters
    ----------
    script : str or dict
        A path to the YAML script or the YAML content. If not specified, you
        can load it later by using :func:`parse` (default is None).
    job_id : None or int
        If you want to split the experiments among different executions,
        you can set this to an integer 1 <= job_id <= n_jobs, and this
        :class:`ExperimentBuilder` will run only 1/n_jobs of the experiments.
    n_jobs : None or int
        If ``job_id`` is specified, this is the total number of jobs that
        you are running in parallel from your script.

    See Also
    --------
    Experiment

    Examples
    --------
    >>> import textwrap
    >>> import openmmtools as mmtools
    >>> import yank.utils
    >>> setup_dir = yank.utils.get_data_filename(os.path.join('..', 'examples',
    ...                                          'p-xylene-implicit', 'input'))
    >>> pxylene_path = os.path.join(setup_dir, 'p-xylene.mol2')
    >>> lysozyme_path = os.path.join(setup_dir, '181L-pdbfixer.pdb')
    >>> with mmtools.utils.temporary_directory() as tmp_dir:
    ...     yaml_content = '''
    ...     ---
    ...     options:
    ...       default_number_of_iterations: 1
    ...       output_dir: {}
    ...     molecules:
    ...       T4lysozyme:
    ...         filepath: {}
    ...       p-xylene:
    ...         filepath: {}
    ...         antechamber:
    ...           charge_method: bcc
    ...     solvents:
    ...       vacuum:
    ...         nonbonded_method: NoCutoff
    ...     systems:
    ...         my_system:
    ...             receptor: T4lysozyme
    ...             ligand: p-xylene
    ...             solvent: vacuum
    ...             leap:
    ...               parameters: [leaprc.gaff, leaprc.ff14SB]
    ...     protocols:
    ...       absolute-binding:
    ...         complex:
    ...           alchemical_path:
    ...             lambda_electrostatics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
    ...             lambda_sterics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
    ...         solvent:
    ...           alchemical_path:
    ...             lambda_electrostatics: [1.0, 0.8, 0.6, 0.3, 0.0]
    ...             lambda_sterics: [1.0, 0.8, 0.6, 0.3, 0.0]
    ...     experiments:
    ...       system: my_system
    ...       protocol: absolute-binding
    ...     '''.format(tmp_dir, lysozyme_path, pxylene_path)
    >>> yaml_builder = ExperimentBuilder(textwrap.dedent(yaml_content))
    >>> yaml_builder.run_experiments()

    """

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    # These are options that can be specified only in the main "options" section.
    GENERAL_DEFAULT_OPTIONS = {
        'verbose': False,
        'resume_setup': False,
        'resume_simulation': False,
        'output_dir': 'output',
        'setup_dir': 'setup',
        'experiments_dir': 'experiments',
        'platform': 'fastest',
        'precision': 'auto',
        'max_n_contexts': 3,
        'switch_experiment_interval': 0,
        'processes_per_experiment': 'auto'
    }

    # These options can be overwritten also in the "experiment"
    # section and they can be thus combinatorially expanded.
    EXPERIMENT_DEFAULT_OPTIONS = {
        'switch_phase_interval': 0,
        'temperature': 298 * unit.kelvin,
        'pressure': 1 * unit.atmosphere,
        'constraints': openmm.app.HBonds,
        'hydrogen_mass': 1 * unit.amu,
        'default_nsteps_per_iteration': 500,
        'default_timestep': 2.0 * unit.femtosecond,
        'default_number_of_iterations': 5000
    }

    def __init__(self, script=None, job_id=None, n_jobs=None):
        """
        Constructor.

        """
        # Check consistency job_id and n_jobs.
        if job_id is not None:
            if n_jobs is None:
                raise ValueError('n_jobs must be specified together with job_id')
            if not 1 <= job_id <= n_jobs:
                raise ValueError('job_id must be between 1 and n_jobs ({})'.format(n_jobs))

        self._job_id = job_id
        self._n_jobs = n_jobs

        self._options = self.GENERAL_DEFAULT_OPTIONS.copy()
        self._options.update(self.EXPERIMENT_DEFAULT_OPTIONS.copy())

        self._version = None
        self._script_dir = os.getcwd()  # basic dir for relative paths
        self._db = None  # Database containing molecules created in parse()
        self._raw_yaml = {}  # Unconverted input YAML script, helpful for
        self._expanded_raw_yaml = {}  # Raw YAML with selective keys chosen and blank dictionaries for missing keys
        self._protocols = {}  # Alchemical protocols description
        self._experiments = {}  # Experiments description

        # Parse YAML script
        if script is not None:
            self.parse(script)

    def update_yaml(self, script):
        """
        Update the current yaml content and reparse it

        Parameters
        ----------
        script : str or dict
            String which accepts multiple forms of YAML content that is one of the following:

            File path to the YAML file

            String containing all the YAML data

            Dict of yaml content you wish to replace

        See Also
        --------
        utils.update_nested_dict

        """
        current_content = self._raw_yaml
        try:
            with open(script, 'r') as f:
                new_content = yaml.load(f, Loader=YankLoader)
        except IOError:  # string
            new_content = yaml.load(script, Loader=YankLoader)
        except TypeError:  # dict
            new_content = script.copy()
        combined_content = utils.update_nested_dict(current_content, new_content)
        self.parse(combined_content)

    def parse(self, script):
        """Parse the given YAML configuration file.

        Validate the syntax and load the script into memory. This does not build
        the actual experiment.

        Parameters
        ----------
        script : str or dict
            A path to the YAML script or the YAML content.

        Raises
        ------
        YamlParseError
            If the input YAML script is syntactically incorrect.

        """
        # TODO check version of yank-yaml language
        # TODO what if there are multiple streams in the YAML file?
        # Load YAML script and decide working directory for relative paths
        try:
            with open(script, 'r') as f:
                yaml_content = yaml.load(f, Loader=YankLoader)
            self._script_dir = os.path.dirname(script)
        except IOError:  # string
            yaml_content = yaml.load(script, Loader=YankLoader)
        except TypeError:  # dict
            yaml_content = script.copy()

        self._raw_yaml = yaml_content.copy()

        # Check that YAML loading was successful
        if yaml_content is None:
            raise YamlParseError('The YAML file is empty!')
        if not isinstance(yaml_content, dict):
            raise YamlParseError('Cannot load YAML from source: {}'.format(script))

        # Check version (currently there's only one)
        try:
            self._version = yaml_content['version']
        except KeyError:
            self._version = HIGHEST_VERSION
        else:
            if self._version != HIGHEST_VERSION:
                raise ValueError('Unsupported syntax version {}'.format(self._version))

        # Expand combinatorial molecules and systems
        yaml_content = self._expand_molecules(yaml_content)
        yaml_content = self._expand_systems(yaml_content)

        # Save raw YAML content that will be needed when generating the YAML files
        self._expanded_raw_yaml = copy.deepcopy({key: yaml_content.get(key, {})
                                                 for key in ['options', 'molecules', 'solvents',
                                                             'systems', 'protocols']})

        # Validate options and overwrite defaults
        self._options.update(self._validate_options(yaml_content.get('options', {}),
                                                    validate_general_options=True))

        # Setup general logging
        utils.config_root_logger(self._options['verbose'], log_file_path=None)

        # Configure ContextCache, platform and precision. A Yank simulation
        # currently needs 3 contexts: 1 for the alchemical states and 2 for
        # the states with expanded cutoff.
        platform = self._configure_platform(self._options['platform'],
                                            self._options['precision'])
        try:
            mmtools.cache.global_context_cache.platform = platform
        except RuntimeError:
            # The cache has been already used. Empty it before switching platform.
            mmtools.cache.global_context_cache.empty()
            mmtools.cache.global_context_cache.platform = platform
        mmtools.cache.global_context_cache.capacity = self._options['max_n_contexts']

        # Initialize and configure database with molecules, solvents and systems
        setup_dir = os.path.join(self._options['output_dir'], self._options['setup_dir'])
        self._db = pipeline.SetupDatabase(setup_dir=setup_dir)
        self._db.molecules = self._validate_molecules(yaml_content.get('molecules', {}))
        self._db.solvents = self._validate_solvents(yaml_content.get('solvents', {}))
        self._db.systems = self._validate_systems(yaml_content.get('systems', {}))

        # Validate protocols
        self._mcmc_moves = self._validate_mcmc_moves(yaml_content)
        self._samplers = self._validate_samplers(yaml_content)
        self._protocols = self._validate_protocols(yaml_content.get('protocols', {}))

        # Validate experiments
        self._parse_experiments(yaml_content)

    def run_experiments(self):
        """
        Set up and run all the Yank experiments.

        See Also
        --------
        Experiment
        """
        # Throw exception if there are no experiments
        if len(self._experiments) == 0:
            raise YamlParseError('No experiments specified!')

        # Setup and run all experiments with paths relative to the script directory.
        with moltools.utils.temporary_cd(self._script_dir):
            self._check_resume()
            self._setup_experiments()
            self._generate_experiments_protocols()

            # Find all the experiments to distribute among mpicomms.
            all_experiments = list(self._expand_experiments())

            # Cycle between experiments every switch_experiment_interval iterations
            # until all of them are done.
            while len(all_experiments) > 0:
                # Allocate the MPI processes to the experiments that still have to be completed.
                group_size = self._get_experiment_mpi_group_size(all_experiments)
                if group_size is None:
                    completed = [False] * len(all_experiments)
                    for exp_index, exp in enumerate(all_experiments):
                        completed[exp_index] = self._run_experiment(exp)
                else:
                    completed = mpi.distribute(self._run_experiment,
                                               distributed_args=all_experiments,
                                               group_size=group_size,
                                               send_results_to='all')

                # Remove any completed experiments, releasing possible parallel resources
                # to be reused. Evaluate in reverse order to avoid shuffling indices.
                for exp_index in range(len(all_experiments)-1, -1, -1):
                    if completed[exp_index]:
                        all_experiments.pop(exp_index)

    def build_experiments(self):
        """
        Generator to configure, build, and yield an experiment

        Yields
        ------
        Experiment
        """
        # Throw exception if there are no experiments
        if len(self._experiments) == 0:
            raise YamlParseError('No experiments specified!')

        # Setup and iterate over all experiments with paths relative to the script directory
        with moltools.utils.temporary_cd(self._script_dir):
            self._check_resume()
            self._setup_experiments()
            self._generate_experiments_protocols()
            for experiment_path, combination in self._expand_experiments():
                yield self._build_experiment(experiment_path, combination)

    def setup_experiments(self):
        """
        Set up all systems required for the Yank experiments without running them.
        """
        # All paths must be relative to the script directory
        with moltools.utils.temporary_cd(self._script_dir):
            self._check_resume(check_experiments=False)
            self._setup_experiments()

    def status(self):
        """Iterate over the status of all experiments in dictionary form.

        The status of each experiment is set to "completed" if both phases
        in the experiments have been completed, "pending" if they are both
        pending, and "ongoing" otherwise.

        Yields
        ------
        experiment_status : namedtuple
            The status of the experiment. It contains the following fields:

            name : str
                The name of the experiment.
            status : str
                One between "completed", "ongoing", or "pending".
            number_of_iterations : int
                The total number of iteration set for this experiment.
            job_id : int or None
                If njobs is specified, this includes the job id associated
                to this experiment.
            phases : dict
                phases[phase_name] is a namedtuple describing the status
                of phase ``phase_name``. The namedtuple has two fields:
                ``iteration`` and ``status``.

        """
        # TODO use Python 3.6 namedtuple syntax when we drop Python 3.5 support.
        PhaseStatus = collections.namedtuple('PhaseStatus', [
            'status',
            'iteration'
        ])
        ExperimentStatus = collections.namedtuple('ExperimentStatus', [
            'name',
            'status',
            'phases',
            'number_of_iterations',
            'job_id'
        ])

        for experiment_idx, (exp_path, exp_description) in enumerate(self._expand_experiments()):
            # Determine the final number of iterations for this experiment.
            number_of_iterations = self._get_experiment_number_of_iterations(exp_description)

            # Determine the phases status.
            phases = collections.OrderedDict()
            for phase_nc_path in self._get_nc_file_paths(exp_path, exp_description):

                # Determine the status of the phase.
                try:
                    phase_status = AlchemicalPhase.read_status(phase_nc_path)
                except FileNotFoundError:
                    iteration = None
                    phase_status = 'pending'
                else:
                    iteration = phase_status.iteration
                    if _is_phase_completed(phase_status, number_of_iterations):
                        phase_status = 'completed'
                    else:
                        phase_status = 'ongoing'
                phase_name = os.path.splitext(os.path.basename(phase_nc_path))[0]
                phases[phase_name] = PhaseStatus(status=phase_status, iteration=iteration)

            # Determine the status of the whole experiment.
            phase_statuses = [phase.status for phase in phases.values()]
            if phase_statuses[0] == phase_statuses[1]:
                # This covers the completed and pending status.
                exp_status = phase_statuses[0]
            else:
                exp_status = 'ongoing'

            # Determine jobid if requested.
            if self._n_jobs is not None:
                job_id = experiment_idx % self._n_jobs + 1
            else:
                job_id = None

            yield ExperimentStatus(name=exp_path, status=exp_status,
                                   phases=phases, job_id=job_id,
                                   number_of_iterations=number_of_iterations)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def verbose(self):
        """bool: the log verbosity."""
        return self._options['verbose']

    @verbose.setter
    def verbose(self, new_verbose):
        self._options['verbose'] = new_verbose
        utils.config_root_logger(self._options['verbose'], log_file_path=None)

    @property
    def output_dir(self):
        """The path to the main output directory."""
        return self._options['output_dir']

    @output_dir.setter
    def output_dir(self, new_output_dir):
        self._options['output_dir'] = new_output_dir
        self._db.setup_dir = os.path.join(new_output_dir, self.setup_dir)

    @property
    def setup_dir(self):
        """The path to the setup files directory relative to the output folder.."""
        return self._options['setup_dir']

    @setup_dir.setter
    def setup_dir(self, new_setup_dir):
        self._options['setup_dir'] = new_setup_dir
        self._db.setup_dir = os.path.join(self.output_dir, new_setup_dir)

    # --------------------------------------------------------------------------
    # Options handling
    # --------------------------------------------------------------------------

    def _determine_experiment_options(self, experiment):
        """Determine all the options required to build the experiment.

        Merge the options specified in the experiment section with the ones
        in the options section, and divide them into several dictionaries to
        feed to different main classes necessary to create an AlchemicalPhase.

        Parameters
        ----------
        experiment : dict
            The dictionary encoding the experiment.

        Returns
        -------
        experiment_options : dict
            The ExperimentBuilder experiment options. This does not contain
            the general ExperimentBuilder options that are accessible through
            self._options.
        phase_options : dict
            The options to pass to the AlchemicalPhaseFactory constructor.
        alchemical_region_options : dict
            The options to pass to AlchemicalRegion.
        alchemical_factory_options : dict
            The options to pass to AlchemicalFactory.

        """
        # First discard general options.
        options = {name: value for name, value in self._options.items()
                   if name not in self.GENERAL_DEFAULT_OPTIONS}

        # Then update with specific experiment options.
        options.update(self._validate_options(experiment.get('options', {}),
                                              validate_general_options=False))

        def _filter_options(reference_options):
            return {name: value for name, value in options.items()
                    if name in reference_options}

        experiment_options = _filter_options(self.EXPERIMENT_DEFAULT_OPTIONS)
        phase_options = _filter_options(AlchemicalPhaseFactory.DEFAULT_OPTIONS)
        alchemical_region_options = _filter_options(mmtools.alchemy._ALCHEMICAL_REGION_ARGS)
        alchemical_factory_options = _filter_options(utils.get_keyword_args(
            mmtools.alchemy.AbsoluteAlchemicalFactory.__init__))

        return (experiment_options, phase_options,
                alchemical_region_options, alchemical_factory_options)

    # --------------------------------------------------------------------------
    # Combinatorial expansion
    # --------------------------------------------------------------------------

    def _expand_molecules(self, yaml_content):
        """Expand combinatorial molecules.

        Generate new YAML content with no combinatorial molecules. The new content
        is identical to the old one but combinatorial molecules are substituted by
        the description of all the non-combinatorial molecules that they generate.
        Moreover, systems that use combinatorial molecules are updated with the new
        molecules ids.

        Parameters
        ----------
        yaml_content : dict
            The YAML content as returned by yaml.load().

        Returns
        -------
        expanded_content : dict
            The new YAML content with combinatorial molecules expanded.

        """
        expanded_content = copy.deepcopy(yaml_content)

        if 'molecules' not in expanded_content:
            return expanded_content

        # First substitute all 'select: all' with the correct combination of indices
        for comb_mol_name, comb_molecule in expanded_content['molecules'].items():
            if 'select' in comb_molecule and comb_molecule['select'] == 'all':
                # Get the number of models in the file
                try:
                    extension = os.path.splitext(comb_molecule['filepath'])[1][1:]  # remove dot
                except KeyError:
                    # Trap an error caused by missing a filepath from the combinatorial expansion
                    # This will ultimately fail cerberus validation, but will be easier to debug than
                    # random Python error
                    continue
                with moltools.utils.temporary_cd(self._script_dir):
                    if extension == 'pdb':
                        n_models = PDBFile(comb_molecule['filepath']).getNumFrames()

                    elif extension == 'csv' or extension == 'smiles':
                        n_models = len(pipeline.read_csv_lines(comb_molecule['filepath'], lines='all'))

                    elif extension == 'sdf' or extension == 'mol2':
                        if not utils.is_openeye_installed(oetools=('oechem',)):
                            err_msg = 'Molecule {}: Cannot "select" from {} file without OpenEye toolkit'
                            raise RuntimeError(err_msg.format(comb_mol_name, extension))
                        n_models = len(utils.load_oe_molecules(comb_molecule['filepath']))

                    else:
                        raise YamlParseError('Molecule {}: Cannot "select" from {} file'.format(
                            comb_mol_name, extension))

                # Substitute select: all with list of all models indices to trigger combinations
                comb_molecule['select'] = utils.CombinatorialLeaf(range(n_models))

        # Expand molecules and update molecule ids in systems
        expanded_content = utils.CombinatorialTree(expanded_content)
        update_nodes_paths = [('systems', '*', 'receptor'), ('systems', '*', 'ligand'),
                              ('systems', '*', 'solute')]
        expanded_content = expanded_content.expand_id_nodes('molecules', update_nodes_paths)

        return expanded_content

    def _expand_systems(self, yaml_content):
        """Expand combinatorial systems.

        Generate new YAML content with no combinatorial systems. The new content
        is identical to the old one but combinatorial systems are substituted by
        the description of all the non-combinatorial systems that they generate.
        Moreover, the experiments that use combinatorial systems are updated with
        the new system ids.

        Molecules must be already expanded when calling this function.

        Parameters
        ----------
        yaml_content : dict
            The YAML content as returned by _expand_molecules().

        Returns
        -------
        expanded_content : dict
            The new YAML content with combinatorial systems expanded.

        """
        expanded_content = copy.deepcopy(yaml_content)

        if 'systems' not in expanded_content:
            return expanded_content

        # Check if we have a sequence of experiments or a single one
        try:
            if isinstance(expanded_content['experiments'], list):  # sequence of experiments
                experiment_names = expanded_content['experiments']
            else:
                experiment_names = ['experiments']
        except KeyError:
            experiment_names = []

        # Expand molecules and update molecule ids in experiments
        expanded_content = utils.CombinatorialTree(expanded_content)
        update_nodes_paths = [(e, 'system') for e in experiment_names]
        expanded_content = expanded_content.expand_id_nodes('systems', update_nodes_paths)

        return expanded_content

    def _expand_experiments(self):
        """Generates all possible combinations of experiment.

        Each generated experiment is uniquely named. If job_id and n_jobs are
        set, this returns only the experiments assigned to this particular job.

        Yields
        ------
        experiment_path : str
            A unique path where to save the experiment output files relative to
            the main output directory specified by the user in the options.
        combination : dict
            The dictionary describing a single experiment.

        """
        # We need to distribute experiments among jobs, but different
        # experiments sectiona may have a different number of combinations,
        # so we need to count them.
        experiment_id = 0

        output_dir = ''
        for exp_name, experiment in self._experiments.items():
            if len(self._experiments) > 1:
                output_dir = exp_name

            # Loop over all combinations
            for name, combination in experiment.named_combinations(separator='_', max_name_length=50):
                # Both self._job_id and self._job_id-1 work (self._job_id is 1-based),
                # but we use the latter just because it makes it easier to identify in
                # advance which job ids are associated to which experiments.
                if self._job_id is None or experiment_id % self._n_jobs == self._job_id-1:
                    yield os.path.join(output_dir, name), combination
                experiment_id += 1

    # --------------------------------------------------------------------------
    # Parsing and syntax validation
    # --------------------------------------------------------------------------

    # Shared schema for leap parameters. Molecules, solvents and systems use it.
    # Simple strings in "parameters" are converted to list of strings.
    _LEAP_PARAMETERS_DEFAULT_SCHEMA = yaml.load("""
    leap:
        required: no
        type: dict
        default_setter: no_parameters
        schema:
            parameters:
                type: list
                coerce: single_str_to_list
                schema:
                    type: string
    """)

    @classmethod
    def _validate_options(cls, options, validate_general_options):
        """Validate molecules syntax.

        Parameters
        ----------
        options : dict
            A dictionary with the options to validate.
        validate_general_options : bool
            If False only the options that can be specified in the
            experiment section are validated.

        Returns
        -------
        validated_options : dict
            The validated options.

        Raises
        ------
        YamlParseError
            If the syntax for any option is not valid.

        """
        template_options = cls.EXPERIMENT_DEFAULT_OPTIONS.copy()
        template_options.update(AlchemicalPhaseFactory.DEFAULT_OPTIONS)
        template_options.update(mmtools.alchemy._ALCHEMICAL_REGION_ARGS)
        template_options.update(utils.get_keyword_args(
            mmtools.alchemy.AbsoluteAlchemicalFactory.__init__))

        if validate_general_options is True:
            template_options.update(cls.GENERAL_DEFAULT_OPTIONS.copy())

        # Remove options that are not supported.
        template_options.pop('alchemical_atoms')  # AlchemicalRegion
        template_options.pop('alchemical_bonds')
        template_options.pop('alchemical_angles')
        template_options.pop('alchemical_torsions')
        template_options.pop('switch_width')  # AbsoluteAlchemicalFactory

        def check_anisotropic_cutoff(cutoff):
            if cutoff == 'auto':
                return cutoff
            else:
                return utils.quantity_from_string(cutoff, unit.angstroms)

        def check_processes_per_experiment(processes_per_experiment):
            if processes_per_experiment == 'auto' or processes_per_experiment is None:
                return processes_per_experiment
            return int(processes_per_experiment)

        special_conversions = {'constraints': to_openmm_app,
                               'default_number_of_iterations': schema.to_integer_or_infinity_coercer,
                               'anisotropic_dispersion_cutoff': check_anisotropic_cutoff,
                               'processes_per_experiment': check_processes_per_experiment}

        # Validate parameters.
        try:
            validated_options = utils.validate_parameters(options, template_options, check_unknown=True,
                                                          process_units_str=True, float_to_int=True,
                                                          special_conversions=special_conversions)
        except (TypeError, ValueError) as e:
            raise YamlParseError(str(e))

        # Overwrite defaults.
        defaults_to_overwrite = {
            # With the analytical dispersion correction for alchemical atoms,
            # the computation of the energy matrix becomes super slow.
            'disable_alchemical_dispersion_correction': True,
        }
        for option, default_value in defaults_to_overwrite.items():
            if option not in validated_options:
                validated_options[option] = default_value

        return validated_options

    @classmethod
    def _validate_molecules(cls, molecules_description):
        """Validate molecules syntax.

        Parameters
        ----------
        molecules_description : dict
            A dictionary representing molecules.

        Returns
        -------
        validated_molecules : dict
            The validated molecules description.

        Raises
        ------
        YamlParseError
            If the syntax for any molecule is not valid.

        """
        regions_schema_yaml = '''
        regions:
            type: dict
            required: no
            keyschema:
                type: string
            valueschema:
                anyof:
                    - type: list
                      validator: positive_int_list
                    - type: string
                    - type: integer
                      min: 0
        '''
        regions_schema = yaml.load(regions_schema_yaml)
        # Define the LEAP schema
        leap_schema = cls._LEAP_PARAMETERS_DEFAULT_SCHEMA
        # Setup the common schema across ALL molecules
        common_molecules_schema = {**leap_schema, **regions_schema}
        # Setup the small molecules schemas
        small_molecule_schema_yaml = """
        smiles:
            type: string
            excludes: [name, filepath]
            required: yes
        name:
            type: string
            excludes: [smiles, filepath]
            required: yes
        filepath:
            type: string
            excludes: [smiles, name]
            required: yes
            validator: [is_small_molecule, file_exists]
        openeye:
            required: no
            type: dict
            schema:
                quacpac:
                    required: yes
                    type: string
                    allowed: [am1-bcc]
        antechamber:
            required: no
            type: dict
            schema:
                charge_method:
                    required: yes
                    type: string
                    nullable: yes
        select:
            required: no
            dependencies: filepath
            validator: int_or_all_string
        """
        # Build small molecule Epik by hand as dict since we are fetching from another source
        epik_schema = schema.generate_signature_schema(moltools.schrodinger.run_epik,
                                                       update_keys={'select': {'required': False, 'type': 'integer'}},
                                                       exclude_keys=['extract_range'])
        epik_schema = {'epik': {
            'required': False,
            'type': 'dict',
            'schema': epik_schema
            }
        }
        small_molecule_schema = {**yaml.load(small_molecule_schema_yaml), **epik_schema, **common_molecules_schema}

        # Peptide schema has some keys excluded from small_molecule checks
        peptide_schema_yaml = """
        filepath:
            required: yes
            type: string
            validator: [is_peptide, file_exists]
        select:
            required: no
            dependencies: filepath
            validator: int_or_all_string
        strip_protons:
            required: no
            type: boolean
            dependencies: filepath
        pdbfixer:
            required: no
            dependencies: filepath
        openeye:
            required: no
            excludes: filepath
        antechamber:
            required: no
            excludes: filepath
        epik:
            required: no
            excludes: filepath
        """
        peptide_schema = {**yaml.load(peptide_schema_yaml), **common_molecules_schema}

        validated_molecules = molecules_description.copy()
        # Schema validation
        for molecule_id, molecule_descr in molecules_description.items():
            small_molecule_validator = schema.YANKCerberusValidator(small_molecule_schema)
            peptide_validator = schema.YANKCerberusValidator(peptide_schema)
            # Test for small molecule
            if small_molecule_validator.validate(molecule_descr):
                validated_molecules[molecule_id] = small_molecule_validator.document
            # Test for peptide
            elif peptide_validator.validate(molecule_descr):
                validated_molecules[molecule_id] = peptide_validator.document
            else:
                # Both failed, lets figure out why
                # Check the is peptide w/ only excluded errors
                if (cerberus.errors.EXCLUDES_FIELD in peptide_validator._errors and
                        peptide_validator.document_error_tree['filepath'] is None):
                    error = ("Molecule {} appears to be a peptide, but uses items exclusive to small molecules:\n"
                             "Please change either the options to peptide-only entries, or your molecule to a "
                             "small molecule.\n"
                             "====== Peptide Schema =====\n"
                             "{}\n"
                             "===========================\n")
                    error = error.format(molecule_id, yaml.dump(peptide_validator.errors))
                # We don't know exactly what went wrong, run both error blocks
                else:
                    error = ("Molecule {} failed to validate against one of the following schemes\n"
                             "Please check the following schemes for errors:\n"
                             "===========================\n"
                             "== Small Molecule Schema ==\n"
                             "{}\n"
                             "===========================\n\n"  # Blank line
                             "===========================\n"
                             "====== Peptide Schema =====\n"
                             "{}\n"
                             "===========================\n")
                    error = error.format(molecule_id, yaml.dump(small_molecule_validator.errors),
                                         yaml.dump(peptide_validator.errors))
                raise YamlParseError(error)

            # Check OpenEye charges - antechamber consistency
            if 'openeye' in validated_molecules[molecule_id]:
                if 'antechamber' not in validated_molecules[molecule_id]:
                    raise YamlParseError('Cannot specify openeye charges without antechamber')
                if validated_molecules[molecule_id]['antechamber']['charge_method'] is not None:
                    raise YamlParseError('Antechamber charge_method must be "null" to read '
                                         'OpenEye charges')

            # Convert epik "select" to "extract_range" which is accepted by run_epik()
            try:
                extract_range = validated_molecules[molecule_id]['epik'].pop('select')
                validated_molecules[molecule_id]['epik']['extract_range'] = extract_range
            except (AttributeError, KeyError):
                pass

        return validated_molecules

    @classmethod
    def _validate_solvents(cls, solvents_description):
        """Validate molecules syntax.

        Parameters
        ----------
        solvents_description : dict
            A dictionary representing solvents.

        Returns
        -------
        validated_solvents : dict
            The validated solvents description.

        Raises
        ------
        YamlParseError
            If the syntax for any solvent is not valid.

        """
        openmm_nonbonded_strings = get_openmm_nonbonded_methods_strings()
        mapped_openmm_nonbonded_methods = {nb_method: to_openmm_app(nb_method) for
                                           nb_method in openmm_nonbonded_strings}
        explicit_strings = get_openmm_explicit_nb_method_strings()
        mapped_explicit_methods = [mapped_openmm_nonbonded_methods[method] for method in explicit_strings]
        all_valid_explicit = explicit_strings + mapped_explicit_methods
        implicit_strings = get_openmm_implicit_nb_method_strings()
        mapped_implicit_methods = [mapped_openmm_nonbonded_methods[method] for method in implicit_strings]
        all_valid_implicit = implicit_strings + mapped_implicit_methods

        def is_supported_solvent_model(field, solvent_model, error):
            """Check that solvent model name is supported."""
            if solvent_model not in pipeline._OPENMM_LEAP_SOLVENT_MODELS_MAP:
                error(field, "{} not in the known solvent models map!".format(solvent_model))

        def ionic_strength_if_explicit_else_none(document, default="0.0*molar"):
            """Set the ionic strength IFF solvent model is explicit"""
            if document['nonbonded_method'] in all_valid_explicit:
                return default
            else:
                return None

        def solvent_model_if_explicit_else_none(document, default='tip4pew'):
            """Set the solvent_model IFF solvent model is explicit"""

            if document['nonbonded_method'] in all_valid_explicit:
                return default
            else:
                return None

        def to_openmm_app_unless_none(input_string):
            """
            Extension method of the :func:`to_openmm_app` method which returns None if None is given
            Primarily used by the schema validators

            Parameters
            ----------
            input_string : str or None
                Method name of openmm.app to fetch

            Returns
            -------
            method : Method of openmm.app or None
                Returns openmm.app.{input_string}
            """
            return to_openmm_app(input_string) if input_string is not None else None

        def to_unit_unless_none_coercer(compatible_units):
            """
            Extension to the :func:`utils.to_unit_coercer` method which also allows a None object to be set

            See call to :func:`utils.to_unit_coercer` for call
            """
            unit_validator = schema.to_unit_coercer(compatible_units)

            def _to_unit_unless_none(input_quantity):
                if input_quantity is None:
                    return None
                else:
                    return unit_validator(input_quantity)
            return _to_unit_unless_none

        # Define solvents Schema
        # Create the basic solvent schema, ignoring things which have a dependency
        # Some keys we manually tweak
        base_solvent_schema = schema.generate_signature_schema(AmberPrmtopFile.createSystem,
                                                               exclude_keys=['nonbonded_method'])
        implicit_solvent_default_schema = {'implicit_solvent': base_solvent_schema.pop('implicit_solvent')}
        rigid_water_default_schema = {'rigid_water': base_solvent_schema.pop('rigid_water')}
        nonbonded_cutoff_default_schema = {'nonbonded_cutoff': base_solvent_schema.pop('nonbonded_cutoff')}
        # Cerberus Schema Processing hierarchy:
        # Input Value -> {default value} -> default setter -> coerce -> allowed/validate
        #  Handle the use cases for special keys
        # nonbonded_method
        base_solvent_schema['nonbonded_method'] = {
            'allowed': [value for _, value in mapped_openmm_nonbonded_methods.items()],  # Only use valid openmm methods
            'coerce': to_openmm_app,  # Cast the string first to valid method
            'required': True,  # This must be set
            'default': openmm_nonbonded_strings[0],  # Choose a default mapping
        }
        # Explicit solvent keys, populate required and dependencies in batch
        explicit_only_keys = {
            'clearance': {
                'type': 'quantity',
                'coerce': schema.to_unit_coercer(unit.angstrom),
            },
            'solvent_model': {
                'type': 'string',
                'nullable': True,
                'validator': is_supported_solvent_model,
                'default_setter': solvent_model_if_explicit_else_none
            },
            'positive_ion': {
                'type': 'string'
            },
            'negative_ion': {
                'type': 'string'
            },
            'ionic_strength': {
                'type': 'quantity',
                'coerce': to_unit_unless_none_coercer(unit.molar),
                'default_setter': ionic_strength_if_explicit_else_none,
                'nullable': True
            },
            **nonbonded_cutoff_default_schema
        }
        # Batch the explicit dependencies
        for key in explicit_only_keys.keys():
            explicit_only_keys[key]['dependencies'] = {'nonbonded_method': [mapped_openmm_nonbonded_methods[value] for
                                                                            value in
                                                                            get_openmm_explicit_nb_method_strings()]}
            explicit_only_keys[key]['required'] = False

        # Implicit solvent keys
        # Input Value -> {default value} -> default setter -> coerce -> allowed/validate
        implicit_only_keys = {**implicit_solvent_default_schema}
        implicit_only_keys['implicit_solvent']['coerce'] = to_openmm_app_unless_none
        implicit_only_keys['implicit_solvent']['dependencies'] = {'nonbonded_method': all_valid_implicit}
        # Batch the implicit dependencies
        for key in implicit_only_keys.keys():
            implicit_only_keys[key]['dependencies'] = {'nonbonded_method': [mapped_openmm_nonbonded_methods[value] for
                                                                            value in
                                                                            get_openmm_implicit_nb_method_strings()]}
            implicit_only_keys[key]['required'] = False

        # Vacuum solvent is implicitly defined when the `NoCutoff` scheme is selected and `implicit_solvent` is None

        # Finally, stitch the schema together
        solvent_schema = {**base_solvent_schema, **explicit_only_keys, **implicit_only_keys,
                          **rigid_water_default_schema, **cls._LEAP_PARAMETERS_DEFAULT_SCHEMA}

        solvent_validator = schema.YANKCerberusValidator(solvent_schema)

        validated_solvents = solvents_description.copy()

        # Schema validation
        for solvent_id, solvent_descr in solvents_description.items():
            if solvent_validator.validate(solvent_descr):
                validated_solvents[solvent_id] = solvent_validator.document
            else:
                error = "Solvent '{}' did not validate! Check the schema error below for details\n{}"
                raise YamlParseError(error.format(solvent_id, yaml.dump(solvent_validator.errors)))

        return validated_solvents

    @staticmethod
    def _validate_protocols(protocols_description):
        """Validate protocols.

        Parameters
        ----------
        protocols_description : dict
            A dictionary representing protocols.

        Returns
        -------
        validated_protocols : dict
            The validated protocols description.

        Raises
        ------
        YamlParseError
            If the syntax for any protocol is not valid.

        """
        def sort_protocol(protocol):
            """Reorder phases in dictionary to have complex/solvent1 first."""
            sortables = [('complex', 'solvent'), ('solvent1', 'solvent2')]
            for sortable in sortables:
                # Phases names must be unambiguous, they can't contain both names
                phase1 = [(k, v) for k, v in protocol.items()
                          if (sortable[0] in k and sortable[1] not in k)]
                phase2 = [(k, v) for k, v in protocol.items()
                          if (sortable[1] in k and sortable[0] not in k)]

                # Phases names must be unique
                if len(phase1) == 1 and len(phase2) == 1:
                    return collections.OrderedDict([phase1[0], phase2[0]])

            # Could not find any sortable
            raise YamlParseError('Non-ordered phases must contain either "complex" and "solvent" '
                                 'OR "solvent1" and "solvent2", the phase names must also be non-ambiguous so each '
                                 'keyword can only appear in a single phase, not multiple.')

        def validate_string_auto(field, value, error):
            if isinstance(value, str) and value != 'auto':
                error(field, "Only the exact string 'auto' is accepted as a string argument, not {}.".format(value))

        def cast_quantity_strings(value):
            """Take an object and try to cast quantity strings to quantity, otherwise return object"""
            if isinstance(value, str):
                value = utils.quantity_from_string(value)
            return value

        def validate_required_entries_dict(field, value, error):
            """Ensure the required entries are in the dict, string is checked by a separate validator"""
            if isinstance(value, dict) or isinstance(value, collections.OrderedDict):
                if 'lambda_sterics' not in value.keys() or 'lambda_electrostatics' not in value.keys():
                    error(field, "Missing required keys lambda_sterics and/or lambda_electrostatics")

        def validate_lambda_min_max(field, value, error):
            """Ensure keys which are lambda values are in fact between 0 and 1"""
            base_error = "Entries with a 'lambda_' must be a float between 0 and 1, inclusive. Values {} are not."
            collected_bad_values = []
            if "lambda_" in field:
                for single_value in value:
                    if not (isinstance(single_value, float) and 0 <= single_value <= 1.0):
                        collected_bad_values.append(single_value)
            if len(collected_bad_values):
                error(field, base_error.format(collected_bad_values))

        # Define protocol Schema
        # Note: Cannot cleanly do yaml.dump(v.errors) for nested `schema`/`*of` logic from Cerberus until its 1.2
        protocol_value_schema = {
            'alchemical_path': {  # The only literal key
                'required': True,  # Key is required
                'type': ['string', 'dict'],  # Must be a string or dictionary
                # Use this to check the string value until Cerberus 1.2 for `oneof`
                # Check string with validate_string_auto, pass other values to next validator
                # Check the dict values with validate_required_entries_doct, ignore other values
                'validator': [validate_string_auto, validate_required_entries_dict],
                'keyschema': {  # Validate the keys of this sub-dictionary against this schema
                    'type': 'string'
                },
                'valueschema': {  # Validate the values of this sub-dictionary against this schema
                    'type': 'list',  # They must be a list (dont accept single values)
                    # Check if it has the `lambda_` string that its a float within [0,1]
                    'validator': validate_lambda_min_max,
                    'schema': {
                        'type': ['float', 'quantity'],  # Ensure the output type is float or quantity
                        # Cast strings to quantity. Everything else had to validate down to this point
                        'coerce': cast_quantity_strings,
                    }
                }
            }
        }

        # Validate the top level keys (cannot be done with Cerberus)
        def validate_protocol_keys_and_values(protocol_id, protocol):
            # Ensure the protocol is 2 keys
            if len(protocol) != 2:
                raise YamlParseError('Protocol {} must only have two phases, found {}'.format(protocol_id,
                                                                                              len(protocol)))
            # Ensure the protocol keys are in fact strings
            keys_not_strings = []
            key_string_error = 'Protocol {} has keys which are not strings: '.format(protocol_id)
            for key in protocol.keys():
                if not isinstance(key, str):
                    keys_not_strings.append(key)
            if len(keys_not_strings) > 0:
                # The join(list_comprehension) forces invalid keys to a string so the join command works
                raise YamlParseError(key_string_error + ', '.join(['{}'.format(key) for key in keys_not_strings]))
            # Check for ordered dict or the sorted keys
            if not isinstance(protocol, collections.OrderedDict):
                protocol = sort_protocol(protocol)
            # Now user cerberus to validate the alchemical path part
            errored_phases = []
            for phase_key, phase_entry in protocol.items():
                phase_validator = schema.YANKCerberusValidator(protocol_value_schema)
                # test the phase
                if phase_validator.validate(phase_entry):
                    protocol[phase_key] = phase_validator.document
                else:
                    # collect the errors
                    errored_phases.append([phase_key, yaml.dump(phase_validator.errors)])
            if len(errored_phases) > 0:
                # Throw error
                error = "Protocol {} failed because one or more of the phases did not validate, see the errors below " \
                        "for more information.\n".format(protocol_id)
                for phase_id, phase_error in errored_phases:
                    error += "Phase: {}\n----\n{}\n====\n".format(phase_id, phase_error)
                raise YamlParseError(error)
            # Finally return if everything is fine
            return protocol

        validated_protocols = protocols_description.copy()
        # Schema validation
        for protocol_id, protocol_descr in protocols_description.items():
            # Error is raised in the function
            validated_protocols[protocol_id] = validate_protocol_keys_and_values(protocol_id, protocol_descr)

        return validated_protocols

    def _validate_systems(self, systems_description):
        """Validate systems.

        Receptors, ligands, and solvents must be already loaded. If they are not
        found an exception is raised.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        Returns
        -------
        validated_systems : dict
            The validated systems description.

        Raises
        ------
        YamlParseError
            If the syntax for any experiment is not valid.

        """

        def generate_region_clash_validator(system_descr, mol_class1, mol_class2=None):
            """
            Check that the regions have non clashing names by looking at regions in each molecule for a given
            system, this is generated at run time per-system
            """
            base_error = ("Cannot resolve molecular regions! "
                          "Found regions(s) clashing for a {}/{} pair!".format(mol_class1, mol_class2))
            error_collection = []
            mol_description1 = self._db.molecules.get(system_descr.get(mol_class1, ''), {})
            mol_description2 = self._db.molecules.get(
                system_descr.get(mol_class2, ''), {}) if mol_class2 is not None else {}
            # Fetch the regions or an empty dict
            regions_1_names = mol_description1.get('regions', {}).keys()
            regions_2_names = mol_description2.get('regions', {}).keys()
            for region_1_name in regions_1_names:
                if region_1_name in regions_2_names:
                    error_collection.append("\n- Region {}".format(region_1_name))

            def _region_clash_validator(field, value, error):
                if len(error_collection) > 0:
                    error(field, base_error + ''.join(error_collection))
            return _region_clash_validator

        def generate_is_pipeline_solvent_with_receptor_validator(system_descr, cross_id):

            def _is_pipeline_solvent_with_receptor(field, solvent_id, error):
                if cross_id in system_descr:
                    solvent = self._db.solvents.get(solvent_id, {})
                    if (solvent.get('nonbonded_method') != openmm.app.NoCutoff and
                        'clearance' not in solvent):
                        error(field, 'Explicit solvent {} does not specify clearance.'.format(solvent_id))
            return _is_pipeline_solvent_with_receptor

        # Define systems Schema
        systems_schema_yaml = """
        # DSL Schema
        ligand_dsl:
            required: no
            type: string
            dependencies: [phase1_path, phase2_path]
        solvent_dsl:
            required: no
            type: string
            dependencies: [phase1_path, phase2_path]

        # Phase paths
        phase1_path:
            required: no
            type: list
            schema:
                type: string
                validator: file_exists
            dependencies: phase2_path
            validator: supported_system_files
        phase2_path:
            required: no
            type: list
            schema:
                type: string
                validator: file_exists
            validator: supported_system_files
        gromacs_include_dir:
            required: no
            type: string
            dependencies: [phase1_path, phase2_path]
            validator: directory_exists

        # Solvents
        solvent:
            required: no
            type: string
            excludes: [solvent1, solvent2]
            allowed: SOLVENT_IDS_POPULATED_AT_RUNTIME
            validator: PIPELINE_SOLVENT_DETERMINED_AT_RUNTIME_WITH_RECEPTOR
            oneof:
                - dependencies: [phase1_path, phase2_path]
                - dependencies: [receptor, ligand]
        solvent1:
            required: no
            type: string
            excludes: solvent
            allowed: SOLVENT_IDS_POPULATED_AT_RUNTIME
            validator: PIPELINE_SOLVENT_DETERMINED_AT_RUNTIME_WITH_SOLUTE
            oneof:
                - dependencies: [phase1_path, phase2_path, solvent2]
                - dependencies: [solute, solvent2]
        solvent2:
            required: no
            type: string
            excludes: solvent
            allowed: SOLVENT_IDS_POPULATED_AT_RUNTIME
            validator: PIPELINE_SOLVENT_DETERMINED_AT_RUNTIME_WITH_SOLUTE
            oneof:
                - dependencies: [phase1_path, phase2_path, solvent1]
                - dependencies: [solute, solvent1]

        # Automatic pipeline
        receptor:
            required: no
            type: string
            dependencies: [ligand, solvent]
            allowed: MOLECULE_IDS_POPULATED_AT_RUNTIME
            excludes: [solute, phase1_path, phase2_path]
            validator: REGION_CLASH_DETERMINED_AT_RUNTIME_WITH_LIGAND
        ligand:
            required: no
            type: string
            dependencies: [receptor, solvent]
            allowed: MOLECULE_IDS_POPULATED_AT_RUNTIME
            excludes: [solute, phase1_path, phase2_path]
        pack:
            # Technically requires receptor, but with default interjects itself even if receptor is not present
            required: no
            type: boolean
            default: no
        solute:
            required: no
            type: string
            allowed: MOLECULE_IDS_POPULATED_AT_RUNTIME
            dependencies: [solvent1, solvent2]
            validator: REGION_CLASH_DETERMINED_AT_RUNTIME
            excludes: [receptor, ligand]
        """
        # Load the YAML into a schema into dict format
        system_schema = yaml.load(systems_schema_yaml)
        # Add the LEAP schema
        leap_schema = self._LEAP_PARAMETERS_DEFAULT_SCHEMA
        # Handle dependencies
        # This does nothing in the case of phase1_path/phase2_path, but I wanted to leave this here in case
        # we decide to actually make these required. The problem is that because this has a `default` key, it inserts
        # itself into the scheme, but then fails if its a phase1_path and phase2_path set. So I silenced the line for
        # now if we decided to engineer this in later.
        # leap_schema['leap']['oneof'] = [{'dependencies': ['receptor', 'ligand']}, {'dependencies': 'solute'}]
        system_schema = {**system_schema, **leap_schema}
        # Handle the populations
        # Molecules
        for molecule_id_key in ['receptor', 'ligand', 'solute']:
            system_schema[molecule_id_key]['allowed'] = [str(key) for key in self._db.molecules.keys()]
        # Solvents
        for solvent_id_key in ['solvent', 'solvent1', 'solvent2']:
            system_schema[solvent_id_key]['allowed'] = [str(key) for key in self._db.solvents.keys()]

        def generate_runtime_schema_for_system(system_descr):
            new_schema = system_schema.copy()
            # Handle the validators
            # Spin up the region clash calculators
            new_schema['receptor']['validator'] = generate_region_clash_validator(system_descr, 'ligand', 'receptor')
            new_schema['solute']['validator'] = generate_region_clash_validator(system_descr, 'solute')
            # "solvent"
            new_schema['solvent']['validator'] = generate_is_pipeline_solvent_with_receptor_validator(system_descr,
                                                                                                      'receptor')
            # "solvent1" and "solvent2
            new_schema['solvent1']['validator'] = generate_is_pipeline_solvent_with_receptor_validator(system_descr,
                                                                                                       'solute')
            new_schema['solvent2']['validator'] = generate_is_pipeline_solvent_with_receptor_validator(system_descr,
                                                                                                       'solute')
            return new_schema

        validated_systems = systems_description.copy()
        # Schema validation
        for system_id, system_descr in systems_description.items():
            runtime_system_schema = generate_runtime_schema_for_system(system_descr)
            system_validator = schema.YANKCerberusValidator(runtime_system_schema)
            if system_validator.validate(system_descr):
                validated_systems[system_id] = system_validator.document
            else:
                error = "System '{}' did not validate! Check the schema error below for details\n{}"
                raise YamlParseError(error.format(system_id, yaml.dump(system_validator.errors)))
        return validated_systems

    @classmethod
    def _validate_mcmc_moves(cls, yaml_content):
        """Validate mcmc_moves section."""
        mcmc_move_descriptions = yaml_content.get('mcmc_moves', None)
        if mcmc_move_descriptions is None:
            return {}

        mcmc_move_schema = """
        mcmc_moves:
            keyschema:
                type: string
            valueschema:
                type: dict
                validator: is_mcmc_move_constructor
                keyschema:
                    type: string
        """
        mcmc_move_schema = yaml.load(mcmc_move_schema)

        mcmc_move_validator = schema.YANKCerberusValidator(mcmc_move_schema)
        if mcmc_move_validator.validate({'mcmc_moves': mcmc_move_descriptions}):
            validated_mcmc_moves = mcmc_move_validator.document
        else:
            error = "MCMC moves validation failed with:\n{}"
            raise YamlParseError(error.format(yaml.dump(mcmc_move_validator.errors)))
        return validated_mcmc_moves['mcmc_moves']

    def _validate_samplers(self, yaml_content):
        """Validate samplers section."""
        sampler_descriptions = yaml_content.get('samplers', None)
        if sampler_descriptions is None:
            return {}

        sampler_schema = """
        samplers:
            keyschema:
                type: string
            valueschema:
                type: dict
                validator: is_sampler_constructor
                allow_unknown: yes
                schema:
                    mcmc_moves:
                        type: string
                        allowed: {MCMC_MOVE_IDS}
        """.format(MCMC_MOVE_IDS=list(self._mcmc_moves.keys()))
        sampler_schema = yaml.load(sampler_schema)

        # Special case for "checkpoint" in online analysis
        # Handle 1-case where its set by user to something other than checkpoint and exclude it, then process if not it
        options_description = yaml_content.get('options', {})
        for sampler_description in sampler_descriptions:
            if not ("online_analysis_interval" in sampler_descriptions[sampler_description] and
                    sampler_descriptions[sampler_description]["online_analysis_interval"] != "checkpoint"):
                sampler_descriptions[sampler_description]["online_analysis_interval"] = \
                    options_description.get('checkpoint_interval',
                                            AlchemicalPhaseFactory.DEFAULT_OPTIONS['checkpoint_interval'])

        sampler_validator = schema.YANKCerberusValidator(sampler_schema)
        if sampler_validator.validate({'samplers': sampler_descriptions}):
            validated_samplers = sampler_validator.document
        else:
            error = "Samplers validation failed with:\n{}"
            raise YamlParseError(error.format(yaml.dump(sampler_validator.errors)))
        return validated_samplers['samplers']

    def _parse_experiments(self, yaml_content):
        """Validate experiments.

        Perform dry run and validate system, protocol and options of every combination.

        Systems and protocols must be already loaded. If they are not found, an exception
        is raised. Experiments options are validated as well.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        Raises
        ------
        YamlParseError
            If the syntax for any experiment is not valid.

        """

        def coerce_and_validate_options_here_against_existing(options):
            coerced_and_validated = {}
            errors = ""
            for option, value in options.items():
                try:
                    validated = ExperimentBuilder._validate_options({option:value}, validate_general_options=False)
                    coerced_and_validated = {**coerced_and_validated, **validated}
                except YamlParseError as yaml_err:
                    # Collect all errors.
                    coerced_and_validated[option] = value
                    errors += "\n{}".format(yaml_err)
            if errors != "":
                raise YamlParseError(errors)
            return coerced_and_validated

        def ensure_restraint_type_is_key(field, restraints_dict, error):
            if 'type' not in restraints_dict:
                error(field, "'type' must be a sub-key in the `restraints` dict")
            rest_type = restraints_dict.get('type')
            if rest_type is not None and not isinstance(rest_type, str):
                error(field, "Restraint type must be a string or None")

        # Check if there is a sequence of experiments or a single one.
        # We need to have a deterministic order of experiments so that
        # if we run multiple experiments in parallel, we won't have
        # multiple processes running the same one.
        try:
            if isinstance(yaml_content['experiments'], list):
                combinatorial_trees = [(exp_name, utils.CombinatorialTree(yaml_content[exp_name]))
                                       for exp_name in yaml_content['experiments']]
            else:
                combinatorial_trees = [('experiments', utils.CombinatorialTree(yaml_content['experiments']))]
            self._experiments = collections.OrderedDict(combinatorial_trees)
        except KeyError:
            self._experiments = collections.OrderedDict()
            return

        # Experiments Schema
        experiment_schema_yaml = """
        system:
            required: yes
            type: string
            allowed: SYSTEM_IDS_POPULATED_AT_RUNTIME
        protocol:
            required: yes
            type: string
            allowed: PROTOCOL_IDS_POPULATED_AT_RUNTIME
        sampler:
            required: no
            type: string
            allowed: SAMPLER_IDS_POPULATED_AT_RUNTIME
        options:
            required: no
            type: dict
            coerce: coerce_and_validate_options_here_against_existing
        restraint:
            required: no
            type: dict
            validator: is_restraint_constructor
            keyschema:
                type: string
        """

        experiment_schema = yaml.load(experiment_schema_yaml)
        # Populate valid types
        experiment_schema['system']['allowed'] = [str(key) for key in self._db.systems.keys()]
        experiment_schema['protocol']['allowed'] = [str(key) for key in self._protocols.keys()]
        experiment_schema['sampler']['allowed'] = [str(key) for key in self._samplers.keys()]
        # Options validator
        experiment_schema['options']['coerce'] = coerce_and_validate_options_here_against_existing

        experiment_validator = schema.YANKCerberusValidator(experiment_schema)
        # Schema validation
        for experiment_path, experiment_descr in self._expand_experiments():
            if not experiment_validator.validate(experiment_descr):
                error = "Experiment '{}' did not validate! Check the schema error below for details\n{}"
                raise YamlParseError(error.format(experiment_path, yaml.dump(experiment_validator.errors)))

    # --------------------------------------------------------------------------
    # File paths utilities
    # --------------------------------------------------------------------------

    def _get_experiment_dir(self, experiment_path):
        """Return the path to the directory where the experiment output files
        should be stored.

        Parameters
        ----------
        experiment_path : str
            The relative path w.r.t. the main experiments directory (determined
            through the options) of the experiment-specific subfolder.

        """
        return os.path.join(self._options['output_dir'], self._options['experiments_dir'],
                            experiment_path)

    def _get_nc_file_paths(self, experiment_path, experiment):
        """Return the paths to the two output .nc files of the experiment.

        Parameters
        ----------
        experiment_path : str
            The relative path w.r.t. the main experiments directory of the
            experiment-specific subfolder.
        experiment : dict
            The dictionary describing the single experiment.

        Returns
        -------
        list of str
            A list with the path of the .nc files for the two phases.

        """
        protocol = self._protocols[experiment['protocol']]
        experiment_dir = self._get_experiment_dir(experiment_path)
        # The order of the phases needs to be well defined for this to make sense.
        assert isinstance(protocol, collections.OrderedDict)
        return [os.path.join(experiment_dir, name + '.nc') for name in protocol.keys()]

    def _get_experiment_file_name(self, experiment_path):
        """Return the extension-less path to use for files referring to the experiment.

        Parameters
        ----------
        experiment_path : str
            The relative path w.r.t. the main experiments directory of the
            experiment-specific subfolder.

        """
        experiment_dir = self._get_experiment_dir(experiment_path)
        if experiment_path == '':
            log_file_name = 'experiments'
        else:
            # Normalize path to drop eventual final slash character.
            log_file_name = os.path.basename(os.path.normpath(experiment_path))
        return os.path.join(experiment_dir, log_file_name)

    def _get_generated_yaml_script_path(self, experiment_path):
        """Return the path for the generated single-experiment YAML script."""
        return self._get_experiment_file_name(experiment_path) + '.yaml'

    def _get_experiment_log_path(self, experiment_path):
        """Return the path for the experiment log file."""
        return self._get_experiment_file_name(experiment_path) + '.log'

    # --------------------------------------------------------------------------
    # Resuming
    # --------------------------------------------------------------------------

    def _check_resume_experiment(self, experiment_path, experiment):
        """Check if Yank output files already exist.

        Parameters
        ----------
        experiment_path : str
            The relative path w.r.t. the main experiments directory of the
            experiment-specific subfolder.
        experiment : dict
            The dictionary describing the single experiment.

        Returns
        -------
        bool
            True if NetCDF output files already exist, or if the protocol
            needs to be found automatically, and the generated YAML file
            exists, False otherwise.

        """
        # If protocol has automatic alchemical paths to generate,
        # check if generated YAML script exist.
        protocol = self._protocols[experiment['protocol']]
        automatic_phases = self._find_automatic_protocol_phases(protocol)
        yaml_generated_script_path = self._get_generated_yaml_script_path(experiment_path)
        if len(automatic_phases) > 0 and os.path.exists(yaml_generated_script_path):
            return True

        # Look for existing .nc files in the folder
        phase_paths = self._get_nc_file_paths(experiment_path, experiment)
        for phase_path in phase_paths:
            if os.path.isfile(phase_path) and os.path.getsize(phase_path) > 0:
                return True

        return False

    @mpi.on_single_node(0, sync_nodes=True)
    def _check_resume(self, check_setup=True, check_experiments=True):
        """Perform dry run to check if we are going to overwrite files.

        If we find folders that ExperimentBuilder should create we raise an exception
        unless resume_setup or resume_simulation are found, in which case we
        assume we need to use the existing files. We never overwrite files, the
        user is responsible to delete them or move them.

        It's important to check all possible combinations at the beginning to
        avoid interrupting the user simulation after few experiments.

        Parameters
        ----------
        check_setup : bool
            Check if we are going to overwrite setup files (default is True).
        check_experiments : bool
            Check if we are going to overwrite experiment files (default is True).

        Raises
        ------
        YamlParseError
            If files to write already exist and we resuming options are not set.

        """
        err_msg = ''

        for experiment_path, combination in self._expand_experiments():

            if check_experiments:
                resume_sim = self._options['resume_simulation']
                if not resume_sim and self._check_resume_experiment(experiment_path,
                                                                    combination):
                    experiment_dir = self._get_experiment_dir(experiment_path)
                    err_msg = 'experiment files in directory {}'.format(experiment_dir)
                    solving_option = 'resume_simulation'

            if check_setup and err_msg == '':
                resume_setup = self._options['resume_setup']
                system_id = combination['system']

                # Check system and molecule setup dirs
                is_sys_setup, is_sys_processed = self._db.is_system_setup(system_id)
                if is_sys_processed and not resume_setup:
                    system_dir = os.path.dirname(
                        self._db.get_system_files_paths(system_id)[0].position_path)
                    err_msg = 'system setup directory {}'.format(system_dir)
                elif not is_sys_setup:  # then this must go through the pipeline
                    try:  # binding free energy system
                        receptor_id = self._db.systems[system_id]['receptor']
                        ligand_id = self._db.systems[system_id]['ligand']
                        molecule_ids = [receptor_id, ligand_id]
                    except KeyError:  # partition/solvation free energy system
                        molecule_ids = [self._db.systems[system_id]['solute']]
                    for molecule_id in molecule_ids:
                        is_processed = self._db.is_molecule_setup(molecule_id)[1]
                        if is_processed and not resume_setup:
                            err_msg = 'molecule {} file'.format(molecule_id)
                            break

                if err_msg != '':
                    solving_option = 'resume_setup'

            # Check for errors
            if err_msg != '':
                err_msg += (' already exists; cowardly refusing to proceed. Move/delete '
                            'directory or set {} options').format(solving_option)
                raise YamlParseError(err_msg)

    # --------------------------------------------------------------------------
    # OpenMM Platform configuration
    # --------------------------------------------------------------------------

    @staticmethod
    def _opencl_device_support_precision(precision_model):
        """
        Check if this device supports the given precision model for OpenCL platform.

        Some OpenCL devices do not support double precision. This offers a test
        function.

        Returns
        -------
        is_supported : bool
            True if this device supports double precision for OpenCL, False
            otherwise.

        """
        opencl_platform = openmm.Platform.getPlatformByName('OpenCL')

        # Platforms are singleton so we need to store
        # the old precision model before modifying it
        old_precision = opencl_platform.getPropertyDefaultValue('OpenCLPrecision')

        # Test support by creating a toy context
        opencl_platform.setPropertyDefaultValue('Precision', precision_model)
        system = openmm.System()
        system.addParticle(1.0 * unit.amu)  # system needs at least 1 particle
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        try:
            context = openmm.Context(system, integrator, opencl_platform)
            is_supported = True
        except Exception:
            is_supported = False
        else:
            del context
        del integrator

        # Restore old precision
        opencl_platform.setPropertyDefaultValue('Precision', old_precision)

        return is_supported

    @classmethod
    def _configure_platform(cls, platform_name, platform_precision):
        """
        Configure the platform to be used for simulation for the given precision.

        Parameters
        ----------
        platform_name : str
            The name of the platform to be used for execution. If 'fastest',
            the fastest available platform is used.
        platform_precision : str or None
            The precision to be used. If 'auto' the default value is used,
            which is always mixed precision except for Reference that only
            supports double precision, and OpenCL when the device supports
            only single precision. If None, the precision mode won't be
            set, so OpenMM default value will be used which is always
            'single' for CUDA and OpenCL.

        Returns
        -------
        platform : simtk.openmm.Platform
           The configured platform.

        Raises
        ------
        RuntimeError
            If the given precision model selected is not compatible with the
            platform.

        """
        # Determine the platform to configure
        if platform_name == 'fastest':
            platform = mmtools.utils.get_fastest_platform()
            platform_name = platform.getName()
        else:
            platform = openmm.Platform.getPlatformByName(platform_name)

        # Set CUDA DeterministicForces (necessary for MBAR).
        if platform_name == 'CUDA':
            platform.setPropertyDefaultValue('DeterministicForces', 'true')

        # Use only a single CPU thread if we are using the CPU platform.
        # TODO: Since there is an environment variable that can control this,
        # TODO: we may want to avoid doing this.
        mpicomm = mpi.get_mpicomm()
        if platform_name == 'CPU' and mpicomm is not None:
            logger.debug("Setting 'CpuThreads' to 1 because MPI is active.")
            platform.setPropertyDefaultValue('CpuThreads', '1')

        # If user doesn't specify precision, determine default value
        if platform_precision == 'auto':
            if platform_name == 'CUDA':
                platform_precision = 'mixed'
            elif platform_name == 'OpenCL':
                if cls._opencl_device_support_precision('mixed'):
                    platform_precision = 'mixed'
                else:
                    logger.info("This device does not support double precision for OpenCL. "
                                "Setting OpenCL precision to 'single'")
                    platform_precision = 'single'
            elif platform_name == 'Reference' or platform_name == 'CPU':
                platform_precision = None  # leave OpenMM default precision

        # Set platform precision
        if platform_precision is not None:
            logger.info("Setting {} platform to use precision model "
                        "'{}'.".format(platform_name, platform_precision))
            if platform_name == 'CUDA':
                platform.setPropertyDefaultValue('Precision', platform_precision)
            elif platform_name == 'OpenCL':
                # Some OpenCL devices do not support double precision so we need to test it
                if cls._opencl_device_support_precision(platform_precision):
                    platform.setPropertyDefaultValue('Precision', platform_precision)
                else:
                    raise RuntimeError('This device does not support double precision for OpenCL.')
            elif platform_name == 'Reference':
                if platform_precision != 'double':
                    raise RuntimeError("Reference platform does not support precision model '{}';"
                                       "only 'double' is supported.".format(platform_precision))
            elif platform_name == 'CPU':
                if platform_precision != 'mixed':
                    raise RuntimeError("CPU platform does not support precision model '{}';"
                                       "only 'mixed' is supported.".format(platform_precision))
            else:  # This is an unkown platform
                raise RuntimeError("Found unknown platform '{}'.".format(platform_name))

        return platform

    # --------------------------------------------------------------------------
    # Experiment setup
    # --------------------------------------------------------------------------

    def _setup_experiments(self):
        """Set up all experiments without running them.

        IMPORTANT: This does not check if we are about to overwrite files, nor it
        cd into the script directory! Use setup_experiments() for that.

        """
        # Create setup directory if it doesn't exist.
        os.makedirs(self._db.setup_dir, exist_ok=True)

        # Configure log file for setup.
        setup_log_file_path = os.path.join(self._db.setup_dir, 'setup.log')
        utils.config_root_logger(self._options['verbose'], setup_log_file_path)

        # Setup all systems.
        self._db.setup_all_systems()

    # --------------------------------------------------------------------------
    # Automatic alchemical path generation
    # --------------------------------------------------------------------------

    @staticmethod
    def _find_automatic_protocol_phases(protocol):
        """Return the list of phase names in the protocol whose alchemical
        path must be generated automatically."""
        assert isinstance(protocol, collections.OrderedDict)
        phases_to_generate = []
        for phase_name in protocol:
            if protocol[phase_name]['alchemical_path'] == 'auto':
                phases_to_generate.append(phase_name)
        return phases_to_generate

    def _generate_experiments_protocols(self):
        """Go through all experiments and generate auto alchemical paths."""
        # Find all experiments that have at least one phase whose
        # alchemical path needs to be generated automatically.
        experiments_to_generate = []
        for experiment_path, experiment in self._expand_experiments():
            # First check if we have already generated the path for it.
            script_filepath = self._get_generated_yaml_script_path(experiment_path)
            if os.path.isfile(script_filepath):
                continue

            # Determine output directory and create it if it doesn't exist.
            os.makedirs(os.path.dirname(script_filepath), exist_ok=True)

            # Check if any of the phases needs to have its path generated.
            protocol = self._protocols[experiment['protocol']]
            phases_to_generate = self._find_automatic_protocol_phases(protocol)
            if len(phases_to_generate) > 0:
                experiments_to_generate.append((experiment_path, experiment))
            else:
                # Export YAML file for reproducibility.
                self._generate_yaml(experiment, script_filepath)

        # Parallelize generation of all protocols among nodes.
        mpi.distribute(self._generate_experiment_protocol,
                       distributed_args=experiments_to_generate,
                       send_results_to=None, group_size=1, sync_nodes=True)

    def _generate_experiment_protocol(self, experiment, constrain_receptor=True,
                                      n_equilibration_iterations=None, **kwargs):
        """Generate auto alchemical paths for the given experiment.

        Creates a YAML script in the experiment folder with the found protocol.

        Parameters
        ----------
        experiment : tuple (str, dict)
            A tuple with the experiment path and the experiment description.
        constrain_receptor : bool, optional
            If True, the receptor in a receptor-ligand system will have its
            CA atoms constrained during optimization (default is True).
        n_equilibration_iterations : None or int
            The number of equilibration iterations to perform before running
            the path search. If None, the function will determine the number
            of iterations to run based on the system dimension.

        Other Parameters
        ----------------
        kwargs : dict
            Other parameters to pass to pipeline.find_alchemical_protocol().

        """
        class DummyReporter(object):
            """A dummy reporter since we don't need to store MultiState stuff on disk."""
            def nothing(self, *args, **kwargs):
                """This serves both as an attribute and a callable."""
                pass

            def __getattr__(self, _):
                return self.nothing

        # Unpack experiment argument that has been distributed among nodes.
        experiment_path, experiment = experiment

        # Maybe only a subset of the phases need to be generated.
        protocol = self._protocols[experiment['protocol']]
        phases_to_generate = self._find_automatic_protocol_phases(protocol)

        # Build experiment. Use a dummy protocol for building since it hasn't been generated yet.
        exp = self._build_experiment(experiment_path, experiment, use_dummy_protocol=True)

        # Generate protocols.
        optimal_protocols = collections.OrderedDict.fromkeys(phases_to_generate)
        for phase_idx, phase_name in enumerate(phases_to_generate):
            logger.debug('Generating alchemical path for {}.{}'.format(experiment_path, phase_name))
            phase = exp.phases[phase_idx]
            state_parameters = []
            is_vacuum = len(phase.topography.receptor_atoms) == 0 and len(phase.topography.solvent_atoms) == 0

            # We may need to slowly turn on a Boresch restraint.
            if isinstance(phase.restraint, restraints.BoreschLike):
                state_parameters.append(('lambda_restraints', [0.0, 1.0]))

            # We support only lambda sterics and electrostatics for now.
            if is_vacuum and not phase.alchemical_regions.annihilate_electrostatics:
                state_parameters.append(('lambda_electrostatics', [1.0, 1.0]))
            else:
                state_parameters.append(('lambda_electrostatics', [1.0, 0.0]))
            if is_vacuum and not phase.alchemical_regions.annihilate_sterics:
                state_parameters.append(('lambda_sterics', [1.0, 1.0]))
            else:
                state_parameters.append(('lambda_sterics', [1.0, 0.0]))

            # Turn the RMSD restraints off slowly at the end
            if isinstance(phase.restraint, restraints.RMSD):
                state_parameters.append(('lambda_restraints', [1.0, 0.0]))

            # We only need to create a single state.
            phase.protocol = {par[0]: [par[1][0]] for par in state_parameters}

            # Remove unsampled state that we don't need for the optimization.
            phase.options['anisotropic_dispersion_correction'] = False

            # If default argument is used, determine number of equilibration iterations.
            # TODO automatic equilibration?
            if n_equilibration_iterations is None:
                if is_vacuum:  # Vacuum or small molecule in implicit solvent.
                    n_equilibration_iterations = 0
                elif len(phase.topography.receptor_atoms) == 0:  # Explicit solvent phase.
                    n_equilibration_iterations = 250
                elif len(phase.topography.solvent_atoms) == 0:  # Implicit complex phase.
                    n_equilibration_iterations = 500
                else:  # Explicit complex phase
                    n_equilibration_iterations = 1000

            # Set number of equilibration iterations.
            phase.options['number_of_equilibration_iterations'] = n_equilibration_iterations

            # Use a reporter that doesn't write anything to save time.
            phase.storage = DummyReporter()

            # Create the thermodynamic state exactly as AlchemicalPhase would make it.
            alchemical_phase = phase.initialize_alchemical_phase()

            # Get sampler and thermodynamic state and delete alchemical phase.
            thermodynamic_state = alchemical_phase._sampler._thermodynamic_states[0]
            sampler_state = alchemical_phase._sampler._sampler_states[0]
            mcmc_move = alchemical_phase._sampler.mcmc_moves[0]
            del alchemical_phase

            # Restrain the receptor heavy atoms to avoid drastic
            # conformational changes (possibly after equilibration).
            if len(phase.topography.receptor_atoms) != 0 and constrain_receptor:
                receptor_atoms_set = set(phase.topography.receptor_atoms)
                # Check first if there are alpha carbons. If not, restrain all carbons.
                restrained_atoms = [atom.index for atom in phase.topography.topology.atoms
                                    if atom.name is 'CA' and atom.index in receptor_atoms_set]
                if len(restrained_atoms) == 0:
                    # Select all carbon atoms of the receptor.
                    restrained_atoms = [atom.index for atom in phase.topography.topology.atoms
                                        if atom.element.symbol is 'C' and atom.index in receptor_atoms_set]
                mmtools.forcefactories.restrain_atoms(thermodynamic_state, sampler_state,
                                                      restrained_atoms, sigma=3.0*unit.angstroms)

            # Find protocol.
            alchemical_path = pipeline.trailblaze_alchemical_protocol(thermodynamic_state, sampler_state,
                                                                      mcmc_move, state_parameters,
                                                                      **kwargs)
            optimal_protocols[phase_name] = alchemical_path

        # Generate yaml script with updated protocol.
        script_path = self._get_generated_yaml_script_path(experiment_path)
        protocol = copy.deepcopy(self._protocols[experiment['protocol']])
        for phase_name, alchemical_path in optimal_protocols.items():
            protocol[phase_name]['alchemical_path'] = alchemical_path
        self._generate_yaml(experiment, script_path, overwrite_protocol=protocol)

    @mpi.on_single_node(rank=0, sync_nodes=True)
    def _generate_yaml(self, experiment, file_path, overwrite_protocol=None):
        """Generate the minimum YAML file needed to reproduce the experiment.

        Parameters
        ----------
        experiment : dict
            The dictionary describing a single experiment.
        file_path : str
            The path to the file to save.
        overwrite_protocol : None or dict
            If not None, this protocol description will be used instead
            of the one in the original YAML file.

        """
        yaml_dir = os.path.dirname(file_path)
        sys_descr = self._db.systems[experiment['system']]  # system description

        # Molecules section data
        try:
            try:  # binding free energy
                molecule_ids = [sys_descr['receptor'], sys_descr['ligand']]
            except KeyError:  # partition/solvation free energy
                molecule_ids = [sys_descr['solute']]
            mol_section = {mol_id: self._expanded_raw_yaml['molecules'][mol_id]
                           for mol_id in molecule_ids}

            # Copy to avoid modifying _expanded_raw_yaml when updating paths
            mol_section = copy.deepcopy(mol_section)
        except KeyError:  # user provided directly system files
            mol_section = {}

        # Solvents section data
        try:  # binding free energy
            solvent_ids = [sys_descr['solvent']]
        except KeyError:  # partition/solvation free energy
            try:
                solvent_ids = [sys_descr['solvent1'], sys_descr['solvent2']]
            except KeyError:  # from xml/pdb system files
                assert 'phase1_path' in sys_descr
                solvent_ids = []
        sol_section = {sol_id: self._expanded_raw_yaml['solvents'][sol_id]
                       for sol_id in solvent_ids}

        # Systems section data
        system_id = experiment['system']
        sys_section = {system_id: copy.deepcopy(self._expanded_raw_yaml['systems'][system_id])}

        # Protocols section data
        protocol_id = experiment['protocol']
        if overwrite_protocol is None:
            prot_section = {protocol_id: self._expanded_raw_yaml['protocols'][protocol_id]}
        else:
            prot_section = {protocol_id: overwrite_protocol}

        # We pop the options section in experiment and merge it to the general one
        exp_section = experiment.copy()
        opt_section = self._expanded_raw_yaml['options'].copy()
        opt_section.update(exp_section.pop('options', {}))

        # Convert relative paths to new script directory
        for molecule in mol_section.values():
            if 'filepath' in molecule and not os.path.isabs(molecule['filepath']):
                molecule['filepath'] = os.path.relpath(molecule['filepath'], yaml_dir)

        try:  # systems for which user has specified directly system files
            for phase in ['phase2_path', 'phase1_path']:
                for path in sys_section[system_id][phase]:
                    sys_section[system_id][path] = os.path.relpath(path, yaml_dir)
        except KeyError:  # system went through pipeline
            pass

        try:  # output directory
            output_dir = opt_section['output_dir']
        except KeyError:
            output_dir = self.GENERAL_DEFAULT_OPTIONS['output_dir']
        if not os.path.isabs(output_dir):
            opt_section['output_dir'] = os.path.relpath(output_dir, yaml_dir)

        # If we are converting a combinatorial experiment into a
        # single one we must set the correct experiment directory
        experiment_dir = os.path.relpath(yaml_dir, output_dir)
        if experiment_dir != self.GENERAL_DEFAULT_OPTIONS['experiments_dir']:
            opt_section['experiments_dir'] = experiment_dir

        # Create YAML with the sections in order
        dump_options = {'Dumper': YankDumper, 'line_break': '\n', 'indent': 4}
        yaml_content = yaml.dump({'version': self._version}, explicit_start=True, **dump_options)
        yaml_content += yaml.dump({'options': opt_section}, **dump_options)
        if mol_section:
            yaml_content += yaml.dump({'molecules': mol_section},  **dump_options)
        if sol_section:
            yaml_content += yaml.dump({'solvents': sol_section},  **dump_options)
        yaml_content += yaml.dump({'systems': sys_section},  **dump_options)
        yaml_content += yaml.dump({'protocols': prot_section},  **dump_options)
        yaml_content += yaml.dump({'experiments': exp_section},  **dump_options)

        # Export YAML into a file
        with open(file_path, 'w') as f:
            f.write(yaml_content)

    def _get_experiment_protocol(self, experiment_path, experiment_description,
                                 use_dummy_protocol=False):
        """Obtain the protocol for this experiment.

        This masks whether the protocol is hardcoded in the input YAML
        script or it has been generated automatically.

        Parameters
        ----------
        experiment_path : str
            The directory where to store the output files relative to the main
            output directory as specified by the user in the YAML script.
        experiment_description : dict
            A dictionary describing a single experiment.
        use_dummy_protocol : bool, optional
            If True, automatically-generated protocols that have not been found
            are substituted by a dummy protocol.

        Returns
        -------
        protocol : OrderedDict
            A dictionary thermodynamic_variable -> list of values.

        """
        protocol_id = experiment_description['protocol']
        protocol = copy.deepcopy(self._protocols[protocol_id])

        # Check if there are automatically-generated protocols.
        generated_alchemical_paths = self._find_automatic_protocol_phases(protocol)
        if len(generated_alchemical_paths) > 0:
            yaml_script_file_path = self._get_generated_yaml_script_path(experiment_path)

            # Use a dummy protocol if the file doesn't exist.
            try:
                with open(yaml_script_file_path, 'r') as f:
                    yaml_script = yaml.load(f, Loader=YankLoader)
            except FileNotFoundError:
                if not use_dummy_protocol:
                    raise
                for phase_name in generated_alchemical_paths:
                    protocol[phase_name]['alchemical_path'] = {}
            else:
                protocol = yaml_script['protocols'][protocol_id]

        return protocol

    # --------------------------------------------------------------------------
    # Experiment building
    # --------------------------------------------------------------------------

    @staticmethod
    def _save_analysis_script(results_dir, phase_names):
        """Store the analysis information about phase signs for analyze."""
        analysis = [[phase_names[0], 1], [phase_names[1], -1]]
        analysis_script_path = os.path.join(results_dir, 'analysis.yaml')
        with open(analysis_script_path, 'w') as f:
            yaml.dump(analysis, f)

    def _get_experiment_mpi_group_size(self, experiments):
        """Return the MPI group size to pass when executing the experiments.

        The heuristic tries to allocate the MPI processes among the experiments
        roughly according to their computational costs using the number of states
        of the first phase (either complex or solvent1).

        Parameters
        ----------
        experiments : list of pairs
            Each pair contains (experiment_path, experiment_description) of an
            experiment that needs to be run (i.e. that hasn't been completed yet).

        Returns
        -------
        groups_size : list of integers
            The MPI processes groups to pass to mpi.distribute().

        """
        mpicomm = mpi.get_mpicomm()
        n_experiments = len(experiments)
        processes_per_experiment = self._options['processes_per_experiment']

        # Check if we need to run the experiments sequentially.
        if mpicomm is None:
            return None
        n_mpi_processes = mpicomm.size

        # If we are using SAMS samplers, use 1 process only for all experiments.
        # TODO when n_replicas is a parameter of the constructor, remove this.
        sampler_names = {self._create_experiment_sampler(exp[1], []).__class__.__name__ for exp in experiments}
        if 'SAMSSampler' in sampler_names:
            if processes_per_experiment != 'auto':
                logger.warning('The option "processes_per_experiment" will be overwritten as SAMS '
                               'simulations are currently only compatible with processes_per_experiment=1')
            if n_mpi_processes > n_experiments:
                logger.warning('One MPI process will be assigned to each experiment but there are '
                               'more MPI processes than experiments. Some process will be unused.')
            return 1

        # Check if the user has specified an hardcoded
        # number of processes per experiments.
        if processes_per_experiment != 'auto':
            # If more processes are requested than MPI processes, run serially.
            if processes_per_experiment is not None and processes_per_experiment >= n_mpi_processes:
                return None
            return processes_per_experiment

        # If there are less MPI processes than experiments, completely split the MPI comm.
        if n_mpi_processes <= n_experiments:
            return 1

        # Split the mpicomm among the experiments.
        group_size = min(1, int(n_mpi_processes / n_experiments))

        # Estimate the computational cost of each experiment taken as the
        # number of thermodynamic states of the complex phase.
        experiment_costs = np.zeros(n_experiments)
        for experiment_idx, (experiment_path, experiment_description) in enumerate(experiments):
            protocol = self._get_experiment_protocol(experiment_path, experiment_description)
            first_phase_name = next(iter(protocol))  # protocol is an OrderedDict
            n_states = len(protocol[first_phase_name]['alchemical_path']['lambda_electrostatics'])
            experiment_costs[experiment_idx] = n_states

        # Find the index of the most expensive jobs.
        n_expensive_experiments = n_mpi_processes - n_experiments
        expensive_experiment_indices = list(reversed(np.argsort(experiment_costs)))
        expensive_experiment_indices = expensive_experiment_indices[:n_expensive_experiments]

        # The most expensive jobs are allocated an extra MPI process.
        group_size = [group_size for _ in range(n_experiments)]
        for expensive_experiment_idx in expensive_experiment_indices:
            group_size[expensive_experiment_idx] += 1
        return group_size

    def _create_experiment_restraint(self, experiment_description):
        """Create a restraint object for the experiment."""
        # Determine restraint description (None if not specified).
        restraint_description = experiment_description.get('restraint', None)
        if restraint_description is not None:
            return schema.call_restraint_constructor(restraint_description)
        return None

    def _create_default_mcmc_move(self, experiment_description, mc_atoms):
        """Instantiate the default MCMCMove."""
        experiment_options = self._determine_experiment_options(experiment_description)[0]
        integrator_move = mmtools.mcmc.LangevinSplittingDynamicsMove(
            timestep=experiment_options['default_timestep'],
            collision_rate=1.0 / unit.picosecond,
            n_steps=experiment_options['default_nsteps_per_iteration'],
            reassign_velocities=True,
            n_restart_attempts=6,
            measure_shadow_work=False,
            measure_heat=False
        )
        # Apply MC rotation displacement to ligand if there are MC atoms.
        if len(mc_atoms) > 0:
            move_list = [
                mmtools.mcmc.MCDisplacementMove(atom_subset=mc_atoms),
                mmtools.mcmc.MCRotationMove(atom_subset=mc_atoms),
                integrator_move
            ]
        else:
            return integrator_move
        return mmtools.mcmc.SequenceMove(move_list=move_list)

    def _get_experiment_sampler_constructor(self, experiment_description):
        """Return the experiment sampler constructor description or the default if None is specified."""
        # Check if we need to use the default sampler.
        sampler_id = experiment_description.get('sampler', None)
        if sampler_id is None:
            constructor_description = {'type': 'ReplicaExchangeSampler'}
        else:
            constructor_description = copy.deepcopy(self._samplers[sampler_id])

        # Overwrite default number of iterations if not specified.
        experiment_options, phase_options, _, _ = self._determine_experiment_options(experiment_description)
        if 'number_of_iterations' not in constructor_description:
            default_number_of_iterations = experiment_options['default_number_of_iterations']
            constructor_description['number_of_iterations'] = default_number_of_iterations

        # Overwrite the online analysis interval if not specified
        if not ("online_analysis_interval" in constructor_description and
                constructor_description["online_analysis_interval"] != "checkpoint"):
            constructor_description["online_analysis_interval"] = \
                phase_options.get('checkpoint_interval',
                                  AlchemicalPhaseFactory.DEFAULT_OPTIONS['checkpoint_interval'])

        return constructor_description

    def _get_experiment_number_of_iterations(self, experiment_description):
        """Return the number of iterations for the experiment.

        Resolve the priority between default_number_of_iterations and the
        options specified in the sampler used for the experiment.
        """
        constructor_description = self._get_experiment_sampler_constructor(experiment_description)
        return constructor_description['number_of_iterations']

    def _create_experiment_sampler(self, experiment_description, default_mc_atoms):
        """Create the sampler object associated to the given experiment."""
        # Obtain the sampler's constructor description.
        constructor_description = self._get_experiment_sampler_constructor(experiment_description)
        # Create the MCMCMove for the sampler.
        mcmc_move_id = constructor_description.get('mcmc_moves', None)
        if mcmc_move_id is None:
            mcmc_move = self._create_default_mcmc_move(experiment_description, default_mc_atoms)
        else:
            mcmc_move = schema.call_mcmc_move_constructor(self._mcmc_moves[mcmc_move_id])
        constructor_description['mcmc_moves'] = mcmc_move
        # Create the sampler.
        return schema.call_sampler_constructor(constructor_description)

    def _build_experiment(self, experiment_path, experiment, use_dummy_protocol=False):
        """Prepare a single experiment.

        Parameters
        ----------
        experiment_path : str
            The directory where to store the output files relative to the main
            output directory as specified by the user in the YAML script.
        experiment : dict
            A dictionary describing a single experiment
        use_dummy_protocol : bool, optional
            If True, automatically-generated protocols that have not been found
            are substituted by a dummy protocol.

        Returns
        -------
        yaml_experiment : Experiment
            A Experiment object.

        """
        system_id = experiment['system']

        # Get and validate experiment sub-options and divide them by class.
        exp_opts = self._determine_experiment_options(experiment)
        (exp_opts, phase_opts, alchemical_region_opts, alchemical_factory_opts) = exp_opts

        # Configure logger file for this experiment.
        experiment_log_file_path = self._get_experiment_log_path(experiment_path)
        utils.config_root_logger(self._options['verbose'], experiment_log_file_path)

        # Initialize alchemical factory.
        alchemical_factory = mmtools.alchemy.AbsoluteAlchemicalFactory(**alchemical_factory_opts)

        # Get ligand resname for alchemical atom selection. If we can't
        # find it, this is a solvation free energy calculation.
        ligand_dsl = None
        try:
            # First try for systems that went through pipeline.
            ligand_molecule_id = self._db.systems[system_id]['ligand']
        except KeyError:
            # Try with system from system files.
            try:
                ligand_dsl = self._db.systems[system_id]['ligand_dsl']
            except KeyError:
                # This is a solvation free energy.
                pass
        else:
            # Make sure that molecule filepath points to the mol2 file
            self._db.is_molecule_setup(ligand_molecule_id)
            ligand_descr = self._db.molecules[ligand_molecule_id]
            ligand_resname = utils.Mol2File(ligand_descr['filepath']).resname
            ligand_dsl = 'resname ' + ligand_resname

        if ligand_dsl is None:
            logger.debug('Cannot find ligand specification. '
                         'Alchemically modifying the whole solute.')
        else:
            logger.debug('DSL string for the ligand: "{}"'.format(ligand_dsl))

        # Determine solvent DSL.
        try:
            solvent_dsl = self._db.systems[system_id]['solvent_dsl']
        except KeyError:
            solvent_dsl = 'auto'  # Topography uses common solvent resnames.
        logger.debug('DSL string for the solvent: "{}"'.format(solvent_dsl))

        # Determine complex and solvent phase solvents while also getting regions
        system_description = self._db.systems[system_id]
        try:  # binding free energy calculations
            solvent_ids = [system_description['solvent'],
                                  system_description['solvent']]
            ligand_regions = self._db.molecules.get(system_description.get('ligand'), {}).get('regions', {})
            receptor_regions = self._db.molecules.get(system_description.get('receptor'), {}).get('regions', {})
            # Name clashes have been resolved in the yaml validation
            regions = {'ligand_atoms': ligand_regions, 'receptor_atoms': receptor_regions}
        except KeyError:  # partition/solvation free energy calculations
            try:
                solvent_ids = [system_description['solvent1'],
                               system_description['solvent2']]
                regions = {'solute_atoms':
                               self._db.molecules.get(system_description.get('solute'), {}).get('regions', {})}
            except KeyError:  # from xml/pdb system files
                assert 'phase1_path' in system_description
                solvent_ids = [None, None]
                regions = {}

        # Obtain the protocol for this experiment.
        protocol = self._get_experiment_protocol(experiment_path, experiment, use_dummy_protocol)

        # Get system files.
        system_files_paths = self._db.get_system(system_id)
        gromacs_include_dir = self._db.systems[system_id].get('gromacs_include_dir', None)

        # Prepare Yank arguments
        phases = [None, None]
        # protocol is an OrderedDict so phases are in the correct
        # order (e.g. [complex, solvent] or [solvent1, solvent2]).
        assert isinstance(protocol, collections.OrderedDict)
        phase_names = list(protocol.keys())
        phase_paths = self._get_nc_file_paths(experiment_path, experiment)
        for phase_idx, (phase_name, phase_path) in enumerate(zip(phase_names, phase_paths)):
            # Check if we need to resume a phase. If the phase has been
            # already created, Experiment will resume from the storage.
            if os.path.isfile(phase_path):
                phases[phase_idx] = phase_path
                continue

            # Create system, topology and sampler state from system files.
            solvent_id = solvent_ids[phase_idx]
            positions_file_path = system_files_paths[phase_idx].position_path
            parameters_file_path = system_files_paths[phase_idx].parameters_path
            if solvent_id is None:
                system_options = None
            else:
                system_options = {**self._db.solvents[solvent_id], **exp_opts}
            logger.info("Reading phase {}".format(phase_name))
            system, topology, sampler_state = pipeline.read_system_files(
                positions_file_path, parameters_file_path, system_options,
                gromacs_include_dir=gromacs_include_dir)

            # Identify system components. There is a ligand only in the complex phase.
            if phase_idx == 0:
                ligand_atoms = ligand_dsl
            else:
                ligand_atoms = None
            topography = Topography(topology, ligand_atoms=ligand_atoms,
                                    solvent_atoms=solvent_dsl)

            # Add regions
            for sub_region, specific_regions in regions.items():
                for region_name, region_description in specific_regions.items():
                    topography.add_region(region_name, region_description, subset=sub_region)

            # Create reference thermodynamic state.
            if system.usesPeriodicBoundaryConditions():
                pressure = exp_opts['pressure']
            else:
                pressure = None
            thermodynamic_state = mmtools.states.ThermodynamicState(system, exp_opts['temperature'],
                                                                    pressure=pressure)

            # Start from AlchemicalPhase default alchemical region
            # and modified it according to the user options.
            phase_protocol = protocol[phase_name]['alchemical_path']
            alchemical_region = AlchemicalPhase._build_default_alchemical_region(system, topography,
                                                                                 phase_protocol)
            alchemical_region = alchemical_region._replace(**alchemical_region_opts)

            # Apply restraint only if this is the first phase. AlchemicalPhase
            # will take care of raising an error if the phase type does not support it.
            if phase_idx == 0:
                restraint = self._create_experiment_restraint(experiment)
            else:
                restraint = None

            # Create MCMC moves and sampler. Apply MC rotation displacement to ligand.
            # We don't try displacing and rotating the ligand with a Boresch restraint
            # since the attempts would likely always fail.
            if len(topography.ligand_atoms) > 0 and not isinstance(restraint, restraints.BoreschLike):
                mc_atoms = topography.ligand_atoms
            else:
                mc_atoms = []
            sampler = self._create_experiment_sampler(experiment, mc_atoms)

            # Create phases.
            phases[phase_idx] = AlchemicalPhaseFactory(sampler, thermodynamic_state, sampler_state,
                                                       topography, phase_protocol, storage=phase_path,
                                                       restraint=restraint, alchemical_regions=alchemical_region,
                                                       alchemical_factory=alchemical_factory, **phase_opts)

        # Dump analysis script
        results_dir = self._get_experiment_dir(experiment_path)
        mpi.run_single_node(0, self._save_analysis_script, results_dir, phase_names)

        # Return new Experiment object.
        number_of_iterations = self._get_experiment_number_of_iterations(experiment)
        return Experiment(phases, number_of_iterations, exp_opts['switch_phase_interval'])

    # --------------------------------------------------------------------------
    # Experiment run
    # --------------------------------------------------------------------------

    def _run_experiment(self, experiment):
        """Run a single experiment.

        This runs the experiment only for ``switch_experiment_interval``
        iterations (if specified).

        Parameters
        ----------
        experiment : tuple (str, dict)
            A tuple with the experiment path and the experiment description.

        Returns
        -------
        is_completed
            True if the experiment has completed the number of iterations
            requested or if it has reached the target statistical error.

        """
        # Unpack experiment argument that has been distributed among nodes.
        experiment_path, experiment = experiment

        # Handle case where we don't have to switch between experiments.
        if self._options['switch_experiment_interval'] <= 0:
            # Run Experiment for number_of_iterations.
            switch_experiment_interval = None
        else:
            switch_experiment_interval = self._options['switch_experiment_interval']

        built_experiment = self._build_experiment(experiment_path, experiment)

        # Trap a NaN'd simulation by capturing only the error we can handle, let all others raise normally
        try:
            built_experiment.run(n_iterations=switch_experiment_interval)
        except multistate.SimulationNaNError:
            # Print out to critical logger.
            nan_warning_string = ('\n\n'  # Initial blank line for spacing.
                                  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                                  '!     CRITICAL: Experiment NaN    !\n'
                                  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                                  'The following experiment threw a NaN! It should NOT be considered!\n'
                                  'Experiment: {}\n'
                                  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                                  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                                  ).format(self._get_experiment_dir(experiment_path))
            logger.critical(nan_warning_string)

            # Flag the experiment as completed to avoid continuing in the next cycle.
            return True
        return built_experiment.is_completed


if __name__ == "__main__":
    import doctest
    doctest.testmod()
