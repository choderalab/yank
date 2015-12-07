#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Tools to build Yank experiments from a YAML configuration file.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import re
import copy
import yaml
import logging
logger = logging.getLogger(__name__)

import numpy as np
import openmoltools
from simtk import unit, openmm

import utils
import pipeline
from yank import Yank
from repex import ReplicaExchange, ThermodynamicState
from sampling import ModifiedHamiltonianExchange


#=============================================================================================
# UTILITY FUNCTIONS
#=============================================================================================

def compute_min_dist(mol_positions, *args):
    """Compute the minimum distance between a molecule and a set of other molecules.

    All the positions must be expressed in the same unit of measure.

    Parameters
    ----------
    mol_positions : numpy.ndarray
        An Nx3 array where, N is the number of atoms, containing the positions of
        the atoms of the molecule for which we want to compute the minimum distance
        from the others

    Other parameters
    ----------------
    args
        A series of numpy.ndarrays containing the positions of the atoms of the other
        molecules

    Returns
    -------
    min_dist : float
        The minimum distance between mol_positions and the other set of positions

    """
    for pos1 in args:
        # Compute squared distances
        # Each row is an array of distances from a mol2 atom to all mol1 atoms
        distances2 = np.array([((pos1 - pos2)**2).sum(1) for pos2 in mol_positions])

        # Find closest atoms and their distance
        min_idx = np.unravel_index(distances2.argmin(), distances2.shape)
        try:
            min_dist = min(min_dist, np.sqrt(distances2[min_idx]))
        except UnboundLocalError:
            min_dist = np.sqrt(distances2[min_idx])
    return min_dist

def remove_overlap(mol_positions, *args, **kwargs):
    """Remove any eventual overlap between a molecule and a set of others.

    The method both randomly shifts and rotates the molecule (when overlapping atoms
    are detected) until it does not clash with any other given molecule anymore. All
    the others are kept fixed.

    All the positions must be expressed in the same unit of measure.

    Parameters
    ----------
    mol_positions : numpy.ndarray
        An Nx3 array where, N is the number of atoms, containing the positions of
        the atoms of the molecule that we want to not clash with the others.
    min_distance : float
        The minimum distance accepted to consider the molecule not clashing with
        the others. Must be in the same unit of measure of the positions.
    sigma : float
        The maximum displacement for a single step. Must be in the same unit of
        measure of the positions.

    Other parameters
    ----------------
    args
        A series of numpy.ndarrays containing the positions of the atoms of the
        molecules that are kept fixed.

    Returns
    -------
    x : numpy.ndarray
        Positions of the atoms of the given molecules that do not clash.

    """
    x = np.copy(mol_positions)
    sigma = kwargs.get('sigma', 1.0)
    min_distance = kwargs.get('min_distance', 1.0)

    # Try until we have a non-overlapping conformation w.r.t. all fixed molecules
    while compute_min_dist(x, *args) <= min_distance:
        # Compute center of geometry
        x0 = x.mean(0)

        # Randomize orientation of ligand.
        q = ModifiedHamiltonianExchange._generate_uniform_quaternion()
        Rq = ModifiedHamiltonianExchange._rotation_matrix_from_quaternion(q)
        x = ((Rq * np.matrix(x - x0).T).T + x0).A

        # Choose a random displacement vector and translate
        x += sigma * np.random.randn(3)

    return x

def to_openmm_app(str):
    """Converter function to be used with validate_parameters()."""
    return getattr(openmm.app, str)

#=============================================================================================
# BUILDER CLASS
#=============================================================================================

class YamlParseError(Exception):
    """Represent errors occurring during parsing of Yank YAML file."""
    def __init__(self, message):
        super(YamlParseError, self).__init__(message)
        logger.error(message)

class YamlBuilder:
    """Parse YAML configuration file and build the experiment.

    The relative paths indicated in the script are assumed to be relative to
    the script directory. However, if YamlBuilder is initiated with a string
    rather than a file path, the paths will be relative to the user's working
    directory.

    The class firstly perform a dry run to check if this is going to overwrite
    some files and raises an exception if it finds already existing output folders
    unless the options resume_setup or resume_simulation are True.

    Properties
    ----------
    yank_options : dict
        The options specified in the parsed YAML file that will be passed to Yank.
        These are not the full range of options specified in the script since some
        of them are used to configure YamlBuilder and not the Yank object.

    Examples
    --------
    >>> import textwrap
    >>> setup_dir = utils.get_data_filename(os.path.join('..', 'examples',
    ...                                     'p-xylene-implicit', 'setup'))
    >>> pxylene_path = os.path.join(setup_dir, 'ligand.tripos.mol2')
    >>> lysozyme_path = os.path.join(setup_dir, 'receptor.pdbfixer.pdb')
    >>> with utils.temporary_directory() as tmp_dir:
    ...     yaml_content = '''
    ...     ---
    ...     options:
    ...         number_of_iterations: 1
    ...         output_dir: {}
    ...     molecules:
    ...         T4lysozyme:
    ...             filepath: {}
    ...             parameters: oldff/leaprc.ff99SBildn
    ...         p-xylene:
    ...             filepath: {}
    ...             parameters: antechamber
    ...     solvents:
    ...         vacuum:
    ...             nonbonded_method: NoCutoff
    ...     experiments:
    ...         components:
    ...             receptor: T4lysozyme
    ...             ligand: p-xylene
    ...             solvent: vacuum
    ...     '''.format(tmp_dir, lysozyme_path, pxylene_path)
    >>> yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
    >>> yaml_builder.build_experiment()

    """

    SYSTEMS_DIR = 'systems'
    MOLECULES_DIR = 'molecules'

    DEFAULT_OPTIONS = {
        'verbose': False,
        'mpi': False,
        'resume_setup': False,
        'resume_simulation': False,
        'output_dir': 'output',
        'setup_dir': 'setup',
        'experiments_dir': 'experiments',
        'temperature': 298 * unit.kelvin,
        'pressure': 1 * unit.atmosphere,
        'constraints': openmm.app.HBonds,
        'hydrogen_mass': 1 * unit.amu
    }

    @property
    def yank_options(self):
        return self._isolate_yank_options(self.options)

    def __init__(self, yaml_source):
        """Parse the given YAML configuration file.

        This does not build the actual experiment but simply checks that the syntax
        is correct and loads the configuration into memory.

        Parameters
        ----------
        yaml_source : str
            A path to the YAML script or the YAML content.

        """

        self._oe_molecules = {}  # molecules generated by OpenEye
        self._fixed_pos_cache = {}  # positions of molecules given as files

        # TODO check version of yank-yaml language
        # TODO what if there are multiple streams in the YAML file?
        # Load YAML script and decide working directory for relative paths
        try:
            with open(yaml_source, 'r') as f:
                yaml_content = yaml.load(f)
            self._script_dir = os.path.dirname(yaml_source)
        except IOError:
            yaml_content = yaml.load(yaml_source)
            self._script_dir = os.getcwd()

        if yaml_content is None:
            raise YamlParseError('The YAML file is empty!')

        # Save raw YAML content that will be needed when generating the YAML files
        self._raw_yaml = copy.deepcopy({key: yaml_content.get(key, {})
                                        for key in ['options', 'molecules', 'solvents']})

        # Parse each section
        self._parse_options(yaml_content)
        self._parse_molecules(yaml_content)
        self._parse_solvents(yaml_content)
        self._parse_experiments(yaml_content)

    def build_experiment(self):
        """Set up and run all the Yank experiments."""
        # Throw exception if there are no experiments
        if len(self._experiments) == 0:
            raise YamlParseError('No experiments specified!')

        # Run all experiments with paths relative to the script directory
        with utils.temporary_cd(self._script_dir):
            self._check_resume()

            for output_dir, combination in self._expand_experiments():
                self._run_experiment(combination, output_dir)

    def _validate_options(self, options):
        """Return a dictionary with YamlBuilder and Yank options validated."""
        template_options = self.DEFAULT_OPTIONS.copy()
        template_options.update(Yank.default_parameters)
        template_options.update(ReplicaExchange.default_parameters)
        openmm_app_type = {'constraints': to_openmm_app}
        try:
            valid = utils.validate_parameters(options, template_options, check_unknown=True,
                                              process_units_str=True, float_to_int=True,
                                              special_conversions=openmm_app_type)
        except (TypeError, ValueError) as e:
            raise YamlParseError(str(e))
        return valid

    def _isolate_yank_options(self, options):
        """Return the options that do not belong to YamlBuilder."""
        return {opt: val for opt, val in options.items()
                if opt not in self.DEFAULT_OPTIONS}

    def _determine_experiment_options(self, experiment):
        """Merge the options specified in the experiment section with the general ones.

        Options in the general section have priority. Options in the experiment section
        are validated.

        Parameters
        ----------
        experiment : dict
            The dictionary encoding the experiment.

        Returns
        -------
        exp_options : dict
            A new dictionary containing the all the options that apply for the experiment.

        """
        exp_options = self.options.copy()
        exp_options.update(self._validate_options(experiment.get('options', {})))
        return exp_options

    def _parse_options(self, yaml_content):
        """Validate and store options in the script.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        """
        # Merge options and metadata and validate
        temp_options = yaml_content.get('options', {})
        temp_options.update(yaml_content.get('metadata', {}))

        # Validate options and fill in default values
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options.update(self._validate_options(temp_options))

    def _parse_molecules(self, yaml_content):
        """Load molecules information and check that their syntax is correct.

        One and only one source must be specified (e.g. filepath, name). Also
        the parameters must be specified, and the extension of filepath must
        match one of the supported file formats.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        """
        file_formats = set(['mol2', 'pdb'])
        sources = set(['filepath', 'name', 'smiles'])
        template_mol = {'filepath': 'str', 'name': 'str', 'smiles': 'str',
                        'parameters': 'str', 'epik': 0}

        self._molecules = yaml_content.get('molecules', {})

        # First validate and convert
        for molecule_id, molecule in self._molecules.items():
            try:
                self._molecules[molecule_id] = utils.validate_parameters(molecule, template_mol,
                                                                         check_unknown=True)
            except (TypeError, ValueError) as e:
                raise YamlParseError(str(e))

        err_msg = ''
        for molecule_id, molecule in self._molecules.items():
            fields = set(molecule.keys())

            # Check that only one source is specified
            specified_sources = sources & fields
            if not specified_sources or len(specified_sources) > 1:
                err_msg = ('need only one between {} for molecule {}').format(
                    ', '.join(list(sources)), molecule_id)

            # Check supported file formats
            elif 'filepath' in specified_sources:
                extension = os.path.splitext(molecule['filepath'])[1][1:]  # remove '.'
                if extension not in file_formats:
                    err_msg = 'molecule {}, only {} files supported'.format(
                        molecule_id, ', '.join(file_formats))

            # Check that parameters are specified
            if 'parameters' not in fields:
                err_msg = 'no parameters specified for molecule {}'.format(molecule_id)

            if err_msg != '':
                raise YamlParseError(err_msg)

    def _parse_solvents(self, yaml_content):
        """Load solvents information and check that their syntax is correct.

        The option nonbonded_method must be specified. All quantities are converted to
        simtk.app.Quantity objects or openmm.app.TYPE (e.g. app.PME, app.OBC2). This
        also perform some consistency checks to verify that the user did not mix
        implicit and explicit solvent parameters.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        """
        template_parameters = {'nonbonded_method': openmm.app.PME, 'nonbonded_cutoff': 1 * unit.nanometer,
                               'implicit_solvent': openmm.app.OBC2, 'clearance': 10.0 * unit.angstroms}
        openmm_app_type = ('nonbonded_method', 'implicit_solvent')
        openmm_app_type = {option: to_openmm_app for option in openmm_app_type}

        self._solvents = yaml_content.get('solvents', {})

        # First validate and convert
        for solvent_id, solvent in self._solvents.items():
            try:
                self._solvents[solvent_id] = utils.validate_parameters(solvent, template_parameters,
                                                         check_unknown=True, process_units_str=True,
                                                         special_conversions=openmm_app_type)
            except (TypeError, ValueError, AttributeError) as e:
                raise YamlParseError(str(e))

        err_msg = ''
        for solvent_id, solvent in self._solvents.items():

            # Test mandatory parameters
            if 'nonbonded_method' not in solvent:
                err_msg = 'solvent {} must specify nonbonded_method'.format(solvent_id)
                raise YamlParseError(err_msg)

            # Test solvent consistency
            nonbonded_method = solvent['nonbonded_method']
            if nonbonded_method == openmm.app.NoCutoff:
                if 'nonbonded_cutoff' in solvent:
                    err_msg = ('solvent {} specify both nonbonded_method: NoCutoff and '
                               'and nonbonded_cutoff').format(solvent_id)
            else:
                if 'implicit_solvent' in solvent:
                    err_msg = ('solvent {} specify both nonbonded_method: {} '
                               'and implicit_solvent').format(solvent_id, nonbonded_method)
                elif 'clearance' not in solvent:
                    err_msg = ('solvent {} uses explicit solvent but '
                               'no clearance specified').format(solvent_id)

            # Raise error
            if err_msg != '':
                raise YamlParseError(err_msg)

    def _expand_experiments(self):
        """Generates all possible combinations of experiment.

        Each generated experiment is uniquely named.

        Returns
        -------
        output_dir : str
            A unique path where to save the experiment output files relative to
            the main output directory specified by the user in the options.
        combination : dict
            The dictionary describing a single experiment.

        """
        output_dir = ''
        for exp_name, experiment in self._experiments.items():
            if len(self._experiments) > 1:
                output_dir = exp_name

            # Loop over all combinations
            for name, combination in experiment.named_combinations(separator='_', max_name_length=40):
                yield os.path.join(output_dir, name), combination

    def _parse_experiments(self, yaml_content):
        """Perform dry run and validate components and options of every combination.

        Receptor, ligand and solvent must be already loaded. If they are not found
        an exception is raised. Experiments options are validated as well.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        """
        experiment_template = {'components': {}, 'options': {}}
        components_template = {'receptor': 'str', 'ligand': 'str', 'solvent': 'str'}

        if 'experiments' not in yaml_content:
            self._experiments = {}
            return

        # Check if there is a sequence of experiments or a single one
        if isinstance(yaml_content['experiments'], list):
            self._experiments = {exp_name: utils.CombinatorialTree(yaml_content[exp_name])
                                 for exp_name in yaml_content['experiments']}
        else:
            self._experiments = {'experiments': utils.CombinatorialTree(yaml_content['experiments'])}

        # Check validity of every experiment combination
        err_msg = ''
        for exp_name, exp in self._expand_experiments():
            if exp_name == '':
                exp_name = 'experiments'

            # Check if we can identify components
            if 'components' not in exp:
                raise YamlParseError('Cannot find components for {}'.format(exp_name))
            components = exp['components']

            # Validate and check for unknowns
            try:
                utils.validate_parameters(exp, experiment_template, check_unknown=True)
                utils.validate_parameters(components, components_template, check_unknown=True)
                self._validate_options(exp.get('options', {}))
            except (ValueError, TypeError) as e:
                raise YamlParseError(str(e))

            # Check that components have been specified
            if components['receptor'] not in self._molecules:
                err_msg = 'Cannot identify receptor for {}'.format(exp_name)
            elif components['ligand'] not in self._molecules:
                err_msg = 'Cannot identify ligand for {}'.format(exp_name)
            elif components['solvent'] not in self._solvents:
                err_msg = 'Cannot identify solvent for {}'.format(exp_name)

            if err_msg != '':
                raise YamlParseError(err_msg)

    @staticmethod
    def _get_setup_dir(experiment_options):
        """Return the path to the directory where the setup output files
        should be stored.

        Parameters
        ----------
        experiment_options : dict
            A dictionary containing the validated options that apply to the
            experiment as obtained by _determine_experiment_options().

        """
        return os.path.join(experiment_options['output_dir'], experiment_options['setup_dir'])

    @staticmethod
    def _get_experiment_dir(experiment_options, experiment_subdir):
        """Return the path to the directory where the experiment output files
        should be stored.

        Parameters
        ----------
        experiment_options : dict
            A dictionary containing the validated options that apply to the
            experiment as obtained by _determine_experiment_options().
        experiment_subdir : str
            The relative path w.r.t. the main experiments directory (determined
            through the options) of the experiment-specific subfolder.

        """
        return os.path.join(experiment_options['output_dir'],
                            experiment_options['experiments_dir'], experiment_subdir)

    def _check_molecule_setup(self, setup_dir, molecule_id):
        """Check whether the molecule has been set up already.

        The molecule must be set up if it needs to be parametrize by antechamber
        (and the gaff.mol2 and frcmod files do not exist) or if the molecule must
        be generated by OpenEye. We set up the molecule in the second case even if
        the final output files already exist since its initial position may change
        from system to system in order to avoid overlapping atoms.

        Parameters
        ----------
        setup_dir : str
            The path to the main setup directory specified by the user in the YAML options.
        molecule_id : str
            The id of the molecule indicated by the user in the YAML file.

        Returns
        -------
        is_setup : bool
            True if the molecule has been already set up.
        molecule_dir : str
            Directory where the files of the molecule are (or should) be stored.
        parameters : str
            If is_setup is true and the molecule must be parametrized, this is
            the path to the parameters file to be used for the molecule. Otherwise
            this is an empty string.
        filepath : str
            If is_setup is true and the molecule must be parametrized, this is
            the path to the file describing the molecule. Otherwise this is an
            empty string.

        """
        filepath = ''
        parameters = ''
        is_setup = False
        raw_molecule_descr = self._raw_yaml['molecules'][molecule_id]
        molecule_dir = os.path.join(setup_dir, self.MOLECULES_DIR, molecule_id)

        # Check that this molecule doesn't have be generated by OpenEye
        # OpenEye and that the eventual antechamber output already exists
        if 'filepath' in raw_molecule_descr:
            # If it has to be parametrized, the antechamber files must exist
            if raw_molecule_descr['parameters'] == 'antechamber':
                parameters = os.path.join(molecule_dir, molecule_id + '.frcmod')
                filepath = os.path.join(molecule_dir, molecule_id + '.gaff.mol2')
                if os.path.isfile(parameters) and os.path.isfile(filepath):
                    is_setup = True
            else:
                is_setup = True

        return is_setup, molecule_dir, parameters, filepath

    @classmethod
    def _check_system_setup(cls, setup_dir, receptor_id, ligand_id, solvent_id):
        """Check whether the system has been set up already.

        Parameters
        ----------
        setup_dir : str
            The path to the main setup directory specified by the user in the YAML options.
        receptor_id : str
            The id of the receptor indicated by the user in the YAML file.
        ligand_id : str
            The id of the ligand indicated by the user in the YAML file.
        solvent_id : str
            The id of the solvent indicated by the user in the YAML file.

        Returns
        -------
        is_setup : bool
            True if the system has been already set up.
        system_dir : str
            Directory where the files of the system are (or should) be stored.

        """
        system_dir = '_'.join((receptor_id, ligand_id, solvent_id))
        system_dir = os.path.join(setup_dir, cls.SYSTEMS_DIR, system_dir)
        is_setup = (os.path.exists(os.path.join(system_dir, 'complex.prmtop')) and
                    os.path.exists(os.path.join(system_dir, 'complex.inpcrd')) and
                    os.path.exists(os.path.join(system_dir, 'solvent.prmtop')) and
                    os.path.exists(os.path.join(system_dir, 'solvent.inpcrd')))
        return is_setup, system_dir

    def _check_resume(self):
        """Perform dry run to check if we are going to overwrite files.

        If we find folders that YamlBuilder should create we throw an Exception
        unless resume_setup or resume_simulation are found, in which case we
        assume we need to use the existing files. We never overwrite files, the
        user is responsible to delete them or move them.

        It's important to check all possible combinations at the beginning to
        avoid interrupting the user simulation after few experiments.

        """
        err_msg = ''
        for exp_sub_dir, combination in self._expand_experiments():
            # Determine and validate options
            exp_options = self._determine_experiment_options(combination)
            resume_sim = exp_options['resume_simulation']
            resume_setup = exp_options['resume_setup']

            # Identify components
            components = combination['components']
            receptor_id = components['receptor']
            ligand_id = components['ligand']
            solvent_id = components['solvent']

            # Check experiment dir
            experiment_dir = self._get_experiment_dir(exp_options, exp_sub_dir)
            if os.path.exists(experiment_dir) and not resume_sim:
                err_msg = 'experiment directory {}'.format(experiment_dir)
                solving_option = 'resume_simulation'
            else:
                # Check system and molecule setup dirs
                setup_dir = self._get_setup_dir(exp_options)
                is_sys_setup, system_dir = self._check_system_setup(setup_dir, receptor_id,
                                                                    ligand_id, solvent_id)
                if is_sys_setup and not resume_setup:
                    err_msg = 'system setup directory {}'.format(system_dir)
                else:
                    for molecule_id in [receptor_id, ligand_id]:
                        is_setup, mol_dir, param = self._check_molecule_setup(setup_dir, molecule_id)[:3]
                        if is_setup and param != '' and not resume_setup:
                            err_msg = 'molecule setup directory {}'.format(mol_dir)
                            break

                if err_msg != '':
                    solving_option = 'resume_setup'

            # Check for errors
            if err_msg != '':
                err_msg += (' already exists; cowardly refusing to proceed. Move/delete '
                            'directory or set {} options').format(solving_option)
                raise YamlParseError(err_msg)

    def _generate_molecule(self, molecule_id):
        """Generate molecule using the OpenEye toolkit from name or smiles.

        The molecules is charged with OpenEye's recommended AM1BCC charge
        selection scheme.

        Parameters
        ----------
        molecule_id : str
            The id of the molecule as given in the YAML script

        Returns
        -------
        molecule : OEMol
            The generated molecule.

        """
        mol_descr = self._molecules[molecule_id]
        try:
            if 'name' in mol_descr:
                molecule = openmoltools.openeye.iupac_to_oemol(mol_descr['name'])
            elif 'smiles' in mol_descr:
                molecule = openmoltools.openeye.smiles_to_oemol(mol_descr['smiles'])
            molecule = openmoltools.openeye.get_charges(molecule, keep_confs=1)
        except ImportError as e:
            error_msg = ('requested molecule generation from name or smiles but '
                         'could not find OpenEye toolkit: ' + str(e))
            raise YamlParseError(error_msg)

        return molecule

    def _setup_molecules(self, setup_dir, *args):
        """Set up the files needed to generate the system for all the molecules.

        If OpenEye tools are installed, this generate the molecules when the source is
        not a file. If two (or more) molecules generated by OpenEye have overlapping
        atoms, the molecules are randomly shifted and rotated until the clash is resolved.
        With the OpenEye toolkit installed, we also perform a sanity check to verify that
        the molecules from files do not have overlapping atoms. An Exception is raised if
        this is not the case.

        If the Schrodinger's suite is install, this can enumerate tautomeric and protonation
        states with epik when requested.

        This also parametrize the molecule with antechamber when requested.

        Parameters
        ----------
        setup_dir : str
            The path to the main setup directory specified by the user in the YAML options

        Other parameters
        ----------------
        args
            All the molecules ids that compose the system. These molecules are the only
            ones considered when trying to resolve the overlapping atoms.

        """

        # Determine which molecules should have fixed positions
        # At the end of parametrization we update the 'filepath' key also for OpenEye-generated
        # molecules so we need to check that the molecule is not in self._oe_molecules as well
        file_mol_ids = {mol_id for mol_id in args if 'filepath' in self._molecules[mol_id] and
                        mol_id not in self._oe_molecules}

        # Generate missing molecules with OpenEye
        self._oe_molecules.update({mol_id: self._generate_molecule(mol_id) for mol_id in args
                                   if mol_id not in file_mol_ids and mol_id not in self._oe_molecules})

        # Check that non-generated molecules don't have overlapping atoms
        # TODO this check should be available even without OpenEye
        # TODO also there should be an option allowing to solve the overlap in this case too?
        fixed_pos = {}  # positions of molecules from files of THIS setup
        if utils.is_openeye_installed():
            # We need positions as a list so we separate the ids and positions in two lists
            mol_id_list = list(file_mol_ids)
            positions = [0 for _ in mol_id_list]
            for i, mol_id in enumerate(mol_id_list):
                try:
                    positions[i] = self._fixed_pos_cache[mol_id]
                except KeyError:
                    positions[i] = utils.get_oe_mol_positions(utils.read_oe_molecule(
                        self._molecules[mol_id]['filepath']))

            # Verify that distances between any pair of fixed molecules is big enough
            for i in range(len(positions) - 1):
                posi = positions[i]
                if compute_min_dist(posi, *positions[i+1:]) < 0.1:
                    raise YamlParseError('The given molecules have overlapping atoms!')

            # Convert positions list to dictionary, this is needed to solve overlaps
            fixed_pos = {mol_id_list[i]: positions[i] for i in range(len(mol_id_list))}

            # Cache positions for future molecule setups
            self._fixed_pos_cache.update(fixed_pos)

        # Find and solve overlapping atoms in OpenEye generated molecules
        for mol_id in args:
            # Retrive OpenEye-generated molecule
            try:
                molecule = self._oe_molecules[mol_id]
            except KeyError:
                continue
            molecule_pos = utils.get_oe_mol_positions(molecule)

            # Remove overlap and save new positions
            if fixed_pos:
                molecule_pos = remove_overlap(molecule_pos, *(fixed_pos.values()),
                                              min_distance=1.0, sigma=1.0)
                utils.set_oe_mol_positions(molecule, molecule_pos)

            # Update fixed positions for next round
            fixed_pos[mol_id] = molecule_pos

        # Save parametrized molecules
        for mol_id in args:
            mol_descr = self._molecules[mol_id]
            is_setup, mol_dir, parameters, filepath = self._check_molecule_setup(setup_dir, mol_id)

            # Have we already processed this molecule? Do we have to do it at all?
            # We don't want to create the output folder if we don't need to
            if is_setup:
                # Be sure that filepath and parameters point to the correct file
                # if this molecule must be parametrize with antechamber
                if parameters != '':
                    mol_descr['parameters'] = parameters
                    mol_descr['filepath'] = filepath
                continue

            # Create output directory if it doesn't exist
            if not os.path.exists(mol_dir):
                os.makedirs(mol_dir)

            # Write OpenEye generated molecules in mol2 files
            if mol_id in self._oe_molecules:
                # We update the 'filepath' key in the molecule description
                mol_descr['filepath'] = os.path.join(mol_dir, mol_id + '.mol2')

                # We set the residue name as the first three uppercase letters
                residue_name = re.sub('[^A-Za-z]+', '', mol_id.upper())
                openmoltools.openeye.molecule_to_mol2(molecule, mol_descr['filepath'],
                                                      residue_name=residue_name)

            # Enumerate protonation states with epik
            if 'epik' in mol_descr:
                epik_idx = mol_descr['epik']
                epik_output_file = os.path.join(mol_dir, mol_id + '-epik.mol2')
                utils.run_epik(mol_descr['filepath'], epik_output_file, extract_range=epik_idx)
                mol_descr['filepath'] = epik_output_file

            # Parametrize the molecule with antechamber
            if mol_descr['parameters'] == 'antechamber':
                # Generate parameters
                input_mol_path = os.path.abspath(mol_descr['filepath'])
                with utils.temporary_cd(mol_dir):
                    openmoltools.amber.run_antechamber(mol_id, input_mol_path)

                # Save new parameters paths, this way if we try to
                # setup the molecule again it will just be skipped
                mol_descr['filepath'] = os.path.join(mol_dir, mol_id + '.gaff.mol2')
                mol_descr['parameters'] = os.path.join(mol_dir, mol_id + '.frcmod')

    def _setup_system(self, setup_dir, components):
        """Create the prmtop and inpcrd files from the given components.

        This calls _setup_molecules() so there's no need to call it ahead. The
        system files are generated with tleap. If no molecule specify a general
        force field, leaprc.ff14SB is loaded.

        Parameters
        ----------
        setup_dir : str
            The path to the main setup directory specified by the user in the YAML options
        components : dict
            A dictionary containing the keys 'receptor', 'ligand' and 'solvent' with the ids
            of molecules and solvents

        Returns
        -------
        system_dir : str
            The path to the directory containing the prmtop and inpcrd files

        """

        # Identify components
        receptor_id = components['receptor']
        ligand_id = components['ligand']
        solvent_id = components['solvent']
        receptor = self._molecules[receptor_id]
        ligand = self._molecules[ligand_id]
        solvent = self._solvents[solvent_id]

        # Check if system has been already processed
        is_setup, system_dir = self._check_system_setup(setup_dir, receptor_id,
                                                        ligand_id, solvent_id)
        if is_setup:
            return system_dir

        # We still need to check if system_dir exists because the set up may
        # have been interrupted
        if not os.path.exists(system_dir):
            os.makedirs(system_dir)

        # Setup molecules
        self._setup_molecules(setup_dir, receptor_id, ligand_id)

        # Create tleap script
        tleap = utils.TLeap()
        tleap.new_section('Load GAFF parameters')
        tleap.load_parameters('leaprc.gaff')

        # Check that AMBER force field is specified
        if not ('leaprc.' in receptor['parameters'] or 'leaprc.' in ligand['parameters']):
            tleap.load_parameters('leaprc.ff14SB')

        # Load receptor and ligand
        for group_name in ['receptor', 'ligand']:
            group = self._molecules[components[group_name]]
            tleap.new_section('Load ' + group_name)
            tleap.load_parameters(group['parameters'])
            tleap.load_group(name=group_name, file_path=group['filepath'])

        # Create complex
        tleap.new_section('Create complex')
        tleap.combine('complex', 'receptor', 'ligand')

        # Configure solvent
        if solvent['nonbonded_method'] == openmm.app.NoCutoff:
            if 'implicit_solvent' in solvent:  # GBSA implicit solvent
                tleap.new_section('Set GB radii to recommended values for OBC')
                tleap.add_commands('set default PBRadii mbondi2')
        else:  # explicit solvent
            tleap.new_section('Solvate systems')
            clearance = float(solvent['clearance'].value_in_unit(unit.angstroms))
            tleap.solvate(group='complex', water_model='TIP3PBOX', clearance=clearance)
            tleap.solvate(group='ligand', water_model='TIP3PBOX', clearance=clearance)

        # Check charge
        tleap.new_section('Check charge')
        tleap.add_commands('check complex', 'charge complex')

        # Save prmtop and inpcrd files
        tleap.new_section('Save prmtop and inpcrd files')
        tleap.save_group('complex', os.path.join(system_dir, 'complex.prmtop'))
        tleap.save_group('complex', os.path.join(system_dir, 'complex.pdb'))
        tleap.save_group('ligand', os.path.join(system_dir, 'solvent.prmtop'))
        tleap.save_group('ligand', os.path.join(system_dir, 'solvent.pdb'))

        # Save tleap script for reference
        tleap.export_script(os.path.join(system_dir, 'leap.in'))

        # Run tleap!
        tleap.run()

        return system_dir

    def _generate_yaml(self, experiment, file_path):
        """Generate the minimum YAML file needed to reproduce the experiment.

        Parameters
        ----------
        experiment : dict
            The dictionary describing a single experiment.
        file_path : str
            The path to the file to save.

        """
        yaml_dir = os.path.dirname(file_path)
        components = set(experiment['components'].values())

        # Molecules section data
        mol_section = {mol_id: molecule for mol_id, molecule in self._raw_yaml['molecules'].items()
                       if mol_id in components}

        # Solvents section data
        sol_section = {solvent_id: solvent for solvent_id, solvent in self._raw_yaml['solvents'].items()
                       if solvent_id in components}

        # We pop the options section in experiment and merge it to the general one
        exp_section = experiment.copy()
        opt_section = self._raw_yaml['options'].copy()
        opt_section.update(exp_section.pop('options', {}))

        # Convert relative paths to new script directory
        mol_section = copy.deepcopy(mol_section)  # copy to avoid modifying raw yaml
        for molecule in mol_section.values():
            if 'filepath' in molecule and not os.path.isabs(molecule['filepath']):
                molecule['filepath'] = os.path.relpath(molecule['filepath'], yaml_dir)

        try:
            output_dir = opt_section['output_dir']
        except KeyError:
            output_dir = self.DEFAULT_OPTIONS['output_dir']
        if not os.path.isabs(output_dir):
            opt_section['output_dir'] = os.path.relpath(output_dir, yaml_dir)

        # If we are converting a combinatorial experiment into a
        # single one we must set the correct experiment directory
        experiment_dir = os.path.relpath(yaml_dir, output_dir)
        if experiment_dir != self.DEFAULT_OPTIONS['experiments_dir']:
            opt_section['experiments_dir'] = experiment_dir

        # Create YAML with the sections in order
        yaml_content = yaml.dump({'options': opt_section}, default_flow_style=False, line_break='\n', explicit_start=True)
        yaml_content += yaml.dump({'molecules': mol_section}, default_flow_style=False, line_break='\n')
        yaml_content += yaml.dump({'solvents': sol_section}, default_flow_style=False, line_break='\n')
        yaml_content += yaml.dump({'experiments': exp_section}, default_flow_style=False, line_break='\n')

        # Export YAML into a file
        with open(file_path, 'w') as f:
            f.write(yaml_content)

    def _run_experiment(self, experiment, experiment_dir):
        """Prepare and run a single experiment.

        Parameters
        ----------
        experiment : dict
            A dictionary describing a single experiment
        experiment_dir : str
            The directory where to store the output files relative to the main
            output directory as specified by the user in the YAML script

        """
        components = experiment['components']
        exp_name = 'experiments' if experiment_dir == '' else os.path.basename(experiment_dir)

        # Get and validate experiment sub-options
        exp_opts = self._determine_experiment_options(experiment)
        yank_opts = self._isolate_yank_options(exp_opts)

        # Configure MPI, if requested
        if exp_opts['mpi']:
            from mpi4py import MPI
            MPI.COMM_WORLD.barrier()
            mpicomm = MPI.COMM_WORLD
        else:
            mpicomm = None

        # TODO configure platform and precision when they are fixed in Yank

        # Create directory and configure logger for this experiment
        results_dir = self._get_experiment_dir(exp_opts, experiment_dir)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
            resume = False
        else:
            resume = True
        utils.config_root_logger(exp_opts['verbose'], os.path.join(results_dir, exp_name + '.log'))

        # Initialize simulation
        yank = Yank(results_dir, mpicomm=mpicomm, **yank_opts)

        if resume:
            yank.resume()
        else:
            # Export YAML file for reproducibility
            self._generate_yaml(experiment, os.path.join(results_dir, exp_name + '.yaml'))

            # Determine system files path
            setup_dir = self._get_setup_dir(exp_opts)
            system_dir = self._setup_system(setup_dir, components)

            # Get ligand resname for alchemical atom selection
            ligand_dsl = utils.get_mol2_resname(self._molecules[components['ligand']]['filepath'])
            if ligand_dsl is None:
                ligand_dsl = 'MOL'
            ligand_dsl = 'resname ' + ligand_dsl

            # System configuration
            create_system_filter = set(('nonbonded_method', 'nonbonded_cutoff', 'implicit_solvent',
                                        'constraints', 'hydrogen_mass'))
            solvent = self._solvents[components['solvent']]
            system_pars = {opt: solvent[opt] for opt in create_system_filter if opt in solvent}
            system_pars.update({opt: exp_opts[opt] for opt in create_system_filter
                                if opt in exp_opts})

            # Convert underscore_parameters to camelCase for OpenMM API
            system_pars = {utils.underscore_to_camelcase(opt): value
                           for opt, value in system_pars.items()}

            # Prepare system
            phases, systems, positions, atom_indices = pipeline.prepare_amber(system_dir, ligand_dsl, system_pars)

            # Create thermodynamic state
            thermodynamic_state = ThermodynamicState(temperature=exp_opts['temperature'],
                                                     pressure=exp_opts['pressure'])

            # Create new simulation
            yank.create(phases, systems, positions, atom_indices, thermodynamic_state)

        # Run the simulation!
        yank.run()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
