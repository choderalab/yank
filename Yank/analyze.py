#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Analyze
=======

YANK Specific anlysis tools for YANK simulations from the :class:`yank.yank.AlchemicalPhase` classes

Extends classes from the MultiStateAnalyzer package to include the


"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

import os
import abc
import yaml
import mdtraj
import logging
import numpy as np
import simtk.unit as units
import openmmtools as mmtools
from pymbar import timeseries
from . import multistate

logger = logging.getLogger(__name__)


# =============================================================================================
# MODULE PARAMETERS
# =============================================================================================

# Extend registry to support standard_state_correction
yank_registry = multistate.default_observables_registry
yank_registry.register_phase_observable('standard_state_correction')


# =============================================================================================
# MODULE CLASSES
# =============================================================================================


class YankPhaseAnalyzer(multistate.PhaseAnalyzer):

    def __init__(self, *args, registry=yank_registry, **kwargs):
        super().__init__(*args, registry=registry, **kwargs)

    # Abstract methods
    @abc.abstractmethod
    def analyze_phase(self, *args, **kwargs):
        """
        Auto-analysis function for the phase

        Function which broadly handles "auto-analysis" for those that do not wish to call all the methods on their own.

        Returns a dictionary of analysis objects
        """
        raise NotImplementedError()


class YankMultiStateSamplerAnalyzer(multistate.MultiStateSamplerAnalyzer, YankPhaseAnalyzer):

    def get_standard_state_correction(self):
        """
        Compute the standard state correction free energy associated with the Phase.

        Returns
        -------
        standard_state_correction : float
            Free energy contribution from the standard_state_correction

        """
        if self._computed_observables['standard_state_correction'] is not None:
            return self._computed_observables['standard_state_correction']

        # Determine if we need to recompute the standard state correction.
        compute_ssc = self.unbias_restraint
        try:
            restraint_force, _, _ = self._get_radially_symmetric_restraint_data()
        except (mmtools.forces.NoForceFoundError, TypeError):
            compute_ssc = False

        if compute_ssc:
            thermodynamic_state = self._get_end_thermodynamic_states()[0]
            restraint_energy_cutoff, restraint_distance_cutoff = self._get_restraint_cutoffs()
            # TODO: Compute average box volume here to feed to max_volume?
            ssc = restraint_force.compute_standard_state_correction(
                thermodynamic_state, square_well=True, radius_cutoff=restraint_distance_cutoff,
                energy_cutoff=restraint_energy_cutoff, max_volume='system')

            # Update observable.
            self._computed_observables['standard_state_correction'] = ssc
            logger.debug('Computed a new standard state correction of {} kT'.format(ssc))

        # Reads the SSC from the reporter if compute_ssc is False.
        if self._computed_observables['standard_state_correction'] is None:
            ssc = self._reporter.read_dict('metadata')['standard_state_correction']
            self._computed_observables['standard_state_correction'] = ssc
        return self._computed_observables['standard_state_correction']

    def analyze_phase(self, cutoff=0.05):
        number_equilibrated, g_t, _ = self._equilibration_data
        self.show_mixing_statistics(cutoff=cutoff, number_equilibrated=number_equilibrated)
        data = {}
        # Accumulate free energy differences
        Deltaf_ij, dDeltaf_ij = self.get_free_energy()
        DeltaH_ij, dDeltaH_ij = self.get_enthalpy()
        data['DeltaF'] = Deltaf_ij[self.reference_states[0], self.reference_states[1]]
        data['dDeltaF'] = dDeltaf_ij[self.reference_states[0], self.reference_states[1]]
        data['DeltaH'] = DeltaH_ij[self.reference_states[0], self.reference_states[1]]
        data['dDeltaH'] = dDeltaH_ij[self.reference_states[0], self.reference_states[1]]
        data['DeltaF_standard_state_correction'] = self.get_standard_state_correction()
        return data


class YankReplicaExchangeAnalyzer(multistate.ReplicaExchangeAnalyzer, YankMultiStateSamplerAnalyzer):
    pass


class YankParallelTemperingAnalyzer(multistate.ParallelTemperingAnalyzer, YankMultiStateSamplerAnalyzer):
    pass


# =============================================================================================
# MODULE FUNCTIONS
# =============================================================================================

def get_analyzer(file_base_path, **analyzer_kwargs):
    """
    Utility function to convert storage file to a Reporter and Analyzer by reading the data on file

    For now this is mostly placeholder functions since there is only the implemented :class:`ReplicaExchangeAnalyzer`,
    but creates the API for the user to work with.

    Parameters
    ----------
    file_base_path : string
        Complete path to the storage file with filename and extension.
    **analyzer_kwargs
        Keyword arguments to pass to the analyzer.

    Returns
    -------
    analyzer : instance of implemented :class:`Yank*Analyzer`
        Analyzer for the specific phase.
    """
    # Eventually extend this to get more reporters, but for now simple placeholder
    reporter = multistate.MultiStateReporter(file_base_path, open_mode='r')
    """
    storage = infer_storage_format_from_extension('complex.nc')  # This is always going to be nc for now.
    metadata = storage.metadata
    sampler_class = metadata['sampler_full_name']
    module_name, cls_name = sampler_full_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    reporter = cls.create_reporter('complex.nc')
    """
    # Eventually change this to auto-detect simulation from reporter:
    if True:
        analyzer = YankReplicaExchangeAnalyzer(reporter, **analyzer_kwargs)
    else:
        raise RuntimeError("Cannot automatically determine analyzer for Reporter: {}".format(reporter))
    return analyzer


def analyze_directory(source_directory, **analyzer_kwargs):
    """
    Analyze contents of store files to compute free energy differences.

    This function is needed to preserve the old auto-analysis style of YANK. What it exactly does can be refined when
    more analyzers and simulations are made available. For now this function exposes the API.

    Parameters
    ----------
    source_directory : string
        The location of the simulation storage files.
    **analyzer_kwargs
        Keyword arguments to pass to the analyzer.

    """
    analysis_script_path = os.path.join(source_directory, 'analysis.yaml')
    if not os.path.isfile(analysis_script_path):
        err_msg = 'Cannot find analysis.yaml script in {}'.format(source_directory)
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    with open(analysis_script_path, 'r') as f:
        analysis = yaml.load(f)
    phase_names = [phase_name for phase_name, sign in analysis]
    data = dict()
    for phase_name, sign in analysis:
        phase_path = os.path.join(source_directory, phase_name + '.nc')
        phase = get_analyzer(phase_path, **analyzer_kwargs)
        data[phase_name] = phase.analyze_phase()
        kT = phase.kT

    # Compute free energy and enthalpy
    DeltaF = 0.0
    dDeltaF = 0.0
    DeltaH = 0.0
    dDeltaH = 0.0
    for phase_name, sign in analysis:
        DeltaF -= sign * (data[phase_name]['DeltaF'] + data[phase_name]['DeltaF_standard_state_correction'])
        dDeltaF += data[phase_name]['dDeltaF']**2
        DeltaH -= sign * (data[phase_name]['DeltaH'] + data[phase_name]['DeltaF_standard_state_correction'])
        dDeltaH += data[phase_name]['dDeltaH']**2
    dDeltaF = np.sqrt(dDeltaF)
    dDeltaH = np.sqrt(dDeltaH)

    # Attempt to guess type of calculation
    calculation_type = ''
    for phase in phase_names:
        if 'complex' in phase:
            calculation_type = ' of binding'
        elif 'solvent1' in phase:
            calculation_type = ' of solvation'

    # Print energies
    logger.info('')
    logger.info('Free energy{:<13}: {:9.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(
        calculation_type, DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole,
        dDeltaF * kT / units.kilocalories_per_mole))
    logger.info('')

    for phase in phase_names:
        logger.info('DeltaG {:<17}: {:9.3f} +- {:.3f} kT'.format(phase, data[phase]['DeltaF'],
                                                                 data[phase]['dDeltaF']))
        if data[phase]['DeltaF_standard_state_correction'] != 0.0:
            logger.info('DeltaG {:<17}: {:18.3f} kT'.format('restraint',
                                                            data[phase]['DeltaF_standard_state_correction']))
    logger.info('')
    logger.info('Enthalpy{:<16}: {:9.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(
        calculation_type, DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole,
        dDeltaH * kT / units.kilocalories_per_mole))


# ==========================================
# HELPER FUNCTIONS FOR TRAJECTORY EXTRACTION
# ==========================================

def extract_u_n(ncfile):
    """
    Extract timeseries of u_n = - log q(X_n) from store file

    where q(X_n) = \pi_{k=1}^K u_{s_{nk}}(x_{nk})

    with X_n = [x_{n1}, ..., x_{nK}] is the current collection of replica configurations
    s_{nk} is the current state of replica k at iteration n
    u_k(x) is the kth reduced potential

    TODO: Figure out a way to remove this function

    Parameters
    ----------
    ncfile : netCDF4.Dataset
       Open NetCDF file to analyze

    Returns
    -------
    u_n : numpy array of numpy.float64
       u_n[n] is -log q(X_n)
    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    logger.info("Reading energies...")
    energies = ncfile.variables['energies']
    u_kln_replica = np.zeros([nstates, nstates, niterations], np.float64)
    for n in range(niterations):
        u_kln_replica[:, :, n] = energies[n, :, :]
    logger.info("Done.")

    # Deconvolute replicas
    logger.info("Deconvoluting replicas...")
    u_kln = np.zeros([nstates, nstates, niterations], np.float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration, :]
        u_kln[state_indices,:,iteration] = energies[iteration, :, :]
    logger.info("Done.")

    # Compute total negative log probability over all iterations.
    u_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        u_n[iteration] = np.sum(np.diagonal(u_kln[:, :, iteration]))

    return u_n


# ==============================================================================
# Extract trajectory from NetCDF4 file
# ==============================================================================

def extract_trajectory(nc_path, nc_checkpoint_file=None, state_index=None, replica_index=None,
                       start_frame=0, end_frame=-1, skip_frame=1, keep_solvent=True,
                       discard_equilibration=False, image_molecules=False):
    """Extract phase trajectory from the NetCDF4 file.

    Parameters
    ----------
    nc_path : str
        Path to the primary nc_file storing the analysis options
    nc_checkpoint_file : str or None, Optional
        File name of the checkpoint file housing the main trajectory
        Used if the checkpoint file is differently named from the default one chosen by the nc_path file.
        Default: None
    state_index : int, optional
        The index of the alchemical state for which to extract the trajectory.
        One and only one between state_index and replica_index must be not None
        (default is None).
    replica_index : int, optional
        The index of the replica for which to extract the trajectory. One and
        only one between state_index and replica_index must be not None (default
        is None).
    start_frame : int, optional
        Index of the first frame to include in the trajectory (default is 0).
    end_frame : int, optional
        Index of the last frame to include in the trajectory. If negative, will
        count from the end (default is -1).
    skip_frame : int, optional
        Extract one frame every skip_frame (default is 1).
    keep_solvent : bool, optional
        If False, solvent molecules are ignored (default is True).
    discard_equilibration : bool, optional
        If True, initial equilibration frames are discarded (see the method
        pymbar.timeseries.detectEquilibration() for details, default is False).

    Returns
    -------
    trajectory: mdtraj.Trajectory
        The trajectory extracted from the netcdf file.

    """
    # Check correct input
    if (state_index is None) == (replica_index is None):
        raise ValueError('One and only one between "state_index" and '
                         '"replica_index" must be specified.')
    if not os.path.isfile(nc_path):
        raise ValueError('Cannot find file {}'.format(nc_path))

    # Import simulation data
    reporter = None
    try:
        reporter = multistate.MultiStateReporter(nc_path, open_mode='r', checkpoint_storage=nc_checkpoint_file)
        metadata = reporter.read_dict('metadata')
        reference_system = mmtools.utils.deserialize(metadata['reference_state']).system
        topography = mmtools.utils.deserialize(metadata['topography'])
        topology = topography.topology

        # Determine if system is periodic
        is_periodic = reference_system.usesPeriodicBoundaryConditions()
        logger.info('Detected periodic boundary conditions: {}'.format(is_periodic))

        # Get dimensions
        # Assume full iteration until proven otherwise
        last_checkpoint = True
        trajectory_storage = reporter._storage_checkpoint
        if not keep_solvent:
            # If tracked solute particles, use any last iteration, set with this logic test
            full_iteration = len(reporter.analysis_particle_indices) == 0
            if not full_iteration:
                trajectory_storage = reporter._storage_analysis
                topology = topology.subset(reporter.analysis_particle_indices)

        n_iterations = reporter.read_last_iteration(last_checkpoint=last_checkpoint)
        n_frames = trajectory_storage.variables['positions'].shape[0]
        n_atoms = trajectory_storage.variables['positions'].shape[2]
        logger.info('Number of frames: {}, atoms: {}'.format(n_frames, n_atoms))

        # Determine frames to extract.
        # Convert negative indices to last indices.
        if start_frame < 0:
            start_frame = n_frames + start_frame
        if end_frame < 0:
            end_frame = n_frames + end_frame + 1
        frame_indices = range(start_frame, end_frame, skip_frame)
        if len(frame_indices) == 0:
            raise ValueError('No frames selected')
        logger.info('Extracting frames from {} to {} every {}'.format(
            start_frame, end_frame, skip_frame))

        # Discard equilibration samples
        if discard_equilibration:
            u_n = extract_u_n(reporter._storage_analysis)
            # Discard frame 0 with minimized energy which throws off automatic equilibration detection.
            n_equil_iterations, g, n_eff = timeseries.detectEquilibration(u_n[1:])
            n_equil_iterations += 1
            logger.info(("Discarding initial {} equilibration samples (leaving {} "
                         "effectively uncorrelated samples)...").format(n_equil_iterations, n_eff))
            # Find first frame post-equilibration.
            if not full_iteration:
                for iteration in range(n_equil_iterations, n_iterations):
                    n_equil_frames = reporter._calculate_checkpoint_iteration(iteration)
                    if n_equil_frames is not None:
                        break
            else:
                n_equil_frames = n_equil_iterations
            frame_indices = frame_indices[n_equil_frames:-1]

        # Determine the number of frames that the trajectory will have.
        if state_index is None:
            n_trajectory_frames = len(frame_indices)
        else:
            # With SAMS, an iteration can have 0 or more replicas in a given state.
            # Deconvolute state indices.
            state_indices = [None for _ in frame_indices]
            for i, iteration in enumerate(frame_indices):
                replica_indices = reporter._storage_analysis.variables['states'][iteration, :]
                state_indices[i] = np.where(replica_indices == state_index)[0]
            n_trajectory_frames = sum(len(x) for x in state_indices)

        # Initialize positions and box vectors arrays.
        # MDTraj Cython code expects float32 positions.
        positions = np.zeros((n_trajectory_frames, n_atoms, 3), dtype=np.float32)
        if is_periodic:
            box_vectors = np.zeros((n_trajectory_frames, 3, 3), dtype=np.float32)

        # Extract state positions and box vectors.
        if state_index is not None:
            logger.info('Extracting positions of state {}...'.format(state_index))

            # Extract state positions and box vectors.
            frame_idx = 0
            for i, iteration in enumerate(frame_indices):
                for replica_index in state_indices[i]:
                    positions[frame_idx, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                    if is_periodic:
                        box_vectors[frame_idx, :, :] = trajectory_storage.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)
                    frame_idx += 1

        else:  # Extract replica positions and box vectors
            logger.info('Extracting positions of replica {}...'.format(replica_index))

            for i, iteration in enumerate(frame_indices):
                positions[i, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = trajectory_storage.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)
    finally:
        if reporter is not None:
            reporter.close()

    # Create trajectory object
    logger.info('Creating trajectory object...')
    trajectory = mdtraj.Trajectory(positions, topology)
    if is_periodic:
        trajectory.unitcell_vectors = box_vectors

    # Force periodic boundary conditions to molecules positions
    if image_molecules and is_periodic:
        logger.info('Applying periodic boundary conditions to molecules positions...')
        # Use the receptor as an anchor molecule.
        anchor_atom_indices = set(topography.receptor_atoms)
        if len(anchor_atom_indices) == 0:  # Hydration free energy.
            anchor_atom_indices = set(topography.solute_atoms)
        anchor_molecules = [{a for a in topology.atoms if a.index in anchor_atom_indices}]
        trajectory.image_molecules(inplace=True, anchor_molecules=anchor_molecules)
    elif image_molecules:
        logger.warning('The molecules will not be imaged because the system is non-periodic.')

    return trajectory
