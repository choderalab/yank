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
import logging
import itertools

import numpy as np
import mdtraj as md

import simtk.unit as unit
import openmmtools as mmtools
from pymbar import timeseries, MBAR
from msmbuilder.cluster import RegularSpatial

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

    def analyze_phase(self, show_mixing=False, cutoff=0.05):
        """
        Auto-analysis function for the phase

        Function which broadly handles "auto-analysis" for those that do not wish to call all the methods on their own.

        This variant computes the following:

        * Equilibration data (accessible through ``n_equilibration_iterations`` and ``statistical_inefficiency``
          properties)
        * Optional mixing statistics printout
        * Free Energy difference between all states with error
        * Enthalpy difference between all states with error (as expectation of the reduced potential)
        * Free energy of the Standard State Correction for the phase.

        Parameters
        ----------
        show_mixing : bool, optional. Default: False
            Toggle to show mixing statistics or not. This can be a slow step so is disabled for speed by default.
        cutoff : float, optional. Default: 0.05
            Threshold below which % of mixing mixing from one state to another is not shown.
            Makes the output table more clean to read (rendered as whitespace)

        Returns
        -------
        data : dict
            A dictionary of analysis objects, in this case Delta F, Delta H, and Standard State Free Energy
            In units of kT (dimensionless)
        """
        number_equilibrated, g_t, _ = self._equilibration_data
        if show_mixing:
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
        calculation_type, DeltaF, dDeltaF, DeltaF * kT / unit.kilocalories_per_mole,
        dDeltaF * kT / unit.kilocalories_per_mole))
    logger.info('')

    for phase in phase_names:
        logger.info('DeltaG {:<17}: {:9.3f} +- {:.3f} kT'.format(phase, data[phase]['DeltaF'],
                                                                 data[phase]['dDeltaF']))
        if data[phase]['DeltaF_standard_state_correction'] != 0.0:
            logger.info('DeltaG {:<17}: {:18.3f} kT'.format('restraint',
                                                            data[phase]['DeltaF_standard_state_correction']))
    logger.info('')
    logger.info('Enthalpy{:<16}: {:9.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(
        calculation_type, DeltaH, dDeltaH, DeltaH * kT / unit.kilocalories_per_mole,
        dDeltaH * kT / unit.kilocalories_per_mole))


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
    trajectory = md.Trajectory(positions, topology)
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

# ==============================================================================
# Cluster ligand conformations and estimate populations in fully-interacting state
# ==============================================================================

# TODO: This is a preliminary draft. This can be heavily refactored after generalizing the analysis code in the MultiStateAnalyzer

def cluster(reference_pdb_filename, netcdf_filename, output_prefix='cluster', nsnapshots_per_cluster=5,
            receptor_dsl_selection='protein and name CA', ligand_dsl_selection='not protein and (mass > 1.5)',
            fully_interacting_state=0, ligand_rmsd_cutoff=0.3, ligand_filter_cutoff=0.3,
            cluster_filter_threshold=0.95):
    """
    Cluster ligand conformations and estimate populations in fully-interacting state

    Parameters
    ----------
    reference_pdb_filename : str
        The name of the PDB file for the solvated complex
    netcdf_filename : str
        The complex NetCDF file to read
    output_prefix : str
        String to prepend to cluster PDB files and populations written
    nsnapshots_per_cluster : int, optional, default=5
        The number of snapshots per state to write
    receptor_dsl_selection : str, optional, default='protein and name CA'
        MDTraj DSL to use for selecting receptor atoms for alignment
    ligand_dsl_selection : str, optional, default='not protein and (mass > 1.5)'
        MDTraj DSL to use for selectinf ligand atoms to cluster
    fully_interacting_state : int, optional, default=0
        0 specifies the fully-interacting state
        1 species the first alchemical state
    ligand_rmsd_cutoff : float, optional, default=0.3
        RMSD cutoff to use for ligand custering (in nanometers)
    ligand_filter_cutoff : float, optional, default=0.3
        Snapshots where ligand atoms are greater than this cutoff from the receptor are filtered out
    cluster_filter_threshold : float, optional, default=0.95
        Only the most populous clusters that add up to more than this threshold in population are written.

    The algorithm
    -------------
    * Compute per-snapshot weights (using MBAR) representing the relative weight of each snapshot in the fully interacting state
    * Cluster the remaining snapshots
    * Assign relative populations to the clusters
    * Sort clusters by population, writing only most populous clusters
    * Sample representative snapshots from the clusters proportional to their weights, writing out PDB files
    * Write out cluster populations

    """
    # mdtraj works in nanometers
    ligand_rmsd_cutoff /= unit.nanometers
    ligand_filter_cutoff /= unit.nanometers

    topology = md.load(reference_pdb_filename)
    solute_indices = topology.top.select('not water')
    logger.info('There are {:d} non-water atoms'.format(len(solute_indices)))
    topology = topology.atom_slice(solute_indices) # check that this is the same as w

    from netCDF4 import Dataset
    ncfile = Dataset(netcdf_filename, 'r')

    # TODO: Extend this to handle more than one replica
    replica_index = 0

    # Extract energy trajectories
    sampled_energy_matrix = np.transpose(np.array(ncfile.variables['energies'][:,replica_index:(replica_index+1),:], np.float32), axes=[1,2,0])
    unsampled_energy_matrix = np.transpose(np.array(ncfile.variables['unsampled_energies'][:,replica_index:(replica_index+1),:], np.float32), axes=[1,2,0])

    # Initialize the MBAR matrices in ln form.
    n_replicas, n_sampled_states, n_iterations = sampled_energy_matrix.shape
    _, n_unsampled_states, _ = unsampled_energy_matrix.shape
    logger.info('n_replicas: {:d}'.format(n_replicas))
    logger.info('n_sampled_states: {:d}'.format(n_sampled_states))
    logger.info('n_iterations: {:d}'.format(n_iterations))

    # Remove some frames
    # TODO: Change this to instead extract the "good" portion of the trajectory that isn't corrupted
    ntrim = 100
    logger.info('Trimming {:d} frames from either end'.format(ntrim))
    retained_snapshot_indices = list(range(ntrim, (n_iterations-ntrim)))
    n_iterations = len(retained_snapshot_indices)
    sampled_energy_matrix = sampled_energy_matrix[:,:,retained_snapshot_indices]
    unsampled_energy_matrix = unsampled_energy_matrix[:,:,retained_snapshot_indices]

    # Extract thermodynamic state indices
    replicas_state_indices = np.transpose(np.array(ncfile.variables['states'][retained_snapshot_indices,replica_index:(replica_index+1)], np.int64), axes=[1,0])

    # TODO: Pre-filter all states remote from fully-interacting state

    # TODO: We could detect the equilibration time and discard data to equilibration here
    #[t0, g, Neff_max] = timeseries.detectEquilibration(replicas_state_indices[replica_index,:], nskip=100)

    # Compute snapshot weights with MBAR

    #
    # Note: This comes from multistateanalyzer.py L1445-1479.
    # That section could be refactored to be more general to avoid code duplication
    #

    logger.info('Reformatting energies...')
    n_total_states = n_sampled_states + n_unsampled_states
    energy_matrix = np.zeros([n_total_states, n_iterations*n_replicas])
    samples_per_state = np.zeros([n_total_states], dtype=int)
    # Compute shift index for how many unsampled states there were.
    # This assume that we set an equal number of unsampled states at the end points.
    first_sampled_state = int(n_unsampled_states/2.0)
    last_sampled_state = n_total_states - first_sampled_state
    # Cast the sampled energy matrix from kln' to ln form.
    energy_matrix[first_sampled_state:last_sampled_state, :] = multistate.MultiStateSamplerAnalyzer.reformat_energies_for_mbar(sampled_energy_matrix)
    # Determine how many samples and which states they were drawn from.
    unique_sampled_states, counts = np.unique(replicas_state_indices, return_counts=True)
    # Assign those counts to the correct range of states.
    samples_per_state[first_sampled_state:last_sampled_state][unique_sampled_states] = counts
    # Add energies of unsampled states to the end points.
    if n_unsampled_states > 0:
        energy_matrix[[0, -1], :] = multistate.MultiStateSamplerAnalyzer.reformat_energies_for_mbar(unsampled_energy_matrix)

    # TODO: Should we instead only run MBAR *after* we have already filtered out problematic snapshots?
    logger.info('Estimating weights...')
    mbar = MBAR(energy_matrix, samples_per_state)
    # Extract weights
    w_n = mbar.W_nk[:,fully_interacting_state]

    # Extract unitcell lengths and angles
    # TODO: Make this more general for non-rectilinear boxes
    x = np.array(ncfile.variables['box_vectors'][retained_snapshot_indices,replica_index,:,:])
    unitcell_lengths = x[:,[0,1,2],[0,1,2]]
    unitcell_angles = 90.0 * np.ones(unitcell_lengths.shape, np.float32)

    # Extract solute trajectory as MDTraj Trajectory
    # NOTE: Only retained snapshots are extracted to speed things up
    # NOTE: This will store the whole trajectory in memory
    traj = md.Trajectory(ncfile.variables['positions'][retained_snapshot_indices,replica_index,solute_indices,:], topology.top, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)


    # Remove counterions
    # TODO: Is there a better way to eliminate everything but receptor and ligand?
    logger.info(traj)
    ion_dsl_selection = 'not (resname "Na+" or resname "Cl-")'
    indices = traj.top.select(ion_dsl_selection)
    traj = traj.atom_slice(indices)
    logger.info(traj)

    # Check snapshot weights are small
    logger.info('Maximum weight from any given snapshot (SHOULD BE SMALL!): {:f}'.format(w_n.max()))
    indices = np.argsort(-w_n)
    MAX_SNAPSHOT_WEIGHT = 0.01
    if w_n.max() > MAX_SNAPSHOT_WEIGHT:
        filename = '%s-outlier.pdb' % (output_prefix)
        logger.warning('WARNING: One snapshot is dominating the weights so clusters and populations will be unreliable')
        logger.warning('Writing outlier to {}'.format(filename))
        snapshot_index = w_n.argmax()
        logger.warning('snaphot {:d} has weight {:f}'.format(snapshot_index, w_n.max()))
        traj[snapshot_index].save(filename)

    # Image molecules into periodic box, ensuring ligand is in closest image to receptor
    traj.image_molecules(inplace=True)

    # Compute minimum heavy atom distance from ligand to protein
    residues = [residue for residue in traj.top.residues]
    protein_residues = [residue.index for residue in traj.top.residues if residue.is_protein]
    logger.info('There are {:d} protein residues'.format(len(protein_residues)))
    ligand_residues = [residue.index for residue in traj.top.residues if not residue.is_protein]
    logger.info('There are {:d} ligand residues'.format(len(ligand_residues)))

    pairs = list(itertools.product(ligand_residues, protein_residues))
    distances, contacts = md.compute_contacts(traj, contacts=pairs, scheme='closest-heavy', ignore_nonprotein=False)
    min_distances = distances.min(1)
    logger.info('Maximum ligand heavy atom distance from protein: {:f} nm'.format(min_distances.max()))
    logger.info('Minimum ligand heavy atom distance from protein: {:f} nm'.format(min_distances.min()))

    # Filter out snapshots where ligand is too far from the protein
    filtered_snapshot_indices = np.where(min_distances <= ligand_filter_cutoff)[0]
    logger.info('Retaining {:d} of {:d} snapshots where ligand heavy atoms are less than {:f} nm from protein'.format(len(filtered_snapshot_indices), len(traj), ligand_filter_cutoff))

    # Filter out snapshots where ligand is too far from the protein
    filtered_traj = traj[filtered_snapshot_indices]
    filtered_w_n = np.array(w_n[filtered_snapshot_indices])
    filtered_w_n /= filtered_w_n.sum() # renormalize

    # Align receptor to first frame
    atoms_to_align = filtered_traj.top.select(receptor_dsl_selection)
    if (len(atoms_to_align) == 0):
        raise Exception("Please check receptor_dsl_selection since no atoms were found in selection!")
    logger.info('aligning on {:d} atoms from receptor_dsl_selection'.format(len(atoms_to_align)))
    aligned_traj = filtered_traj.superpose(filtered_traj[0], frame=0, atom_indices=atoms_to_align)

    # Extract ligand trajectory
    ligand_atom_indices = aligned_traj.topology.select(ligand_dsl_selection)
    ligand_trajectory = aligned_traj.atom_slice(ligand_atom_indices)
    logger.info('{:d} atoms in ligand trajectory'.format(len(ligand_atom_indices)))

    # Perform regular spatial clustering on ligand trajectory
    nsnapshots, natoms, _ = ligand_trajectory.xyz.shape
    x = np.array(ligand_trajectory.xyz).reshape([nsnapshots, natoms*3], order='C')
    reg_space = RegularSpatial(d_min=3*natoms*ligand_rmsd_cutoff**2, metric='sqeuclidean').fit([x])
    cluster_assignments = reg_space.fit_predict([x])[0]
    nclusters = cluster_assignments.max() + 1
    logger.info('There are {:d} clusters'.format(nclusters))

    # Sort clusters by probability
    cluster_probabilities = np.zeros([nclusters], np.float64)
    for cluster_index in range(nclusters):
        snapshot_indices = np.where(cluster_assignments == cluster_index)[0]
        cluster_probabilities[cluster_index] = filtered_w_n[snapshot_indices].sum()
    # Permute clusters
    sorted_indices = np.argsort(-cluster_probabilities)
    new_cluster_assignments = np.array(cluster_assignments)
    for cluster_index in range(nclusters):
        indices = np.where(cluster_assignments == sorted_indices[cluster_index])[0]
        new_cluster_assignments[indices] = cluster_index
    cluster_assignments = new_cluster_assignments
    cluster_probabilities = cluster_probabilities[sorted_indices]

    # Write cluster populations
    for cluster_index in range(nclusters):
        logger.info('Cluster {:5d} : {:12.8f}'.format(cluster_index, cluster_probabilities[cluster_index]))

    cumsum = np.cumsum(cluster_probabilities)
    cutoff_index = np.where(cumsum > cluster_filter_threshold)[0][0] # first index where weight is below threshold
    nclusters = max(cutoff_index, 1)
    logger.info('There are {:d} clusters after retaining only those where the cumulative weight exceeds {:f}'.format(nclusters, cluster_filter_threshold))

    # Write reference protein conformation
    receptor_atom_indices = aligned_traj.topology.select('protein')
    filename = '%s-reference.pdb' % (output_prefix)
    logger.info('Writing reference coordinates to {}'.format(filename))
    aligned_traj[0].atom_slice(receptor_atom_indices).save(filename)

    # Write aligned frames
    for cluster_index in range(nclusters):
        indices = np.where(cluster_assignments == cluster_index)[0]
        # Remove indices with zero probability
        indices = indices[filtered_w_n[indices] > 0.0]

        nsnapshots = len(indices)
        logger.info('Cluster {:5d} : pop {:12.8f} : {:8d} members'.format(cluster_index, cluster_probabilities[cluster_index], nsnapshots))

        # Sample frames
        filename = '%s-cluster%03d.pdb' % (output_prefix, cluster_index)
        logger.info('   writing {}'.format(filename))
        if nsnapshots <= nsnapshots_per_cluster:
            aligned_traj[indices].save(filename)
        else:
            p = filtered_w_n[indices] / filtered_w_n[indices].sum()
            sampled_indices = np.random.choice(indices, size=nsnapshots_per_cluster, p=p, replace=False)
            aligned_traj[sampled_indices].save('%s-cluster%03d.pdb' % (output_prefix, cluster_index))
    # Write cluster populations to a file
    filename = '%s-populations.txt' % (output_prefix)
    logger.info('Writing populations to {}'.format(filename))
    outfile = open(filename, 'w')
    for cluster_index in range(nclusters):
        outfile.write('%05d %12.8f\n' % (cluster_index, cluster_probabilities[cluster_index]))
    outfile.close()
