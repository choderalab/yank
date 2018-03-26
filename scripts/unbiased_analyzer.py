#!/usr/local/bin/env python

import os
import copy
import scipy
import logging
import collections

import yaml
import numpy as np

import mdtraj
import pymbar
from simtk import openmm, unit as unit
import openmmtools as mmtools
import yank.restraints
from yank import analyze, repex, utils

logger = logging.getLogger(__name__)


def get_restraint_force(system):
    """Extract the radially symmetric restraint Custom(Centroid)BondForce of the system."""
    restraint_force = None
    for i, force in enumerate(system.getForces()):
        if force.__class__.__name__ in ['CustomCentroidBondForce', 'CustomBondForce']:
            if force.getGlobalParameterName(0) == 'lambda_restraints':
                restraint_force = copy.deepcopy(force)
                break
    return restraint_force


def set_restrained_particles(restraint_force, particles1, particles2):
    try:
        # CustomCentroidBondForce
        restraint_force.setGroupParameters(0, list(particles1))
        restraint_force.setGroupParameters(1, list(particles2))
    except AttributeError:
        # CustomBondForce
        _, _, bond_parameters = restraint_force.getBondParameters(0)
        restraint_force.setBondParameters(0, particles1[0], particles2[0], bond_parameters)


def compute_centroid_distance(positions_group1, positions_group2, weights_group1, weights_group2):
    """Compute the distance between the centers of mass of the two groups.

    The two positions given must have the same units.

    Parameters
    ----------
    positions_group1 : numpy.array
        The positions of the particles in the first CustomCentroidBondForce group.
    positions_group2 : numpy.array
        The positions of the particles in the second CustomCentroidBondForce group.
    weights_group1 : list of float
        The mass of the particle in the first CustomCentroidBondForce group.
    weights_group2 : list of float
        The mass of the particles in the second CustomCentroidBondForce group.

    """
    assert len(positions_group1) == len(weights_group1)
    assert len(positions_group2) == len(weights_group2)
    # Compute center of mass for each group.
    com_group1 = np.average(positions_group1, axis=0, weights=weights_group1)
    com_group2 = np.average(positions_group2, axis=0, weights=weights_group2)
    # Compute distance between centers of mass.
    distance = np.linalg.norm(com_group1 - com_group2)
    return distance


class UnbiasedAnalyzer(analyze.ReplicaExchangeAnalyzer):

    def __init__(self, *args, restraint_energy_cutoff=None, restraint_distance_cutoff=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._restraint_energy_cutoff = restraint_energy_cutoff
        self._restraint_distance_cutoff = restraint_distance_cutoff
        self._n_iterations = None
        self._n_equilibration_iterations = None
        self._state_indices_kn = None
        self._restraint_data = None
        self._distances_kn = None
        self._energies_kn = None

    @property
    def n_iterations(self):
        if self._n_iterations is None:
            # The + 1 accounts for iteration 0.
            self._n_iterations = self._reporter.read_last_iteration(full_iteration=False) + 1
        return self._n_iterations

    @property
    def n_equilibration_iterations(self):
        if self._equilibration_data is None:
            self._get_equilibration_data_auto()
        return self._equilibration_data[0]

    @property
    def statistical_inefficiency(self):
        if self._equilibration_data is None:
            self._get_equilibration_data_auto()
        return self._equilibration_data[1]

    @property
    def uncorrelated_iterations(self):
        equilibrium_iterations = np.array(range(self.n_equilibration_iterations, self.n_iterations))
        uncorrelated_iterations_indices = pymbar.timeseries.subsampleCorrelatedData(equilibrium_iterations,
                                                                                    self.statistical_inefficiency)
        return equilibrium_iterations[uncorrelated_iterations_indices]

    @property
    def state_indices_kn(self):
        """Return the uncorrelated replica state indices in kn format."""
        if self._state_indices_kn is None:
            uncorrelated_iterations = self.uncorrelated_iterations  # Shortcut.
            replica_state_indices = self._reporter.read_replica_thermodynamic_states()
            n_correlated_iterations, n_replicas = replica_state_indices.shape

            # Initialize output array.
            n_frames = n_replicas * len(uncorrelated_iterations)
            self._state_indices_kn = np.zeros(n_frames, dtype=np.int32)

            # Map kn columns to the sta
            for iteration_idx, iteration in enumerate(uncorrelated_iterations):
                for replica_idx in range(n_replicas):
                    # Deconvolute index.
                    state_idx = replica_state_indices[iteration, replica_idx]
                    frame_idx = state_idx*len(uncorrelated_iterations) + iteration_idx
                    # Set output array.
                    self._state_indices_kn[frame_idx] = state_idx
        return self._state_indices_kn

    @property
    def restraint_energy_cutoff(self):
        return self._restraint_energy_cutoff

    @restraint_energy_cutoff.setter
    def restraint_energy_cutoff(self, new_value):
        self._restraint_energy_cutoff = new_value
        if self._mbar is not None:
            self._invalidate_observables()
            u_kn, N_k = self._compute_unbiased_mbar_data()
            self._create_mbar(u_kn, N_k)

    @property
    def restraint_distance_cutoff(self):
        return self._restraint_distance_cutoff

    @restraint_distance_cutoff.setter
    def restraint_distance_cutoff(self, new_value):
        self._restraint_distance_cutoff = new_value
        if self._mbar is not None:
            self._invalidate_observables()
            u_kn, N_k = self._compute_unbiased_mbar_data()
            self._create_mbar(u_kn, N_k)

    def _read_thermodynamic_states(self):
        """Read thermodynamic states and caches useful info in the meantime."""
        thermodynamic_states, unsampled_states = self._reporter.read_thermodynamic_states()
        # TODO should we read all temperatures and let kT property depend on reference_states?
        self._kT = unsampled_states[0].kT  # Avoid reading TS again when we need kT.
        return thermodynamic_states, unsampled_states

    def _invalidate_observables(self):
        for observable in self.observables:
            self._computed_observables[observable] = None

    def _prepare_mbar_input_data(self, sampled_energy_matrix, unsampled_energy_matrix):
        """Convert the sampled and unsampled energy matrices into MBAR ready data"""
        u_kln, N_k = super()._prepare_mbar_input_data(sampled_energy_matrix, unsampled_energy_matrix)
        u_kn = pymbar.utils.kln_to_kn(u_kln, N_k)
        self._uncorrelated_u_kn = u_kn
        self._uncorrelated_N_k = N_k
        return self._compute_unbiased_mbar_data()

    def _create_mbar(self, energy_matrix, samples_per_state):
        # TODO: the original _create_mbar resets the observables, which deletes the
        # TODO:     standard state correction computed in _prepare_mbar_input_data().
        # Delete observables cache since we are now resetting the estimator
        # for observable in self.observables:
        #     self._computed_observables[observable] = None

        # Initialize MBAR (computing free energy estimates, which may take a while)
        logger.info("Computing free energy differences...")
        mbar = pymbar.MBAR(energy_matrix, samples_per_state, **self._extra_analysis_kwargs)
        self._mbar = mbar

    def _compute_unbiased_mbar_data(self):
        """Unbias the restraint and apply energy/distance cutoff."""
        # Shortcut.
        u_kn, N_k = copy.deepcopy(self._uncorrelated_u_kn), copy.deepcopy(self._uncorrelated_N_k)

        # Check if we need to unbias the restraint.
        compute_distances = self._restraint_distance_cutoff is not None

        # Isolate part of the system used to re-compute restraint energies/distances.
        try:
            restraint_data = self._get_restraint_data()
        except TypeError as e:
            logger.info(str(e) + ' The restraint will not be unbiased.')
            return u_kn, N_k
        reduced_system, restraint_force = restraint_data[:2]
        particle_indices_group1, particle_indices_group2 = restraint_data[2:4]
        weights_group1, weights_group2 = restraint_data[4:]

        # Recompute unbiased SSC with given cutoffs.
        # TODO: This code is redundant with yank.py.
        # TODO: Compute average box volume here?
        box_vectors = reduced_system.getDefaultPeriodicBoxVectors()
        box_volume = mmtools.states._box_vectors_volume(box_vectors)
        ssc = - np.log(yank.restraints.V0 / box_volume)
        if self._restraint_distance_cutoff is not None or self._restraint_energy_cutoff is not None:
            max_dimension = np.max(unit.Quantity(box_vectors) / unit.nanometers) * unit.nanometers
            ssc_cutoff = self._compute_standard_state_correction(restraint_force, unbiased=True, max_distance=max_dimension)
            # The restraint volume can't be bigger than the box volume.
            if ssc_cutoff < ssc:
                ssc = ssc_cutoff

        self._computed_observables['standard_state_correction'] = ssc
        logger.debug('New standard state correction: {} kT'.format(ssc))

        # Compute restraint energies/distances.
        distances_kn, energies_kn = self._compute_restrain_energies(particle_indices_group1, particle_indices_group2,
                                                                    weights_group1, weights_group2, reduced_system,
                                                                    compute_distances=compute_distances)

        # Convert energies to kT unit for comparison to energy cutoff.
        energies_kn = energies_kn / self.kT
        logger.debug('Restraint energy mean: {} kT; std: {} kT'
                     ''.format(np.mean(energies_kn), np.std(energies_kn, ddof=1)))

        # Convert energies to u_kn format.
        assert len(energies_kn) == u_kn.shape[1]

        # We need to take into account the initial unsampled states to index correctly N_k.
        state_idx_shift = 0
        while N_k[state_idx_shift] == 0:
            state_idx_shift +=1

        # Determine samples outside the cutoffs.
        columns_to_keep = []
        for iteration_kn, energy in enumerate(energies_kn):
            if ((self._restraint_energy_cutoff is not None and energy > self._restraint_energy_cutoff) or
                    (compute_distances and distances_kn[iteration_kn] > self._restraint_distance_cutoff)):
                # Update the number of samples generated from its state.
                state_idx = self.state_indices_kn[iteration_kn]
                N_k[state_idx + state_idx_shift] -= 1
            else:
                columns_to_keep.append(iteration_kn)

        # Drop all columns that exceed the cutoff(s).
        n_discarded = len(energies_kn) - len(columns_to_keep)
        logger.debug('Discarding {}/{} samples outside the cutoff.'.format(n_discarded, len(energies_kn)))
        u_kn = u_kn[:, columns_to_keep]
        energies_kn = energies_kn[columns_to_keep]

        # Add new end states that don't include the restraint.
        n_states, n_iterations = u_kn.shape
        n_states_new = n_states + 2
        N_k_new = np.zeros(n_states_new, N_k.dtype)
        u_kn_new = np.zeros((n_states_new, n_iterations), u_kn.dtype)
        u_kn_new[0, :] = u_kn[0] - energies_kn
        u_kn_new[-1, :] = u_kn[-1] - energies_kn

        # Copy old values.
        N_k_new[1:-1] = N_k
        u_kn_new[1:-1, :] = u_kn

        return u_kn_new, N_k_new

    def _get_restraint_data(self):
        """Return the two unsampled states and a reduced version of them containing only the restraint force."""
        # Check cached value.
        if self._restraint_data is not None:
            return copy.deepcopy(self._restraint_data)

        # Isolate the end states.
        sampled_states, unsampled_states = self._read_thermodynamic_states()
        if len(unsampled_states) == 0:
            end_states = [sampled_states[0], sampled_states[-1]]
        else:
            end_states = unsampled_states

        # Isolate restraint force.
        system = end_states[0].system
        restraint_force = get_restraint_force(system)

        # Check this is a radially symmetric restraint and it was turned on at the end states.
        if restraint_force is None:
            raise TypeError('Cannot find a radially symmetric restraint.')
        if end_states[0].lambda_restraints != 1.0 or end_states[-1].lambda_restraints != 1.0:
            raise TypeError('Cannot unbias a restraint that is turned off at one of the end states.')

        # Log bond parameters.
        bond_parameters = restraint_force.getBondParameters(0)[-1]
        try:  # FlatBottom
            logger.debug('Bond parameters: K={}, r0={}'.format(*bond_parameters))
        except IndexError:  # Harmonic
            logger.debug('Bond parameters: K={}'.format(*bond_parameters))

        # Obtain restraint's particle indices to compute restraint distance.
        try:
            # CustomCentroidBondForce
            particle_indices_group1, weights_group1 = restraint_force.getGroupParameters(0)
            particle_indices_group2, weights_group2 = restraint_force.getGroupParameters(1)
            assert len(weights_group1) == 0  # Use masses to compute centroid.
            assert len(weights_group2) == 0  # Use masses to compute centroid.
        except AttributeError:
            # CustomBondForce
            particle_indices_group1, particle_indices_group2, _ = restraint_force.getBondParameters(0)
            particle_indices_group1 = [particle_indices_group1]  # Convert to list.
            particle_indices_group2 = [particle_indices_group2]  # Convert to list.

        # Convert tuples of np.integers to lists of ints.
        particle_indices_group1 = [int(i) for i in sorted(particle_indices_group1)]
        particle_indices_group2 = [int(i) for i in sorted(particle_indices_group2)]
        logger.debug('receptor restrained atoms: {}'.format(particle_indices_group1))
        logger.debug('ligand restrained atoms: {}'.format(particle_indices_group2))

        # Create new system with only solute and restraint forces.
        reduced_system = openmm.System()
        for particle_indices_group in [particle_indices_group1, particle_indices_group2]:
            for i in particle_indices_group:
                reduced_system.addParticle(system.getParticleMass(i))
        reduced_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())

        # Compute weights restrained particles.
        weights_group1 = [system.getParticleMass(i) for i in particle_indices_group1]
        weights_group2 = [system.getParticleMass(i) for i in particle_indices_group2]

        # Adapt the restraint force atom indices to the reduced system.
        assert max(particle_indices_group1) < min(particle_indices_group2)
        n_atoms_group1 = len(particle_indices_group1)
        tot_n_atoms = n_atoms_group1 + len(particle_indices_group2)
        restraint_force = copy.deepcopy(restraint_force)
        set_restrained_particles(restraint_force, particles1=range(n_atoms_group1),
                                 particles2=range(n_atoms_group1, tot_n_atoms))
        reduced_system.addForce(restraint_force)

        self._restraint_data = (reduced_system, restraint_force,
                                particle_indices_group1, particle_indices_group2,
                                weights_group1, weights_group2)
        return copy.deepcopy(self._restraint_data)

    def _compute_restrain_energies(self, particle_indices_group1, particle_indices_group2,
                                   weights_group1, weights_group2, reduced_system,
                                   compute_distances=False):
        """Compute the restrain distances for the given iterations.

        Parameters
        ----------
        particle_indices_group1 : list of int
            The particle indices of the first CustomCentroidBondForce group.
        particle_indices_group2 : list of int
            The particle indices of the second CustomCentroidBondForce group.
        weights_group1 : list of float
            The mass of the particle in the first CustomCentroidBondForce group.
        weights_group2 : list of float
            The mass of the particles in the second CustomCentroidBondForce group.

        Returns
        -------
        restrain_distances_kn : np.array
            The restrain distances.
        """
        #Check cached values.
        distances_kn = None
        if compute_distances and self._distances_kn is not None:
            distances_kn = self._distances_kn
            compute_distances = False
        if compute_distances is False and self._energies_kn is not None:
            return copy.deepcopy(distances_kn), copy.deepcopy(self._energies_kn)

        uncorrelated_iterations = self.uncorrelated_iterations  # Shortcut.

        # subset_particles_indices = list(self._reporter.analysis_particle_indices)
        subset_particles_indices = particle_indices_group1 + particle_indices_group2
        replica_state_indices = self._reporter.read_replica_thermodynamic_states()
        n_correlated_iterations, n_replicas = replica_state_indices.shape

        # Create output arrays. We unfold the replicas the same way
        # it is done during the kln_to_kn conversion.
        n_frames = n_replicas * len(uncorrelated_iterations)
        energies_kn = np.zeros(n_frames, dtype=np.float64) * unit.kilojoules_per_mole

        if compute_distances:
            n_atoms = len(subset_particles_indices)
            n_atoms_group1 = len(particle_indices_group1)
            traj_particle_indices_group1 = list(range(n_atoms_group1))
            traj_particle_indices_group2 = list(range(n_atoms_group1, n_atoms))

            # Create topology of the restrained atoms.
            serialized_topography = self._reporter.read_dict('metadata/topography')
            topology = mmtools.utils.deserialize(serialized_topography).topology
            topology = topology.subset(subset_particles_indices)

            distances_kn = np.zeros(n_frames, dtype=np.float32)
            # Initialize trajectory object needed for imaging molecules.
            trajectory = mdtraj.Trajectory(xyz=np.zeros((n_atoms, 3)), topology=topology)

        # Create context used to compute the energies.
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = openmm.Context(reduced_system, integrator)

        # Pre-computing distances.
        logger.debug('Computing restraint energies...')
        for iteration_idx, iteration in enumerate(uncorrelated_iterations):
            # Obtain solute only sampler states.
            sampler_states = self._reporter.read_sampler_states(iteration=iteration,
                                                                analysis_particles_only=True)

            for replica_idx, sampler_state in enumerate(sampler_states):
                # Deconvolute index.
                state_idx = replica_state_indices[iteration, replica_idx]
                frame_idx = state_idx*len(uncorrelated_iterations) + iteration_idx

                sliced_sampler_state = sampler_state[subset_particles_indices]
                sliced_sampler_state.apply_to_context(context)
                potential_energy = context.getState(getEnergy=True).getPotentialEnergy()
                energies_kn[frame_idx] = potential_energy

                if compute_distances:
                    # Update trajectory positions/box vectors.
                    trajectory.xyz = (sampler_state.positions[subset_particles_indices] / unit.nanometers).astype(np.float32)
                    trajectory.unitcell_vectors = np.array([sampler_state.box_vectors / unit.nanometers], dtype=np.float32)
                    trajectory.image_molecules(inplace=True, make_whole=False)
                    positions_group1 = trajectory.xyz[0][traj_particle_indices_group1]
                    positions_group2 = trajectory.xyz[0][traj_particle_indices_group2]

                    # Set output arrays.
                    distances_kn[frame_idx] = compute_centroid_distance(positions_group1, positions_group2,
                                                                        weights_group1, weights_group2)

        # Set MDTraj units to distances.
        if compute_distances:
            distances_kn = distances_kn * unit.nanometer
        self._distances_kn = distances_kn
        self._energies_kn = energies_kn
        return copy.deepcopy(self._distances_kn), copy.deepcopy(self._energies_kn)

    def _compute_standard_state_correction(self, restraint_force, unbiased=False, max_distance=None):
        """Compute the standard state correction."""
        # TODO refactor: the redundant code with yank.restraints.RadiallySymmetricRestraint._standard_state_correction
        r_min = 0 * unit.nanometers
        if self._restraint_distance_cutoff is not None:
            r_max = self._restraint_distance_cutoff
        elif max_distance is not None:
            r_max = max_distance
        else:
            r_max = 100 * unit.nanometers

        # Create a System object containing two particles connected by the reference force
        system = openmm.System()
        system.addParticle(1.0 * unit.amu)
        system.addParticle(1.0 * unit.amu)
        force = copy.deepcopy(restraint_force)
        set_restrained_particles(force, particles1=[0], particles2=[1])
        # Disable the PBC if on for this approximation of the analytical solution
        force.setUsesPeriodicBoundaryConditions(False)
        system.addForce(force)

        # Create a Reference context to evaluate energies on the CPU.
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system, integrator, platform)

        # Set default positions.
        positions = unit.Quantity(np.zeros([2,3]), unit.nanometers)
        context.setPositions(positions)

        # Create a function to compute integrand as a function of interparticle separation.
        beta = 1 / self.kT

        def integrand(r):
            """
            Parameters
            ----------
            r : float
                Inter-particle separation in nanometers

            Returns
            -------
            dI : float
               Contribution to integrand (in nm^2).

            """
            positions[1, 0] = r * unit.nanometers
            context.setPositions(positions)
            state = context.getState(getEnergy=True)
            potential = beta * state.getPotentialEnergy()  # In kT.

            if (self._restraint_energy_cutoff is not None and
                        potential > self._restraint_energy_cutoff):
                return 0.0
            elif unbiased:
                potential = 0.0

            dI = 4.0 * np.pi * r**2 * np.exp(-potential)
            return dI

        # Integrate shell volume.
        shell_volume, shell_volume_error = scipy.integrate.quad(lambda r: integrand(r), r_min / unit.nanometers,
                                                                r_max / unit.nanometers) * unit.nanometers**3

        # Compute standard-state volume for a single molecule in a box of
        # size (1 L) / (avogadros number). Should also generate constant V0.
        liter = 1000.0 * unit.centimeters**3  # one liter
        standard_state_volume = liter / (unit.AVOGADRO_CONSTANT_NA*unit.mole)  # standard state volume

        # Compute standard state correction for releasing shell restraints into standard-state box (in units of kT).
        DeltaG = - np.log(standard_state_volume / shell_volume)

        # Return standard state correction (in kT).
        return DeltaG


def analyze_phase(analyzer):
    data = dict()
    Deltaf_ij, dDeltaf_ij = analyzer.get_free_energy()
    data['DeltaF'] = Deltaf_ij[analyzer.reference_states[0], analyzer.reference_states[1]]
    data['dDeltaF'] = dDeltaf_ij[analyzer.reference_states[0], analyzer.reference_states[1]]
    data['DeltaF_standard_state_correction'] = analyzer.get_standard_state_correction()
    return data


def analyze_directory(source_directory, energy_cutoffs=None, distance_cutoffs=None, solvent_df=None, solvent_ddf=None):
    # Handle default value.
    if isinstance(energy_cutoffs, collections.Iterable):
        cutoffs = energy_cutoffs
        cutoff_attribute = 'restraint_energy_cutoff'
    elif isinstance(distance_cutoffs, collections.Iterable):
        cutoffs = distance_cutoffs
        cutoff_attribute = 'restraint_distance_cutoff'
    else:
        raise ValueError('One between energy or distance cutoff must be specified.')

    complex_phase_names = ['complex-' + str(cutoff) for cutoff in cutoffs]

    analysis_script_path = os.path.join(source_directory, 'analysis.yaml')
    with open(analysis_script_path, 'r') as f:
        analysis = yaml.load(f)

    data = dict()
    for phase_name, sign in analysis:
        # Avoid recomputing the solvent phase if known.
        if phase_name == 'solvent' and solvent_df is not None and solvent_ddf is not None:
            data[phase_name] = {}
            data[phase_name]['DeltaF'] = solvent_df
            data[phase_name]['dDeltaF'] = solvent_ddf
            data[phase_name]['DeltaF_standard_state_correction'] = 0.0
            continue

        phase_path = os.path.join(source_directory, phase_name + '.nc')
        reporter = repex.Reporter(phase_path, open_mode='r')
        phase = UnbiasedAnalyzer(reporter)
        kT = phase.kT

        # For the complex phase, analyze at all cutoffs.
        if phase_name == 'complex':
            for complex_phase_name, cutoff in zip(complex_phase_names, cutoffs):
                setattr(phase, cutoff_attribute, cutoff)
                data[complex_phase_name] = analyze_phase(phase)
        else:
            data[phase_name] = analyze_phase(phase)

    kT_to_kcalmol = kT / unit.kilocalories_per_mole

    # Compute free energy and enthalpy for all cutoffs.
    all_free_energies = []
    all_sscs = []
    phase_names = [phase_name for phase_name, sign in analysis]
    for complex_phase_name, cutoff in zip(complex_phase_names, cutoffs):
        DeltaF = 0.0
        dDeltaF = 0.0
        for phase_name, sign in analysis:
            if phase_name == 'complex':
                phase_name = complex_phase_name
            DeltaF -= sign * (data[phase_name]['DeltaF'] + data[phase_name]['DeltaF_standard_state_correction'])
            dDeltaF += data[phase_name]['dDeltaF']**2
        dDeltaF = np.sqrt(dDeltaF)

        all_free_energies.append((DeltaF * kT_to_kcalmol, dDeltaF * kT_to_kcalmol))
        all_sscs.append(data[complex_phase_name]['DeltaF_standard_state_correction'] * kT_to_kcalmol)

        # Attempt to guess type of calculation
        calculation_type = ''
        for phase in phase_names:
            if 'complex' in phase:
                calculation_type = ' of binding'
            elif 'solvent1' in phase:
                calculation_type = ' of solvation'

        # Print energies
        logger.info('')
        logger.info('Reporting free energy for cutoff: {}'.format(cutoff))
        logger.info('-------------------------------------')
        logger.info('Free energy{:<13}: {:9.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(
            calculation_type, DeltaF, dDeltaF, DeltaF * kT / unit.kilocalories_per_mole,
            dDeltaF * kT / unit.kilocalories_per_mole))

        for phase in phase_names:
            if phase == 'complex':
                phase = complex_phase_name
            logger.info('DeltaG {:<17}: {:9.3f} +- {:.3f} kT'.format(phase, data[phase]['DeltaF'],
                                                                     data[phase]['dDeltaF']))
            if data[phase]['DeltaF_standard_state_correction'] != 0.0:
                logger.info('DeltaG {:<17}: {:18.3f} kT'.format('restraint',
                                                                data[phase]['DeltaF_standard_state_correction']))

    print('Free energies (kcal/mol):', all_free_energies)
    print('SSC (kcal/mol)', all_sscs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyzer to unbias radially symmetric restraints.')
    parser.add_argument('-s', '--store', metavar='store', type=str, help='Storage directory for NetCDF data files.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--energy-cutoff', metavar='energy_cutoff', type=float,
                        default=None, help='Energy cutoff in kT units.')
    parser.add_argument('--distance-cutoff', metavar='distance_cutoff', type=float,
                        default=None, help='Distance cutoff in nanometers.')
    args = parser.parse_args()

    if args.verbose:
        utils.config_root_logger(verbose=True)
    else:
        utils.config_root_logger(verbose=False)

    if args.energy_cutoff:
        energy_cutoff = args.energy_cutoff
    else:
        energy_cutoff = None

    if args.distance_cutoff:
        distance_cutoff = args.distance_cutoff * unit.nanometers
    else:
        distance_cutoff = None

    # The function doesn't currently support both cutoff and needs to be fixed.
    if energy_cutoff is not None and distance_cutoff is not None:
        raise ValueError('Only one between energy and distance cutoff can be specified.')

    analyze_directory(args.store, energy_cutoff, distance_cutoff)
