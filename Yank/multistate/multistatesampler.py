#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
MultistateSampler
=================

Base multi-thermodynamic state multistate class

COPYRIGHT

Current version by Andrea Rizzi <andrea.rizzi@choderalab.org>, Levi N. Naden <levi.naden@choderalab.org> and
John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

Original version by John D. Chodera <jchodera@gmail.com> while at the University of
California Berkeley.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import os
import copy
import time
import typing
import inspect
import logging
import datetime

import numpy as np
from simtk import unit, openmm

import openmmtools as mmtools

from .multistatereporter import MultiStateReporter
from .utils import SimulationNaNError

from .. import mpi
from pymbar.utils import ParameterError

logger = logging.getLogger(__name__)


# ==============================================================================
# MULTISTATE SAMPLER
# ==============================================================================

class MultiStateSampler(object):
    """
    Base class for samplers that sample multiple thermodynamic states using
    one or more replicas.

    This base class provides a general simulation facility for multistate from multiple
    thermodynamic states, allowing any set of thermodynamic states to be specified.
    If instantiated on its own, the thermodynamic state indices associated with each
    state are specified and replica mixing does not change any thermodynamic states,
    meaning that each replica remains in its original thermodynamic state.

    Stored configurations, energies, swaps, and restart information are all written
    to a single output file using the platform portable, robust, and efficient
    NetCDF4 library.

    Parameters
    ----------
    mcmc_moves : MCMCMove or list of MCMCMove, optional
        The MCMCMove used to propagate the thermodynamic states. If a list of MCMCMoves,
        they will be assigned to the correspondent thermodynamic state on
        creation. If None is provided, Langevin dynamics with 2fm timestep, 5.0/ps collision rate,
        and 500 steps per iteration will be used.
    number_of_iterations : int or infinity, optional, default: 1
        The number of iterations to perform. Both ``float('inf')`` and
        ``numpy.inf`` are accepted for infinity. If you set this to infinity,
        be sure to set also ``online_analysis_interval``.
    online_analysis_interval : None or Int >= 1, optional, default None
        Choose the interval at which to perform online analysis of the free energy.

        After every interval, the simulation will be stopped and the free energy estimated.

        If the error in the free energy estimate is at or below ``online_analysis_target_error``, then the simulation
        will be considered completed.

    online_analysis_target_error : float >= 0, optional, default 0.2
        The target error for the online analysis measured in kT per phase.

        Once the free energy is at or below this value, the phase will be considered complete.

        If ``online_analysis_interval`` is None, this option does nothing.

    online_analysis_minimum_iterations : int >= 0, optional, default 50
        Set the minimum number of iterations which must pass before online analysis is carried out.

        Since the initial samples likely not to yield a good estimate of free energy, save time and just skip them
        If ``online_analysis_interval`` is None, this does nothing

    locality : int > 0, optional, default None
        If None, the energies at all states will be computed for every replica each iteration.
        If int > 0, energies will only be computed for states ``range(max(0, state-locality), min(n_states, state+locality))``.

    Attributes
    ----------
    n_replicas
    n_states
    iteration
    mcmc_moves
    sampler_states
    metadata
    is_completed

    :param number_of_iterations: Maximum number of integer iterations that will be run

    :param online_analysis_interval: How frequently to carry out online analysis in number of iterations

    :param online_analysis_target_error: Target free energy difference error float at which simulation will be stopped during online analysis, in dimensionless energy

    :param online_analysis_minimum_iterations: Minimum number of iterations needed before online analysis is run as int

    """

    # -------------------------------------------------------------------------
    # Constructors.
    # -------------------------------------------------------------------------

    def __init__(self, mcmc_moves=None, number_of_iterations=1,
                 online_analysis_interval=None, online_analysis_target_error=0.2,
                 online_analysis_minimum_iterations=50,
                 locality=None):
        # These will be set on initialization. See function
        # create() for explanation of single variables.
        self._thermodynamic_states = None
        self._unsampled_states = None
        self._sampler_states = None
        self._replica_thermodynamic_states = None
        self._iteration = None
        self._energy_thermodynamic_states = None
        self._neighborhoods = None
        self._energy_unsampled_states = None
        self._n_accepted_matrix = None
        self._n_proposed_matrix = None
        self._reporter = None
        self._metadata = None

        # Handling default propagator.
        if mcmc_moves is None:
            # This will be converted to a list in create().
            self._mcmc_moves = mmtools.mcmc.LangevinDynamicsMove(timestep=2.0 * unit.femtosecond,
                                                                 collision_rate=5.0 / unit.picosecond,
                                                                 n_steps=500, reassign_velocities=True,
                                                                 n_restart_attempts=6)
        else:
            self._mcmc_moves = copy.deepcopy(mcmc_moves)

        # Store constructor parameters. Everything is marked for internal
        # usage because any change to these attribute implies a change
        # in the storage file as well. Use properties for checks.
        self.number_of_iterations = number_of_iterations

        # Store locality
        self.locality = locality

        # Online analysis options.
        self.online_analysis_interval = online_analysis_interval
        self.online_analysis_target_error = online_analysis_target_error
        self.online_analysis_minimum_iterations = online_analysis_minimum_iterations
        self._online_error_trap_counter = 0  # Counter for errors in the online estimate
        self._online_error_bank = []

        self._last_mbar_f_k = None
        self._last_err_free_energy = None

        self._have_displayed_citations_before = False

        # Check convergence.
        if self.number_of_iterations == np.inf:
            if self.online_analysis_target_error == 0.0:
                logger.warning("WARNING! You have specified an unlimited number of iterations and a target error "
                               "for online analysis of 0.0! Your simulation may never reach 'completed' state!")
            elif self.online_analysis_interval is None:
                logger.warning("WARNING! This simulation will never be considered 'complete' since there is no "
                               "specified maximum number of iterations!")

    @classmethod
    def from_storage(cls, storage):
        """Constructor from an existing storage file.

        Parameters
        ----------
        storage : str or Reporter
            If str: The path to the storage file.
            If :class:`Reporter`: uses the :class:`Reporter` options
            In the future this will be able to take a Storage class as well.

        Returns
        -------
        sampler : MultiStateSampler
            A new instance of MultiStateSampler (or subclass) in the same state of the
            last stored iteration.

        """
        # Handle case in which storage is a string.
        reporter = cls._reporter_from_storage(storage, check_exist=True)

        # Open a reporter to read the data.
        reporter.open(mode='r')

        # Retrieve options and create new simulation.
        options = reporter.read_dict('options')
        options['mcmc_moves'] = reporter.read_mcmc_moves()
        sampler = cls(**options)

        # Display papers to be cited.
        sampler._display_citations()

        # Read the last iteration reported to ensure we don't include junk
        # data written just before a crash.
        iteration = reporter.read_last_iteration()

        # Retrieve other attributes.
        logger.debug("Reading storage file {}...".format(reporter.filepath))
        thermodynamic_states, unsampled_states = reporter.read_thermodynamic_states()
        sampler_states = reporter.read_sampler_states(iteration=iteration)
        state_indices = reporter.read_replica_thermodynamic_states(iteration=iteration)
        energy_thermodynamic_states, neighborhoods, energy_unsampled_states = reporter.read_energies(iteration=iteration)
        n_accepted_matrix, n_proposed_matrix = reporter.read_mixing_statistics(iteration=iteration)
        metadata = reporter.read_dict('metadata')

        # Search for last cached free energies only if online analysis is activated.
        if sampler.online_analysis_interval is not None:
            online_analysis_info = sampler._read_last_free_energy(reporter, iteration)
            last_mbar_f_k, (_, last_err_free_energy) = online_analysis_info
        else:
            last_mbar_f_k, last_err_free_energy = None, None

        # Close reading reporter.
        reporter.close()

        # Assign attributes.
        sampler._iteration = iteration
        sampler._thermodynamic_states = thermodynamic_states
        sampler._unsampled_states = unsampled_states
        sampler._sampler_states = sampler_states
        sampler._replica_thermodynamic_states = state_indices
        sampler._energy_thermodynamic_states = energy_thermodynamic_states
        sampler._neighborhoods = neighborhoods
        sampler._energy_unsampled_states = energy_unsampled_states
        sampler._n_accepted_matrix = n_accepted_matrix
        sampler._n_proposed_matrix = n_proposed_matrix
        sampler._metadata = metadata

        sampler._last_mbar_f_k = last_mbar_f_k
        sampler._last_err_free_energy = last_err_free_energy

        # We open the reporter only in node 0.
        sampler._reporter = reporter
        mpi.run_single_node(0, sampler._reporter.open, mode='a',
                            broadcast_result=False, sync_nodes=False)

        # Don't write the new last iteration, we have not technically written anything yet, so there is no "junk"
        return sampler

    # TODO use Python 3.6 namedtuple syntax when we drop Python 3.5 support.
    Status = typing.NamedTuple('Status', [
        ('iteration', int),
        ('target_error', float),
        ('is_completed', bool)
    ])

    @classmethod
    def read_status(cls, storage):
        """Read the status of the calculation from the storage file.

        This class method can be used to quickly check the status of the
        simulation before loading the full ``ReplicaExchange`` object
        from disk.

        Parameters
        ----------
        storage : str or Reporter
            The path to the storage file or the reporter object.

        Returns
        -------
        status : ReplicaExchange.Status
            The status of the replica-exchange calculation. It has three
            fields: ``iteration``, ``target_error``, and ``is_completed``.

        """
        # Handle case in which storage is a string.
        reporter = cls._reporter_from_storage(storage, check_exist=True)

        # Read iteration and online analysis info.
        reporter.open(mode='r')
        options = reporter.read_dict('options')
        iteration = reporter.read_last_iteration(full_iteration=False)
        # Search for last cached free energies only if online analysis is activated.
        if options['online_analysis_interval'] is not None:
            target_error = options['online_analysis_target_error']
            last_err_free_energy = cls._read_last_free_energy(reporter, iteration)[1][1]
        else:
            target_error = None
            last_err_free_energy = None
        reporter.close()

        # Check if the calculation is done.
        number_of_iterations = options['number_of_iterations']
        online_analysis_target_error = options['online_analysis_target_error']
        is_completed = cls._is_completed_static(number_of_iterations, iteration,
                                                last_err_free_energy,
                                                online_analysis_target_error)

        return cls.Status(iteration=iteration, target_error=target_error,
                          is_completed=is_completed)

    # -------------------------------------------------------------------------
    # Public properties.
    # -------------------------------------------------------------------------

    @property
    def n_states(self):
        """The integer number of thermodynamic states (read-only)."""
        if self._thermodynamic_states is None:
            return 0
        else:
            return len(self._thermodynamic_states)

    @property
    def n_replicas(self):
        """The integer number of replicas (read-only)."""
        if self._sampler_states is None:
            return 0
        else:
            return len(self._sampler_states)

    @property
    def iteration(self):
        """The integer current iteration of the simulation (read-only).

        If the simulation has not been created yet, this is None.

        """
        return self._iteration

    @property
    def mcmc_moves(self):
        """A copy of the MCMCMoves list used to propagate the simulation.

        This can be set only before creation.

        """
        return copy.deepcopy(self._mcmc_moves)

    @mcmc_moves.setter
    def mcmc_moves(self, new_value):
        if self._thermodynamic_states is not None:
            # We can't modify representation of the MCMCMoves because it's
            # impossible to delete groups/variables from an NetCDF file. We
            # could support this by JSONizing the dict serialization and
            # store it as a string instead, if we needed this.
            raise RuntimeError('Cannot modify MCMCMoves after creation.')
        # If this is a single MCMCMove, it'll be transformed to a list in create().
        self._mcmc_moves = copy.deepcopy(new_value)

    @property
    def sampler_states(self):
        """A copy of the sampler states list at the current iteration.

        This can be set only before running.
        """
        return copy.deepcopy(self._sampler_states)

    @sampler_states.setter
    def sampler_states(self, value):
        if self._iteration != 0:
            raise RuntimeError('Sampler states can be assigned only between '
                               'create() and run().')
        if len(value) != self.n_replicas:
            raise ValueError('Passed {} sampler states for {} replicas'.format(
                len(value), self.n_replicas))

        # Update sampler state in the object and on storage.
        self._sampler_states = copy.deepcopy(value)
        mpi.run_single_node(0, self._reporter.write_sampler_states,
                            self._sampler_states, self._iteration)

    @property
    def is_periodic(self):
        """Return True if system is periodic, False if not, and None if not initialized"""
        if self._sampler_states is None:
            return None
        return self._thermodynamic_states[0].is_periodic

    class _StoredProperty(object):
        """
        Descriptor of a property stored as an option.

        validate_function is a simple function for checking things like "X > 0", but exposes both the
            ReplicaExchange instance and the new value for the variable, in that order.
        More complex checks which relies on the ReplicaExchange instance, like "if Y == True, then check X" can be
            accessed through the instance object of the function

        """

        def __init__(self, option_name, validate_function=None):
            self._option_name = option_name
            self._validate_function = validate_function

        def __get__(self, instance, owner_class=None):
            return getattr(instance, '_' + self._option_name)

        def __set__(self, instance, new_value):
            if self._validate_function is not None:
                new_value = self._validate_function(instance, new_value)
            setattr(instance, '_' + self._option_name, new_value)
            # Update storage if we ReplicaExchange is initialized.
            if instance._thermodynamic_states is not None:
                mpi.run_single_node(0, instance._store_options)

        # ----------------------------------
        # Value Validation of the properties
        # Should be @staticmethod with arguments of (instance, value) in that order, even if instance is not used
        # ----------------------------------

        @staticmethod
        def _number_of_iterations_validator(instance, number_of_iterations):
            # Support infinite number of iterations.
            if not (0 <= number_of_iterations <= float('inf')):
                raise ValueError('Accepted values for number_of_iterations are'
                                 'non-negative integers and infinity.')
            return number_of_iterations

        @staticmethod
        def _oa_interval_validator(instance, online_analysis_interval):
            """Check the online_analysis_interval value for consistency"""
            if online_analysis_interval is not None and (
                            type(online_analysis_interval) != int or online_analysis_interval < 1):
                raise ValueError("online_analysis_interval must be an integer 1 or greater, or None")
            return online_analysis_interval

        @staticmethod
        def _oa_target_error_validator(instance, online_analysis_target_error):
            if instance.online_analysis_interval is not None:
                if online_analysis_target_error < 0:
                    raise ValueError("online_analysis_target_error must be a float >= 0")
                elif online_analysis_target_error == 0:
                    logger.warning("online_analysis_target_error of 0 may never converge.")
            return online_analysis_target_error

        @staticmethod
        def _oa_min_iter_validator(instance, online_analysis_minimum_iterations):
            if (instance.online_analysis_interval is not None and
                    (type(
                        online_analysis_minimum_iterations) is not int or online_analysis_minimum_iterations < 0)):
                raise ValueError("online_analysis_minimum_iterations must be an integer >= 0")
            return online_analysis_minimum_iterations

        @staticmethod
        def _locality_validator(instance, locality):
            if locality is not None:
                if (type(locality) != int) or (locality <= 0):
                    raise ValueError("locality must be an int > 0")
            return locality


    number_of_iterations = _StoredProperty('number_of_iterations',
                                           validate_function=_StoredProperty._number_of_iterations_validator)
    online_analysis_interval = _StoredProperty('online_analysis_interval',
                                               validate_function=_StoredProperty._oa_interval_validator)  #:interval to carry out online analysis
    online_analysis_target_error = _StoredProperty('online_analysis_target_error',
                                                   validate_function=_StoredProperty._oa_target_error_validator)
    online_analysis_minimum_iterations = _StoredProperty('online_analysis_minimum_iterations',
                                                         validate_function=_StoredProperty._oa_min_iter_validator)
    locality = _StoredProperty('locality', validate_function=_StoredProperty._locality_validator)

    @property
    def metadata(self):
        """A copy of the metadata dictionary passed on creation (read-only)."""
        return copy.deepcopy(self._metadata)

    @property
    def is_completed(self):
        """Check if we have reached any of the stop target criteria (read-only)"""
        return self._is_completed()

    # -------------------------------------------------------------------------
    # Main public interface.
    # -------------------------------------------------------------------------

    _TITLE_TEMPLATE = ('Multi-state sampler simulation created using MultiStateSampler class '
                       'of yank.multistate on {}')

    def create(self, thermodynamic_states, sampler_states, storage,
               initial_thermodynamic_states=None, unsampled_thermodynamic_states=None,
               metadata=None):
        """Create new multistate sampler simulation.

        Parameters
        ----------
        thermodynamic_states : list of openmmtools.states.ThermodynamicState
            Thermodynamic states to simulate, where one replica is allocated per state.
            Each state must have a system with the same number of atoms.
        sampler_states : openmmtools.states.SamplerState or list
            One or more sets of initial sampler states.
            The number of replicas is taken to be the number of sampler states provided.
            If the sampler states do not have box_vectors attached and the system is periodic,
            an exception will be thrown.
        storage : str or instanced Reporter
            If str: the path to the storage file. Default checkpoint options from Reporter class are used
            If Reporter: Uses the reporter options and storage path
            In the future this will be able to take a Storage class as well.
        initial_thermodynamic_states : None or list or array-like of int of length len(sampler_states), optional,
            default: None.
            Initial thermodynamic_state index for each sampler_state.
            If no initial distribution is chosen, ``sampler_states`` are distributed between the
            ``thermodynamic_states`` following these rules:

                * If ``len(thermodynamic_states) == len(sampler_states)``: 1-to-1 distribution

                * If ``len(thermodynamic_states) > len(sampler_states)``: First and last state distributed first
                  remaining ``sampler_states`` spaced evenly by index until ``sampler_states`` are depleted.
                  If there is only one ``sampler_state``, then the only first ``thermodynamic_state`` will be chosen

                * If ``len(thermodynamic_states) < len(sampler_states)``, each ``thermodynamic_state`` receives an
                  equal number of ``sampler_states`` until there are insufficient number of ``sampler_states`` remaining
                  to give each ``thermodynamic_state`` an equal number. Then the rules from the previous point are
                  followed.

        unsampled_thermodynamic_states : list of openmmtools.states.ThermodynamicState, optional, default=None
            These are ThermodynamicStates that are not propagated, but their
            reduced potential is computed at each iteration for each replica.
            These energy can be used as data for reweighting schemes (default
            is None).
        metadata : dict, optional, default=None
           Simulation metadata to be stored in the file.
        """

        # Make sure sampler_states is an iterable of SamplerStates for later.
        if isinstance(sampler_states, mmtools.states.SamplerState):
            sampler_states = [sampler_states]

        # Check all systems are either periodic or not.
        is_periodic = thermodynamic_states[0].is_periodic
        for thermodynamic_state in thermodynamic_states:
            if thermodynamic_state.is_periodic != is_periodic:
                raise Exception('Thermodynamic states contain a mixture of '
                                'systems with and without periodic boundary conditions.')

        # Check that sampler states specify box vectors if the system is periodic
        if is_periodic:
            for sampler_state in sampler_states:
                if sampler_state.box_vectors is None:
                    raise Exception('All sampler states must have box_vectors defined if the system is periodic.')

        # Make sure all states have same number of particles. We don't
        # currently support writing storage with different n_particles
        n_particles = thermodynamic_states[0].n_particles
        for states in [thermodynamic_states, sampler_states]:
            for state in states:
                if state.n_particles != n_particles:
                    raise ValueError('All ThermodynamicStates and SamplerStates must '
                                     'have the same number of particles')

        # Handle case in which storage is a string.
        reporter = self._reporter_from_storage(storage, check_exist=False)

        # Check if netcdf files exist. This is run only on MPI node 0 and
        # broadcasted. This is to avoid the case where the other nodes
        # arrive to this line after node 0 has already created the storage
        # file, causing an error.
        if mpi.run_single_node(0, reporter.storage_exists, broadcast_result=True):
            raise RuntimeError('Storage file {} already exists; cowardly '
                               'refusing to overwrite.'.format(reporter.filepath))

        # Handle default argument for metadata and add default simulation title.
        default_title = (self._TITLE_TEMPLATE.format(time.asctime(time.localtime())))
        if metadata is None:
            metadata = dict(title=default_title)
        elif 'title' not in metadata:
            metadata['title'] = default_title
        self._metadata = metadata

        # Save thermodynamic states. This sets n_replicas.
        self._thermodynamic_states = copy.deepcopy(thermodynamic_states)

        # Handle default unsampled thermodynamic states.
        if unsampled_thermodynamic_states is None:
            self._unsampled_states = []
        else:
            self._unsampled_states = copy.deepcopy(unsampled_thermodynamic_states)

        # Deep copy sampler states.
        self._sampler_states = [copy.deepcopy(sampler_state) for sampler_state in sampler_states]

        # Make sure all sampler states have box vectors defined; add dummies if needed.
        default_box_vectors = thermodynamic_states[0].system.getDefaultPeriodicBoxVectors()
        for sampler_state in self._sampler_states:
            if sampler_state.box_vectors is None:
                sampler_state.box_vectors = default_box_vectors

        # Set initial thermodynamic state indices if not specified
        if initial_thermodynamic_states is None:
            initial_thermodynamic_states = self._default_initial_thermodynamic_states(thermodynamic_states,
                                                                                      sampler_states)
        self._replica_thermodynamic_states = np.array(initial_thermodynamic_states, np.int64)

        # Assign default system box vectors if None has been specified.
        for replica_id, thermodynamic_state_id in enumerate(self._replica_thermodynamic_states):
            sampler_state = self._sampler_states[replica_id]
            if sampler_state.box_vectors is not None:
                continue
            thermodynamic_state = self._thermodynamic_states[thermodynamic_state_id]
            sampler_state.box_vectors = thermodynamic_state.system.getDefaultPeriodicBoxVectors()

        # Ensure there is an MCMCMove for each thermodynamic state.
        if isinstance(self._mcmc_moves, mmtools.mcmc.MCMCMove):
            self._mcmc_moves = [copy.deepcopy(self._mcmc_moves) for _ in range(self.n_states)]
        elif len(self._mcmc_moves) != self.n_states:
            raise RuntimeError('The number of MCMCMoves ({}) and ThermodynamicStates ({}) must '
                               'be the same.'.format(len(self._mcmc_moves), self.n_states))

        # Reset iteration counter.
        self._iteration = 0

        # Reset statistics.
        # _n_accepted_matrix[i][j] is the number of swaps proposed between thermodynamic states i and j.
        # _n_proposed_matrix[i][j] is the number of swaps proposed between thermodynamic states i and j.
        self._n_accepted_matrix = np.zeros([self.n_states, self.n_states], np.int64)
        self._n_proposed_matrix = np.zeros([self.n_states, self.n_states], np.int64)

        # Allocate memory for energy matrix. energy_thermodynamic/unsampled_states[k][l]
        # is the reduced potential computed at the positions of SamplerState sampler_states[k]
        # and ThermodynamicState thermodynamic/unsampled_states[l].
        self._energy_thermodynamic_states = np.zeros([self.n_replicas, self.n_states], np.float64)
        self._neighborhoods = np.zeros([self.n_replicas, self.n_states], 'i1')
        self._energy_unsampled_states = np.zeros([self.n_replicas, len(self._unsampled_states)], np.float64)

        # Display papers to be cited.
        self._display_citations()

        # Close the reporter file so its ready for use
        reporter.close()
        self._reporter = reporter
        self._initialize_reporter()

    @mmtools.utils.with_timer('Minimizing all replicas')
    def minimize(self, tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometers,
                 max_iterations=0):
        """Minimize all replicas.

        Minimized positions are stored at the end.

        Parameters
        ----------
        tolerance : simtk.unit.Quantity, optional
            Minimization tolerance (units of energy/mole/length, default is
            ``1.0 * unit.kilojoules_per_mole / unit.nanometers``).
        max_iterations : int, optional
            Maximum number of iterations for minimization. If 0, minimization
            continues until converged.

        """
        # Check that simulation has been created.
        if self.n_replicas == 0:
            raise RuntimeError('Cannot minimize replicas. The simulation must be created first.')

        logger.debug("Minimizing all replicas...")

        # Distribute minimization across nodes. Only node 0 will get all positions.
        # The other nodes, only need the positions that they use for propagation and
        # computation of the energy matrix entries.
        minimized_positions, sampler_state_ids = mpi.distribute(self._minimize_replica, range(self.n_replicas),
                                                                tolerance, max_iterations,
                                                                send_results_to=0)

        # Update all sampler states. For non-0 nodes, this will update only the
        # sampler states associated to the replicas propagated by this node.
        for sampler_state_id, minimized_pos in zip(sampler_state_ids, minimized_positions):
            self._sampler_states[sampler_state_id].positions = minimized_pos

        # Save the stored positions in the storage
        mpi.run_single_node(0, self._reporter.write_sampler_states, self._sampler_states, self._iteration)

    def equilibrate(self, n_iterations, mcmc_moves=None):
        """Equilibrate all replicas.

        This does not increase the iteration counter. The equilibrated
        positions are stored at the end.

        Parameters
        ----------
        n_iterations : int
            Number of equilibration iterations.
        mcmc_moves : MCMCMove or list of MCMCMove, optional
            Optionally, the MCMCMoves to use for equilibration can be
            different from the ones used in production.

        """
        # Check that simulation has been created.
        if self.n_replicas == 0:
            raise RuntimeError('Cannot minimize replicas. The simulation must be created first.')

        # If no MCMCMove is specified, use the ones for production.
        if mcmc_moves is None:
            mcmc_moves = self._mcmc_moves

        # Make sure there is one MCMCMove per thermodynamic state.
        if isinstance(mcmc_moves, mmtools.mcmc.MCMCMove):
            mcmc_moves = [copy.deepcopy(mcmc_moves) for _ in range(self.n_states)]
        elif len(mcmc_moves) != self.n_states:
            raise RuntimeError('The number of MCMCMoves ({}) and ThermodynamicStates ({}) for equilibration'
                               ' must be the same.'.format(len(self._mcmc_moves), self.n_states))

        # Temporarily set the equilibration MCMCMoves.
        production_mcmc_moves = self._mcmc_moves
        self._mcmc_moves = mcmc_moves
        for iteration in range(n_iterations):
            logger.debug("Equilibration iteration {}/{}".format(iteration, n_iterations))
            self._propagate_replicas()

        # Restore production MCMCMoves.
        self._mcmc_moves = production_mcmc_moves

        # Update stored positions.
        mpi.run_single_node(0, self._reporter.write_sampler_states, self._sampler_states, self._iteration)

    def run(self, n_iterations=None):
        """Run the replica-exchange simulation.

        This runs at most ``number_of_iterations`` iterations. Use :func:`extend`
        to pass the limit.

        Parameters
        ----------
        n_iterations : int, optional
           If specified, only at most the specified number of iterations
           will be run (default is None).
        """
        # If this is the first iteration, compute and store the
        # starting energies of the minimized/equilibrated structures.
        if self._iteration == 0:
            try:
                self._compute_energies()
            # We're intercepting a possible initial NaN position here thrown by OpenMM, which is a simple exception
            # So we have to under-specify this trap.
            except Exception as e:
                if 'coordinate is nan' in str(e).lower():
                    err_message = "Initial coordinates were NaN! Check your inputs!"
                    logger.critical(err_message)
                    raise SimulationNaNError(err_message)
                else:
                    # If not the special case, raise the error normally
                    raise e
            mpi.run_single_node(0, self._reporter.write_energies, self._energy_thermodynamic_states,
                                self._neighborhoods, self._energy_unsampled_states, self._iteration)
            self._check_nan_energy()

        timer = mmtools.utils.Timer()
        timer.start('Run ReplicaExchange')
        run_initial_iteration = self._iteration

        # Handle default argument and determine number of iterations to run.
        if n_iterations is None:
            iteration_limit = self.number_of_iterations - self._iteration
        else:
            iteration_limit = min(self._iteration + n_iterations, self.number_of_iterations)

        # Main loop.
        while not self._is_completed(iteration_limit):
            # Increment iteration counter.
            self._iteration += 1

            logger.debug('Iteration {}/{}'.format(self._iteration, iteration_limit))
            timer.start('Iteration')

            # Update thermodynamic states
            self._mix_replicas()

            # Propagate replicas.
            self._propagate_replicas()

            # Compute energies of all replicas at all states
            self._compute_energies()

            # Write iteration to storage file
            self._report_iteration()

            # Update analysis
            self._update_analysis()

            # Computing timing information
            iteration_time = timer.stop('Iteration')
            partial_total_time = timer.partial('Run ReplicaExchange')
            time_per_iteration = partial_total_time / (self._iteration - run_initial_iteration)
            estimated_time_remaining = time_per_iteration * (iteration_limit - self._iteration)
            estimated_total_time = time_per_iteration * iteration_limit
            estimated_finish_time = time.time() + estimated_time_remaining
            # TODO: Transmit timing information

            # Show timing statistics if debug level is activated.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Iteration took {:.3f}s.".format(iteration_time))
                if estimated_time_remaining != float('inf'):
                    logger.debug("Estimated completion in {}, at {} (consuming total wall clock time {}).".format(
                        str(datetime.timedelta(seconds=estimated_time_remaining)),
                        time.ctime(estimated_finish_time),
                        str(datetime.timedelta(seconds=estimated_total_time))))

            # Perform sanity checks to see if we should terminate here.
            self._check_nan_energy()

    def extend(self, n_iterations):
        """Extend the simulation by the given number of iterations.

        Contrarily to :func:`run`, this will extend the number of iterations past
        ``number_of_iteration`` if requested.

        Parameters
        ----------
        n_iterations : int
           The number of iterations to run.

        """
        if self._iteration + n_iterations > self.number_of_iterations:
            # This MUST be assigned to a property or the storage won't be updated.
            self.number_of_iterations = self._iteration + n_iterations
        self.run(n_iterations)

    def __repr__(self):
        """Return a 'formal' representation that can be used to reconstruct the class, if possible."""
        # TODO: Can we make this a more useful expression?
        return "<instance of MultiStateSampler>"

    def __del__(self):
        # The reporter could be None if MultiStateSampler was not created.
        if self._reporter is not None:
            mpi.run_single_node(0, self._reporter.close)

    # -------------------------------------------------------------------------
    # Internal-usage.
    # -------------------------------------------------------------------------

    def _check_nan_energy(self):
        """Checks that energies are finite and abort otherwise.

        Checks both sampled and unsampled thermodynamic states.

        """
        # Find faulty replicas to create error message.
        faulty_replicas = set()

        # Check sampled thermodynamic states first.
        state_type = 'thermodynamic state'

        for (replica_id, state_id) in enumerate(self._replica_thermodynamic_states):
            neighborhood = self._neighborhood(state_id)
            if np.any(np.isnan(self._energy_thermodynamic_states[replica_id,neighborhood])):
                faulty_replicas.add(replica_id)

        # If there are no NaNs in energies, the problem is in the unsampled states.
        if (len(faulty_replicas) == 0) and (self._energy_unsampled_states.shape[1] > 0):
            state_type = 'unsampled thermodynamic state'
            for replica_id in range(self.n_replicas):
                if np.any(np.isnan(self._energy_unsampled_states[replica_id])):
                    faulty_replicas.add(replica_id)

        if len(faulty_replicas) > 0:
            # Raise exception.
            err_msg = "NaN encountered in {} energies for replicas {}".format(state_type, faulty_replicas)
            logger.critical(err_msg)
            raise SimulationNaNError(err_msg)

    @mpi.on_single_node(rank=0, broadcast_result=False, sync_nodes=False)
    def _display_citations(self, overwrite_global=False, citation_stack=None):
        """
        Display papers to be cited.
        The overwrite_global command will force the citation to display even if the "have_citations_been_shown" variable
            is True
        """
        # TODO Add original citations for various replica-exchange schemes.
        # TODO Show subset of OpenMM citations based on what features are being used.
        openmm_citations = """\
        Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing unit. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
        Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. Comput. Sci. Eng. 12:34, 2010. DOI: 10.1109/MCSE.2010.27
        Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. J. Comput. Chem. 31:1268, 2010. DOI: 10.1002/jcc.21413
        Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. J. Chem. Theor. Comput. 6:434, 2010. DOI: 10.1021/ct900463w"""

        mbar_citations = """\
        Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states. J. Chem. Phys. 129:124105, 2008. DOI: 10.1063/1.2978177"""

        if citation_stack is not None:
            citation_stack = [openmm_citations] + citation_stack
        else:
            citation_stack = [openmm_citations]

        if overwrite_global or (not self._have_displayed_citations_before and not self._global_citation_silence):
            print("Please cite the following:")
            print("")
            for citation in citation_stack:
                print(citation)
            self._have_displayed_citations_before = True

    # -------------------------------------------------------------------------
    # Internal-usage: Initialization and storage utilities.
    # -------------------------------------------------------------------------

    def _default_initial_thermodynamic_states(self, thermodynamic_states, sampler_states):
        """
        Create the initial_thermodynamic_states obeying the following rules:

        * If ``len(thermodynamic_states) == len(sampler_states)``: 1-to-1 distribution

        * If ``len(thermodynamic_states) > len(sampler_states)``: First and last state distributed first
          remaining ``sampler_states`` spaced evenly by index until ``sampler_states`` are depleted.
          If there is only one ``sampler_state``, then the only first ``thermodynamic_state`` will be chosen

        * If ``len(thermodynamic_states) < len(sampler_states)``, each ``thermodynamic_state`` receives an
          equal number of ``sampler_states`` until there are insufficient number of ``sampler_states`` remaining
          to give each ``thermodynamic_state`` an equal number. Then the rules from the previous point are
          followed.
        """
        # Ignore IDE's saying this may be static because subclasses implement changes and need to call super()
        # which does not work for staticmethods
        n_thermo = len(thermodynamic_states)
        n_sampler = len(sampler_states)
        thermo_indices = np.arange(n_thermo, dtype=int)
        initial_thermo_states = np.zeros(n_sampler, dtype=int)
        # Determine how many loops we can do
        loops = n_sampler // n_thermo  # Floor division (//)
        n_looped = n_thermo * loops
        initial_thermo_states[:n_looped] = np.tile(thermo_indices, loops)
        # Distribute remaining values, -1 from n_thermo to handle indices correctly
        initial_thermo_states[n_looped:] = np.linspace(0, n_thermo - 1, n_sampler - n_looped, dtype=int)
        return initial_thermo_states

    @staticmethod
    def _does_file_exist(file_path):
        """Check if there is a file at the given path."""
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    @staticmethod
    def _reporter_from_storage(storage, check_exist=True):
        """Return the Reporter object associated to this storage.

        If check_exist is True, FileNotFoundError is raised if the files
        are not found.
        """
        if isinstance(storage, str):
            # Open a reporter to read the data.
            reporter = MultiStateReporter(storage)
        else:
            reporter = storage

        # Check if netcdf file exists.
        if check_exist and not reporter.storage_exists():
            reporter.close()
            raise FileNotFoundError('Storage file {} or its subfiles do not exist; '
                                    'cannot read status.'.format(reporter.filepath))
        return reporter

    @mpi.on_single_node(rank=0, broadcast_result=False, sync_nodes=True)
    def _initialize_reporter(self):
        """Initialize the reporter and store initial information.

        This is executed only on MPI node 0 and it is blocking. This is to
        avoid the case where the other nodes skip ahead and try to read
        from a file that hasn't been created yet.

        """
        self._reporter.open(mode='w')
        self._reporter.write_thermodynamic_states(self._thermodynamic_states,
                                                  self._unsampled_states)

        # Store run metadata and ReplicaExchange options.
        self._store_options()
        self._reporter.write_dict('metadata', self._metadata)

        # Store initial conditions. This forces the storage to be synchronized.
        self._report_iteration()

    @mpi.on_single_node(rank=0, broadcast_result=False, sync_nodes=False)
    @mpi.delayed_termination
    @mmtools.utils.with_timer('Writing iteration information to storage')
    def _report_iteration(self):
        """Store positions, states, and energies of current iteration.

        This is executed only on MPI node 0 and it's not blocking. The
        termination is delayed so that the file is not written only with
        partial data if the program gets interrupted.

        """
        self._reporter.write_sampler_states(self._sampler_states, self._iteration)
        self._reporter.write_replica_thermodynamic_states(self._replica_thermodynamic_states, self._iteration)
        self._reporter.write_mcmc_moves(self._mcmc_moves)  # MCMCMoves can store internal statistics.
        self._reporter.write_energies(self._energy_thermodynamic_states, self._neighborhoods, self._energy_unsampled_states,
                                      self._iteration)
        self._reporter.write_mixing_statistics(self._n_accepted_matrix, self._n_proposed_matrix, self._iteration)
        self._reporter.write_timestamp(self._iteration)
        self._reporter.write_last_iteration(self._iteration)
        self._reporter.sync()

    @classmethod
    def default_options(cls):
        """
        dict of all default class options (keyword arguments for __init__ for class and superclasses)
        """
        options_to_report = dict()
        for c in inspect.getmro(cls):
            parameter_names, _, _, defaults = inspect.getargspec(c.__init__)
            if defaults:
                class_options = {parameter_name: defaults[index] for (index, parameter_name) in
                                 enumerate(parameter_names[-len(defaults):])}
                options_to_report.update(class_options)
        options_to_report.pop('mcmc_moves')
        return options_to_report

    @property
    def options(self):
        """
        dict of all class options (keyword arguments for __init__ for class and superclasses)
        """
        options_to_report = dict()
        for cls in inspect.getmro(type(self)):
            parameter_names, _, _, defaults = inspect.getargspec(cls.__init__)
            if defaults:
                class_options = {parameter_name: getattr(self, '_' + parameter_name) for
                                 parameter_name in parameter_names[-len(defaults):]}
                options_to_report.update(class_options)
        options_to_report.pop('mcmc_moves')
        return options_to_report

    def _store_options(self):
        """Store __init__ parameters (beside MCMCMoves) in storage file."""
        logger.debug("Storing general ReplicaExchange options...")
        self._reporter.write_dict('options', self.options)

    # -------------------------------------------------------------------------
    # Locality
    # -------------------------------------------------------------------------

    def _neighborhood(self, state_index):
        """Compute the states in the local neighborhood determined by self.locality

        Parameters
        ----------
        current_state_index : int
            The currrent state

        Returns
        -------
        neighborhood : list of int
            The states in the local neighborhood
        """
        if self.locality == None:
            # Global neighborhood
            return list(range(0, self.n_states))
        else:
            # Local neighborhood specified by 'locality'
            return list(range(max(0, state_index - self.locality), min(self.n_states, state_index + self.locality + 1)))

    # -------------------------------------------------------------------------
    # Internal-usage: Distributed tasks.
    # -------------------------------------------------------------------------

    @mmtools.utils.with_timer('Propagating all replicas')
    def _propagate_replicas(self):
        """Propagate all replicas."""
        # TODO  Report on efficiency of dyanmics (fraction of time wasted to overhead).
        logger.debug("Propagating all replicas...")

        # Distribute propagation across nodes. Only node 0 will get all positions
        # and box vectors. The other nodes, only need the positions that they use
        # for propagation and computation of the energy matrix entries.
        propagated_states, replica_ids = mpi.distribute(self._propagate_replica, range(self.n_replicas),
                                                        send_results_to=0)

        # Update all sampler states. For non-0 nodes, this will update only the
        # sampler states associated to the replicas propagated by this node.
        for replica_id, propagated_state in zip(replica_ids, propagated_states):
            propagated_positions, propagated_box_vectors = propagated_state  # Unpack.
            self._sampler_states[replica_id].positions = propagated_positions
            self._sampler_states[replica_id].box_vectors = propagated_box_vectors

        # Gather all MCMCMoves statistics. All nodes must have these up-to-date
        # since they are tied to the ThermodynamicState, not the replica.
        all_statistics = mpi.distribute(self._get_replica_move_statistics, range(self.n_replicas),
                                        send_results_to='all')
        for replica_id in range(self.n_replicas):
            if len(all_statistics[replica_id]) > 0:
                thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
                self._mcmc_moves[thermodynamic_state_id].statistics = all_statistics[replica_id]

    def _propagate_replica(self, replica_id):
        """Propagate thermodynamic state associated to the given replica."""
        # Retrieve thermodynamic, sampler states, and MCMC move of this replica.
        thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
        thermodynamic_state = self._thermodynamic_states[thermodynamic_state_id]
        mcmc_move = self._mcmc_moves[thermodynamic_state_id]
        sampler_state = self._sampler_states[replica_id]

        # Apply MCMC move.
        try:
            mcmc_move.apply(thermodynamic_state, sampler_state)
        except mmtools.mcmc.IntegratorMoveError as e:
            # Save NaNnig context and MCMove before aborting.
            output_dir = os.path.join(os.path.dirname(self._reporter.filepath), 'nan-error-logs')
            file_name = 'iteration{}-replica{}-state{}'.format(self._iteration, replica_id,
                                                               thermodynamic_state_id)
            e.serialize_error(os.path.join(output_dir, file_name))
            message = ('This Multistate Sampler simulation threw a NaN!\nLook for error logs in:\n'
                       '\tDirectory: {}\tFile Name Base: {}').format(output_dir, file_name)
            logger.critical(message)
            raise SimulationNaNError(message)

        # Return new positions and box vectors.
        return sampler_state.positions, sampler_state.box_vectors

    def _get_replica_move_statistics(self, replica_id):
        """Return the statistics of the MCMCMove currently associated to this replica."""
        thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
        mcmc_move = self._mcmc_moves[thermodynamic_state_id]

        try:
            move_statistics = mcmc_move.statistics
        except AttributeError:
            move_statistics = {}

        return move_statistics

    def _minimize_replica(self, replica_id, tolerance, max_iterations):
        """Minimize the specified replica."""
        # Retrieve thermodynamic and sampler states.
        thermodynamic_state_id = self._replica_thermodynamic_states[replica_id]
        thermodynamic_state = self._thermodynamic_states[thermodynamic_state_id]
        sampler_state = self._sampler_states[replica_id]

        # Retrieve a context. Any Integrator works.
        context, integrator = mmtools.cache.global_context_cache.get_context(thermodynamic_state)

        # Set initial positions and box vectors.
        sampler_state.apply_to_context(context)

        # Compute the initial energy of the system for logging.
        initial_energy = thermodynamic_state.reduced_potential(context)
        logger.debug('Replica {}/{}: initial energy {:8.3f}kT'.format(
            replica_id + 1, self.n_replicas, initial_energy))

        # Minimize energy.
        openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)

        # Get the minimized positions.
        sampler_state.update_from_context(context)

        # Compute the final energy of the system for logging.
        final_energy = thermodynamic_state.reduced_potential(sampler_state)
        logger.debug('Replica {}/{}: final energy {:8.3f}kT'.format(
            replica_id + 1, self.n_replicas, final_energy))

        # Return minimized positions.
        return sampler_state.positions

    @mmtools.utils.with_timer('Computing energy matrix')
    def _compute_energies(self):
        """Compute energies of all replicas at all states."""

        # Determine neighborhoods (all nodes)
        self._neighborhoods[:,:] = False
        for (replica_index, state_index) in enumerate(self._replica_thermodynamic_states):
            neighborhood = self._neighborhood(state_index)
            self._neighborhoods[replica_index,neighborhood] = True

        # Distribute energy computation across nodes. Only node 0 receives
        # all the energies since it needs to store them and mix states.
        new_energies, replica_ids = mpi.distribute(self._compute_replica_energies, range(self.n_replicas),
                                                   send_results_to=0)

        # Update energy matrices. Non-0 nodes update only the energies computed by this replica.
        for replica_id, energies in zip(replica_ids, new_energies):
            energy_thermodynamic_states, energy_unsampled_states = energies  # Unpack.
            self._energy_thermodynamic_states[replica_id] = energy_thermodynamic_states
            self._energy_unsampled_states[replica_id] = energy_unsampled_states

    def _compute_replica_energies(self, replica_id):
        """Compute the energy for the replica in every ThermodynamicState."""
        # Initialize replica energies for each thermodynamic state.
        energy_thermodynamic_states = np.zeros(self.n_states)
        energy_unsampled_states = np.zeros(len(self._unsampled_states))

        # Retrieve sampler state associated to this replica.
        sampler_state = self._sampler_states[replica_id]

        # Determine neighborhood
        state_index = self._replica_thermodynamic_states[replica_id]
        neighborhood = self._neighborhood(state_index)
        neighborhood_slice = slice(neighborhood[0], neighborhood[-1])

        # Compute energy for all thermodynamic states.
        for energies, states in [(energy_thermodynamic_states[neighborhood_slice], self._thermodynamic_states[neighborhood_slice]),
                                 (energy_unsampled_states, self._unsampled_states)]:
            # Group thermodynamic states by compatibility.
            compatible_groups, original_indices = mmtools.states.group_by_compatibility(states)

            # Compute the reduced potentials of all the compatible states.
            for compatible_group, state_indices in zip(compatible_groups, original_indices):
                # Get the context, any Integrator works.
                context, integrator = mmtools.cache.global_context_cache.get_context(compatible_group[0])

                # Update positions and box vectors. We don't need
                # to set Context velocities for the potential.
                sampler_state.apply_to_context(context, ignore_velocities=True)

                # Compute and update the reduced potentials.
                compatible_energies = mmtools.states.ThermodynamicState.reduced_potential_at_states(
                    context, compatible_group)
                for energy_idx, state_idx in enumerate(state_indices):
                    energies[state_idx] = compatible_energies[energy_idx]

        # Return the new energies.
        return energy_thermodynamic_states, energy_unsampled_states

    # -------------------------------------------------------------------------
    # Internal-usage: Replicas mixing.
    # -------------------------------------------------------------------------

    @mpi.on_single_node(0, broadcast_result=True)
    def _mix_replicas(self):
        """Do nothing to replicas."""
        logger.debug("Mixing replicas (does nothing for MultiStateSampler)...")

        # Reset storage to keep track of swap attempts this iteration.
        self._n_accepted_matrix[:, :] = 0
        self._n_proposed_matrix[:, :] = 0

        # Determine fraction of swaps accepted this iteration.
        n_swaps_proposed = self._n_proposed_matrix.sum()
        n_swaps_accepted = self._n_accepted_matrix.sum()
        swap_fraction_accepted = 0.0
        if n_swaps_proposed > 0:
            swap_fraction_accepted = n_swaps_accepted / n_swaps_proposed  # Python 3 uses true division for /
        logger.debug("Accepted {}/{} attempted swaps ({:.1f}%)".format(n_swaps_accepted, n_swaps_proposed,
                                                                       swap_fraction_accepted * 100.0))

    # -------------------------------------------------------------------------
    # Internal-usage: Offline and online analysis
    # -------------------------------------------------------------------------

    @mpi.on_single_node(rank=0, broadcast_result=True)
    @mpi.delayed_termination
    @mmtools.utils.with_timer('Computing offline free energy estimate')
    def _offline_analysis(self):
        """Compute offline estimate of free energies

        This scheme only works with global localities.
        """
        # TODO: Currently, this just uses MBAR, which only works for global neighborhoods.
        # TODO: Add Local WHAM support.

        if (self._locality != None):
            raise Exception('Cannot use MBAR with non-global locality.')

        # This relative import is down here because having it at the top causes an ImportError.
        # __init__ pulls in multistate, which pulls in analyze, which pulls in MultiState. Because the first
        # MultiStateSampler never finished importing, its not in the name space which causes relative analyze import of
        # MultiStateSampler to crash as neither of them are the __main__ package.
        # https://stackoverflow.com/questions/6351805/cyclic-module-dependencies-and-relative-imports-in-python
        from .multistateanalyzer import MultiStateSamplerAnalyzer

        # Start the analysis
        bump_error_counter = False
        analysis = MultiStateSamplerAnalyzer(self._reporter, analysis_kwargs={'initial_f_k': self._last_mbar_f_k})

        # Indices for online analysis, "i'th index, j'th index"
        idx, jdx = 0, -1
        timer = mmtools.utils.Timer()
        timer.start("MBAR")
        logger.debug("Computing free energy with MBAR...")
        try:  # Trap errors for MBAR being under sampled and the W_nk matrix not being normalized correctly
            mbar = analysis.mbar
            free_energy, err_free_energy = analysis.get_free_energy()
        except ParameterError as e:
            # We don't update self._last_err_free_energy here since if it
            # wasn't below the target threshold before, it won't stop MultiStateSampler now.
            bump_error_counter = True
            self._online_error_bank.append(e)
        else:
            self._last_mbar_f_k = mbar.f_k
            free_energy = free_energy[idx, jdx]
            self._last_err_free_energy = err_free_energy[idx, jdx]
            logger.debug("Current Free Energy Estimate is {} +- {} kT".format(free_energy,
                                                                              self._last_err_free_energy))
            # Trap a case when errors don't converge (usually due to under sampling)
            if np.isnan(self._last_err_free_energy):
                self._last_err_free_energy = np.inf
        timer.stop("MBAR")

        # Raise an exception after 6 times MBAR gave an error.
        if bump_error_counter:
            self._online_error_trap_counter += 1
            if self._online_error_trap_counter >= 6:
                logger.debug("Thrown MBAR Errors:")
                for err in self._online_error_bank:
                    logger.debug(str(err))
                raise RuntimeError("Online Analysis has failed too many times! Please "
                                   "check the latest logs to see the thrown errors!")
            # Don't write out the free energy in case of error.
            return

        # Write out the numbers
        self._reporter.write_mbar_free_energies(self._iteration, self._last_mbar_f_k,
                                                (free_energy, self._last_err_free_energy))

        return self._last_err_free_energy

    @mpi.on_single_node(rank=0, broadcast_result=True)
    @mpi.delayed_termination
    @mmtools.utils.with_timer('Computing online free energy estimate')
    def _online_analysis(self, gamma0=1.0):
        """Perform online analysis of free energies

        This scheme works with all localities: global and local.

        """
        timer = mmtools.utils.Timer()
        timer.start("Online analysis")
        from scipy.special import logsumexp

        # TODO: This is experimental

        gamma = gamma0 / float(self._iteration+1)

        if self._last_mbar_f_k is None:
            self._last_mbar_f_k = np.zeros([self.n_states], np.float64)

        logZ = - self._last_mbar_f_k

        for (replica_index, state_index) in enumerate(self._replica_thermodynamic_states):
            neighborhood = self._neighborhood(state_index)
            u_k = self._energy_thermodynamic_states[replica_index,:]
            log_P_k = np.zeros([self.n_states], np.float64)
            log_pi_k = np.zeros([self.n_states], np.float64)
            log_weights = np.zeros([self.n_states], np.float64)
            log_P_k[neighborhood] = log_weights[neighborhood] - u_k[neighborhood]
            log_P_k[neighborhood] -= logsumexp(log_P_k[neighborhood])
            logZ[neighborhood] += gamma * np.exp(log_P_k[neighborhood] - log_pi_k[neighborhood])

        # Subtract off logZ[0] to prevent logZ from growing without bound
        logZ[:] -= logZ[0]

        self._last_mbar_f_k = -logZ
        free_energy = self._last_mbar_f_k[-1] - self._last_mbar_f_k[0]
        self._last_err_free_energy = np.Inf

        # Store free energy estimate
        self._reporter.write_mbar_free_energies(self._iteration, self._last_mbar_f_k,
                                                (free_energy, self._last_err_free_energy))

        timer.stop("Online analysis")

        return self._last_err_free_energy

    def _update_analysis(self):
        """Update online analysis of free energies"""

        # TODO: Currently, this just calls the offline analysis at certain intervals, if requested.
        # TODO: Refactor this to always compute fast online analysis, updating with offline analysis infrequently.

        # TODO: Simplify this
        if (self.online_analysis_interval is None):
            analysis_to_perform = None
        elif (self._iteration <= self.online_analysis_minimum_iterations):
            analysis_to_perform = 'online'
        elif (self._iteration % self.online_analysis_interval != 0):
            analysis_to_perform = 'online'
        elif (self._locality != None):
            analysis_to_perform = 'online'
        else:
            # All conditions are met for offline analysis
            analysis_to_perform = 'offline'

        # Execute selected analysis (only runs on node 0)
        if analysis_to_perform == 'online':
            self._last_err_free_energy = self._online_analysis()
        elif analysis_to_perform == 'offline':
            self._last_err_free_energy = self._offline_analysis()

        return

    @staticmethod
    def _read_last_free_energy(reporter, iteration):
        """Get the last free energy computed from online analysis"""
        last_f_k = None
        last_free_energy = None

        # Search for a valid free energy from the given iteration
        # to the start of the calculation.
        for index in range(iteration, 0, -1):
            try:
                last_f_k, last_free_energy = reporter.read_mbar_free_energies(index)
            except (IndexError, KeyError):
                # No such f_k written yet (or variable created).
                break
            # Find an f_k that is not all zeros (or masked and empty)
            if not (np.ma.is_masked(last_f_k) or np.all(last_f_k == 0)):
                break  # Don't need to continue the loop if we already found one

        return last_f_k, last_free_energy

    def _is_completed(self, iteration_limit=None):
        """Check if we have reached the required number of iterations or statistical error."""
        if iteration_limit is None:
            iteration_limit = self.number_of_iterations
        return self._is_completed_static(iteration_limit, self._iteration,
                                         self._last_err_free_energy,
                                         self.online_analysis_target_error)

    @staticmethod
    def _is_completed_static(iteration_limit, iteration, last_err_free_energy,
                             online_analysis_target_error):
        """Check if we have reached the required number of iterations or statistical error."""
        # Return if we have reached the number of iterations
        # or the statistical error target required.
        if (iteration >= iteration_limit or (
                        last_err_free_energy is not None and last_err_free_energy <= online_analysis_target_error)):
            return True
        return False

    # -------------------------------------------------------------------------
    # Internal-usage: Test globals
    # -------------------------------------------------------------------------
    _global_citation_silence = False


# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
