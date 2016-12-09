#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test State classes in states.py.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import nose
from openmmtools import testsystems

from yank.states import *


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_barostat_temperature(barostat):
    """Backward-compatibly get barostat's temperature"""
    try:  # TODO drop this when we stop openmm7.0 support
        return barostat.getDefaultTemperature()
    except AttributeError:  # versions previous to OpenMM 7.1
        return barostat.setTemperature()


# =============================================================================
# TEST THERMODYNAMIC STATE
# =============================================================================

class TestThermodynamicState(object):

    @classmethod
    def setup_class(cls):
        """Create the test systems used in the test suite."""
        cls.temperature = 300*unit.kelvin
        cls.pressure = 1.01325*unit.bar
        cls.toluene_vacuum = testsystems.TolueneVacuum().system
        cls.toluene_implicit = testsystems.TolueneImplicit().system

        alanine_explicit = testsystems.AlanineDipeptideExplicit()
        cls.alanine_explicit = alanine_explicit.system
        cls.alanine_positions = alanine_explicit.positions

        # A system correctly barostated
        cls.barostated_alanine = copy.deepcopy(cls.alanine_explicit)
        barostat = openmm.MonteCarloBarostat(cls.pressure, cls.temperature)
        cls.barostated_alanine.addForce(barostat)

        # A non-periodic system barostated
        cls.barostated_toluene = copy.deepcopy(cls.toluene_vacuum)
        barostat = openmm.MonteCarloBarostat(cls.pressure, cls.temperature)
        cls.barostated_toluene.addForce(barostat)

        # A system with two identical MonteCarloBarostats
        cls.multiple_barostat_alanine = copy.deepcopy(cls.barostated_alanine)
        barostat = openmm.MonteCarloBarostat(cls.pressure, cls.temperature)
        cls.multiple_barostat_alanine.addForce(barostat)

        # A system with an unsupported MonteCarloAnisotropicBarostat
        cls.unsupported_barostat_alanine = copy.deepcopy(cls.alanine_explicit)
        pressure_in_bars = cls.pressure / unit.bar
        anisotropic_pressure = openmm.Vec3(pressure_in_bars, pressure_in_bars,
                                           pressure_in_bars)
        barostat = openmm.MonteCarloAnisotropicBarostat(anisotropic_pressure,
                                                        cls.temperature)
        cls.unsupported_barostat_alanine.addForce(barostat)

        # A system with an inconsistent pressure in the barostat.
        cls.inconsistent_pressure_alanine = copy.deepcopy(cls.alanine_explicit)
        barostat = openmm.MonteCarloBarostat(cls.pressure + 0.2*unit.bar,
                                             cls.temperature)
        cls.inconsistent_pressure_alanine.addForce(barostat)

        # A system with an inconsistent temperature in the barostat.
        cls.inconsistent_temperature_alanine = copy.deepcopy(cls.alanine_explicit)
        barostat = openmm.MonteCarloBarostat(cls.pressure,
                                             cls.temperature + 1.0*unit.kelvin)
        cls.inconsistent_temperature_alanine.addForce(barostat)

    def test_method_find_barostat(self):
        """ThermodynamicState._find_barostat() method."""
        barostat = ThermodynamicState._find_barostat(self.barostated_alanine)
        assert isinstance(barostat, openmm.MonteCarloBarostat)

        # Raise exception if multiple or unsupported barostats found
        TE = ThermodynamicsError  # shortcut
        test_cases = [(self.multiple_barostat_alanine, TE.MULTIPLE_BAROSTATS),
                      (self.unsupported_barostat_alanine, TE.UNSUPPORTED_BAROSTAT)]
        for system, err_code in test_cases:
            with nose.tools.assert_raises(ThermodynamicsError) as cm:
                ThermodynamicState._find_barostat(system)
            assert cm.exception.code == err_code

    def test_method_is_barostat_consistent(self):
        """ThermodynamicState._is_barostat_consistent() method."""
        temperature = self.temperature
        pressure = self.pressure
        state = ThermodynamicState(self.barostated_alanine, temperature)

        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        assert state._is_barostat_consistent(barostat)
        barostat = openmm.MonteCarloBarostat(pressure + 0.2*unit.bar, temperature)
        assert not state._is_barostat_consistent(barostat)
        barostat = openmm.MonteCarloBarostat(pressure, temperature + 10*unit.kelvin)
        assert not state._is_barostat_consistent(barostat)

    def test_method_set_barostat_temperature(self):
        """ThermodynamicState._set_barostat_temperature() method."""
        state = ThermodynamicState(self.barostated_alanine, self.temperature)
        new_temperature = self.temperature + 10*unit.kelvin
        state._temperature = new_temperature

        barostat = state._barostat
        assert get_barostat_temperature(barostat) == self.temperature
        assert state._set_barostat_temperature(barostat)
        assert get_barostat_temperature(barostat) == new_temperature
        assert not state._set_barostat_temperature(barostat)

    def test_property_temperature(self):
        """ThermodynamicState.temperature property."""
        state = ThermodynamicState(self.barostated_alanine,
                                   self.temperature)
        assert state.temperature == self.temperature

        temperature = self.temperature + 10.0*unit.kelvin
        state.temperature = temperature
        assert state.temperature == temperature
        assert get_barostat_temperature(state._barostat) == temperature

    def test_method_set_system_pressure(self):
        """ThermodynamicState._set_system_pressure() method."""
        state = ThermodynamicState(self.alanine_explicit, self.temperature)
        assert not state._set_system_pressure(state._system, None)
        assert state._set_system_pressure(state._system, self.pressure)
        assert not state._set_system_pressure(state._system, self.pressure)

    def test_property_pressure(self):
        """ThermodynamicState.pressure property."""
        # Vacuum and implicit system are read with no pressure
        nonperiodic_testcases = [self.toluene_vacuum, self.toluene_implicit]
        for system in nonperiodic_testcases:
            state = ThermodynamicState(system, self.temperature)
            assert state.pressure is None

            # We can't set the pressure on non-periodic systems
            with nose.tools.assert_raises(ThermodynamicsError) as cm:
                state.pressure = 1.0*unit.bar
            assert cm.exception.code == ThermodynamicsError.BAROSTATED_NONPERIODIC

        # Correctly reads and set system pressures
        periodic_testcases = [self.alanine_explicit]
        for system in periodic_testcases:
            state = ThermodynamicState(system, self.temperature)
            assert state.pressure is None
            assert state._barostat is None

            # Setting pressure adds a barostat
            state.pressure = self.pressure
            assert state.pressure == self.pressure
            barostat = state._barostat
            assert barostat.getDefaultPressure() == self.pressure
            assert get_barostat_temperature(barostat) == self.temperature

            # Setting new pressure changes the barostat parameters
            new_pressure = self.pressure + 1.0*unit.bar
            state.pressure = new_pressure
            assert state.pressure == new_pressure
            barostat = state._barostat
            assert barostat.getDefaultPressure() == new_pressure
            assert get_barostat_temperature(barostat) == self.temperature

            # Setting pressure to None removes barostat
            state.pressure = None
            assert state._barostat is None

    def test_property_volume(self):
        """Check that volume is computed correctly."""
        # For volume-fluctuating systems volume is None.
        state = ThermodynamicState(self.barostated_alanine, self.temperature)
        assert state.volume is None

        # For periodic systems in NVT, volume is correctly computed.
        system = self.alanine_explicit
        box_vectors = system.getDefaultPeriodicBoxVectors()
        volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]
        state = ThermodynamicState(system, self.temperature)
        assert state.volume == volume

        # For non-periodic systems, volume is None.
        state = ThermodynamicState(self.toluene_vacuum, self.temperature)
        assert state.volume is None

    def test_property_system(self):
        """Cannot set a system with an incompatible barostat."""
        state = ThermodynamicState(self.barostated_alanine, self.temperature)
        assert state.pressure == self.pressure  # pre-condition

        TE = ThermodynamicsError  # shortcut
        test_cases = [(self.toluene_vacuum, TE.BAROSTATED_NONPERIODIC),
                      (self.barostated_toluene, TE.BAROSTATED_NONPERIODIC),
                      (self.multiple_barostat_alanine, TE.MULTIPLE_BAROSTATS),
                      (self.inconsistent_pressure_alanine, TE.INCONSISTENT_BAROSTAT),
                      (self.inconsistent_temperature_alanine, TE.INCONSISTENT_BAROSTAT)]
        for system, error_code in test_cases:
            with nose.tools.assert_raises(ThermodynamicsError) as cm:
                state.system = system
            assert cm.exception.code == error_code

        # It is possible to set an inconsistent system
        # if thermodynamic state is changed first.
        inconsistent_system = self.inconsistent_pressure_alanine
        state.pressure = self.pressure + 0.2*unit.bar
        state.system = self.inconsistent_pressure_alanine
        state_system_str = openmm.XmlSerializer.serialize(state.system)
        inconsistent_system_str = openmm.XmlSerializer.serialize(inconsistent_system)
        assert state_system_str == inconsistent_system_str

    def test_constructor_unsupported_barostat(self):
        """Exception is raised on construction with unsupported barostats."""
        TE = ThermodynamicsError  # shortcut
        test_cases = [(self.barostated_toluene, TE.BAROSTATED_NONPERIODIC),
                      (self.multiple_barostat_alanine, TE.MULTIPLE_BAROSTATS),
                      (self.unsupported_barostat_alanine, TE.UNSUPPORTED_BAROSTAT)]
        for i, (system, err_code) in enumerate(test_cases):
            with nose.tools.assert_raises(TE) as cm:
                ThermodynamicState(system=system, temperature=self.temperature)
            assert cm.exception.code == err_code

    def test_constructor_barostat(self):
        """The system barostat is properly configured on construction."""
        system = self.alanine_explicit
        old_serialization = openmm.XmlSerializer.serialize(system)
        assert ThermodynamicState._find_barostat(system) is None  # test-precondition

        # If we don't specify pressure, no barostat is added
        state = ThermodynamicState(system=system, temperature=self.temperature)
        assert state._barostat is None

        # If we specify pressure, barostat is added
        state = ThermodynamicState(system=system, temperature=self.temperature,
                                   pressure=self.pressure)
        assert state._barostat is not None

        # If we feed a barostat with an inconsistent temperature, it's fixed.
        state = ThermodynamicState(self.inconsistent_temperature_alanine,
                                   temperature=self.temperature)
        assert state._is_barostat_consistent(state._barostat)

        # If we feed a barostat with an inconsistent pressure, it's fixed.
        state = ThermodynamicState(self.inconsistent_pressure_alanine,
                                   temperature=self.temperature,
                                   pressure=self.pressure)
        assert state.pressure == self.pressure

        # The original system is unaltered.
        new_serialization = openmm.XmlSerializer.serialize(system)
        assert new_serialization == old_serialization

    def test_method_is_integrator_consistent(self):
        """The integrator must have the same temperature of the state."""
        inconsistent_temperature = self.temperature + 1.0*unit.kelvin
        friction = 5.0/unit.picosecond
        time_step = 2.0*unit.femtosecond
        state = ThermodynamicState(self.toluene_vacuum, self.temperature)

        compound = openmm.CompoundIntegrator()
        compound.addIntegrator(openmm.VerletIntegrator(time_step))
        compound.addIntegrator(openmm.LangevinIntegrator(self.temperature,
                                                         friction, time_step))
        compound.addIntegrator(openmm.LangevinIntegrator(inconsistent_temperature,
                                                         friction, time_step))

        test_cases = [
            (True, openmm.LangevinIntegrator(self.temperature,
                                             friction, time_step)),
            (False, openmm.LangevinIntegrator(inconsistent_temperature,
                                              friction, time_step)),
            (False, openmm.BrownianIntegrator(inconsistent_temperature,
                                              friction, time_step)),
            (True, 0), (True, 1), (False, 2)  # Compound integrator
        ]
        for consistent, integrator in test_cases:
            if isinstance(integrator, int):
                compound.setCurrentIntegrator(integrator)
                integrator = compound
            assert state._is_integrator_consistent(integrator) is consistent

            # create_context() should perform the same check.
            if not consistent:
                with nose.tools.assert_raises(ThermodynamicsError) as cm:
                    state.create_context(integrator)
                assert cm.exception.code == ThermodynamicsError.INCONSISTENT_INTEGRATOR

    def test_method_set_integrator_temperature(self):
        """ThermodynamicState._set_integrator_temperature() method."""
        temperature = self.temperature + 1.0*unit.kelvin
        friction = 5.0/unit.picosecond
        time_step = 2.0*unit.femtosecond
        state = ThermodynamicState(self.toluene_vacuum, self.temperature)

        integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
        assert state._set_integrator_temperature(integrator)
        assert integrator.getTemperature() == self.temperature
        assert not state._set_integrator_temperature(integrator)

        # It doesn't explode with integrators not coupled to a heat bath
        integrator = openmm.VerletIntegrator(time_step)
        assert not state._set_integrator_temperature(integrator)

    def test_method_turn_to_standard_system(self):
        """ThermodynamicState.turn_to_standard_system() class method."""
        system = copy.deepcopy(self.barostated_alanine)

        ThermodynamicState.turn_to_standard_system(system)
        assert ThermodynamicState._find_barostat(system) is None

    def test_method_create_context(self):
        """ThermodynamicState.create_context() method."""
        state = ThermodynamicState(self.toluene_vacuum, self.temperature)
        toluene_str = openmm.XmlSerializer.serialize(self.toluene_vacuum)

        test_cases = [
            (None, openmm.VerletIntegrator(1.0*unit.femtosecond)),
            (
                openmm.Platform.getPlatformByName('Reference'),
                openmm.LangevinIntegrator(self.temperature, 5.0/unit.picosecond,
                                          2.0*unit.femtosecond)
            )
        ]

        for platform, integrator in test_cases:
            context = state.create_context(integrator, platform)
            system_str = openmm.XmlSerializer.serialize(context.getSystem())
            assert system_str == toluene_str
            assert isinstance(context.getIntegrator(), integrator.__class__)
            if platform is not None:
                assert platform.getName() == context.getPlatform().getName()

    def test_method_is_compatible(self):
        """ThermodynamicState context and state compatibility methods."""
        def check_compatibility(state1, state2, is_compatible):
            assert state1.is_state_compatible(state2) is is_compatible
            assert state2.is_state_compatible(state1) is is_compatible
            time_step = 1.0*unit.femtosecond
            integrator1 = openmm.VerletIntegrator(time_step)
            integrator2 = openmm.VerletIntegrator(time_step)
            context1 = state1.create_context(integrator1)
            context2 = state2.create_context(integrator2)
            assert state1.is_context_compatible(context2) is is_compatible
            assert state2.is_context_compatible(context1) is is_compatible

        toluene_vacuum = ThermodynamicState(self.toluene_vacuum, self.temperature)
        toluene_implicit = ThermodynamicState(self.toluene_implicit, self.temperature)
        alanine_explicit = ThermodynamicState(self.alanine_explicit, self.temperature)
        barostated_alanine = ThermodynamicState(self.barostated_alanine, self.temperature)

        check_compatibility(toluene_vacuum, toluene_vacuum, True)
        check_compatibility(toluene_vacuum, toluene_implicit, False)
        check_compatibility(toluene_implicit, alanine_explicit, False)

        # When we set the system, cached values are updated correctly.
        toluene_implicit.system = self.toluene_vacuum
        check_compatibility(toluene_vacuum, toluene_implicit, True)

        # Different values of temperature/pressure do not affect compatibility.
        toluene_implicit.temperature = self.temperature + 1.0*unit.kelvin
        check_compatibility(toluene_vacuum, toluene_implicit, True)
        check_compatibility(alanine_explicit, barostated_alanine, True)

    def test_method_apply_to_context(self):
        """ThermodynamicState.apply_to_context() method."""
        friction = 5.0/unit.picosecond
        time_step = 2.0*unit.femtosecond
        integrator = openmm.LangevinIntegrator(self.temperature, friction, time_step)
        state0 = ThermodynamicState(self.alanine_explicit, self.temperature)
        context = state0.create_context(integrator)

        # Convert context to constant pressure.
        state1 = ThermodynamicState(self.barostated_alanine, self.temperature)
        state1.apply_to_context(context)
        context.setPositions(self.alanine_positions)
        barostat = ThermodynamicState._find_barostat(context.getSystem())
        assert barostat.getDefaultPressure() == self.pressure

        # The cached parameters on the context must be updated.
        old_box_vectors = context.getState(getPositions=True).getPeriodicBoxVectors(asNumpy=True)
        integrator.step(100)
        new_box_vectors = context.getState(getPositions=True).getPeriodicBoxVectors(asNumpy=True)
        assert not np.allclose(old_box_vectors, new_box_vectors)

        # Switch to different pressure and temperature.
        pressure = self.pressure + 1.0*unit.bar
        temperature = self.temperature + 10.0*unit.kelvin
        state2 = ThermodynamicState(self.barostated_alanine, temperature, pressure)
        state2.apply_to_context(context)
        barostat = ThermodynamicState._find_barostat(context.getSystem())
        assert barostat.getDefaultPressure() == pressure
        assert get_barostat_temperature(barostat) == temperature
        assert context.getIntegrator().getTemperature() == temperature

        # Now switch back to constant volume.
        state0.apply_to_context(context)
        assert ThermodynamicState._find_barostat(context.getSystem()) is None

    def test_method_reduced_potential(self):
        """ThermodynamicState.reduced_potential() method."""
        kJmol = unit.kilojoule_per_mole
        beta = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * self.temperature)
        state = ThermodynamicState(self.alanine_explicit, self.temperature)
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = state.create_context(integrator)
        context.setPositions(self.alanine_positions)
        sampler_state = SamplerState.from_context(context)

        # Compute constant volume reduced potential.
        reduced_potential = state.reduced_potential(sampler_state)
        potential_energy = reduced_potential / beta / kJmol
        assert np.isclose(sampler_state.potential_energy / kJmol, potential_energy)

        # Compute constant pressure reduced potential.
        state.pressure = self.pressure
        reduced_potential = state.reduced_potential(sampler_state)
        pressure_volume_work = (self.pressure * sampler_state.volume *
                                unit.AVOGADRO_CONSTANT_NA)
        potential_energy = (reduced_potential / beta - pressure_volume_work) / kJmol
        assert np.isclose(sampler_state.potential_energy / kJmol, potential_energy)


# =============================================================================
# TEST SAMPLER STATE
# =============================================================================

class TestSamplerState(object):

    @classmethod
    def setup_class(cls):
        temperature = 300*unit.kelvin
        alanine_vacuum = testsystems.AlanineDipeptideVacuum()
        cls.alanine_vacuum_positions = alanine_vacuum.positions
        cls.alanine_vacuum_state = ThermodynamicState(alanine_vacuum.system,
                                                      temperature)

        alanine_explicit = testsystems.AlanineDipeptideExplicit()
        cls.alanine_explicit_positions = alanine_explicit.positions
        cls.alanine_explicit_state = ThermodynamicState(alanine_explicit.system,
                                                        temperature)

    @staticmethod
    def is_sampler_state_equal_context(sampler_state, context):
        equal = True
        openmm_state = context.getState(getPositions=True, getEnergy=True,
                                        getVelocities=True)
        equal = equal and np.all(sampler_state.positions == openmm_state.getPositions())
        equal = equal and np.all(sampler_state.velocities == openmm_state.getVelocities())
        equal = equal and np.all(sampler_state.box_vectors == openmm_state.getPeriodicBoxVectors())
        equal = equal and sampler_state.potential_energy == openmm_state.getPotentialEnergy()
        equal = equal and sampler_state.kinetic_energy == openmm_state.getKineticEnergy()
        nm3 = unit.nanometers**3
        equal = equal and np.isclose(sampler_state.volume / nm3, openmm_state.getPeriodicBoxVolume() / nm3)
        return equal

    def test_inconsistent_velocities(self):
        """Exception raised with inconsistent velocities."""
        positions = self.alanine_vacuum_positions
        sampler_state = SamplerState(positions)

        # If velocities have different length, an error is raised.
        velocities = [0.0 for _ in range(len(positions) - 1)]
        with nose.tools.assert_raises(SamplerStateError) as cm:
            sampler_state.velocities = velocities
        assert cm.exception.code == SamplerStateError.INCONSISTENT_VELOCITIES

        # The same happens in constructor.
        with nose.tools.assert_raises(SamplerStateError) as cm:
            SamplerState(positions, velocities)
        assert cm.exception.code == SamplerStateError.INCONSISTENT_VELOCITIES

    def test_constructor_from_context(self):
        """SamplerState.from_context constructor."""
        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        alanine_vacuum_context = self.alanine_vacuum_state.create_context(integrator)
        alanine_vacuum_context.setPositions(self.alanine_vacuum_positions)

        sampler_state = SamplerState.from_context(alanine_vacuum_context)
        assert self.is_sampler_state_equal_context(sampler_state, alanine_vacuum_context)

    def test_method_is_context_compatible(self):
        """SamplerState.is_context_compatible() method."""
        time_step = 1*unit.femtosecond
        integrator1 = openmm.VerletIntegrator(time_step)
        integrator2 = openmm.VerletIntegrator(time_step)

        # Vacuum.
        alanine_vacuum_context = self.alanine_vacuum_state.create_context(integrator1)
        vacuum_sampler_state = SamplerState(self.alanine_vacuum_positions)

        # Explicit solvent.
        alanine_explicit_context = self.alanine_explicit_state.create_context(integrator2)
        explicit_sampler_state = SamplerState(self.alanine_explicit_positions)

        assert vacuum_sampler_state.is_context_compatible(alanine_vacuum_context)
        assert not vacuum_sampler_state.is_context_compatible(alanine_explicit_context)
        assert explicit_sampler_state.is_context_compatible(alanine_explicit_context)
        assert not explicit_sampler_state.is_context_compatible(alanine_vacuum_context)

    def test_method_update_from_context(self):
        """SamplerState.update_from_context() method."""
        time_step = 1*unit.femtosecond
        integrator1 = openmm.VerletIntegrator(time_step)
        integrator2 = openmm.VerletIntegrator(time_step)
        vacuum_context = self.alanine_vacuum_state.create_context(integrator1)
        explicit_context = self.alanine_explicit_state.create_context(integrator2)

        # Test that the update is successful
        vacuum_context.setPositions(self.alanine_vacuum_positions)
        sampler_state = SamplerState.from_context(vacuum_context)
        integrator1.step(10)
        assert not self.is_sampler_state_equal_context(sampler_state, vacuum_context)
        sampler_state.update_from_context(vacuum_context)
        assert self.is_sampler_state_equal_context(sampler_state, vacuum_context)

        # Trying to update with an inconsistent context raise error.
        explicit_context.setPositions(self.alanine_explicit_positions)
        with nose.tools.assert_raises(SamplerStateError) as cm:
            sampler_state.update_from_context(explicit_context)
        assert cm.exception.code == SamplerStateError.INCONSISTENT_VELOCITIES

    def test_method_apply_to_context(self):
        """SamplerState.apply_to_context() method."""
        integrator = openmm.VerletIntegrator(1*unit.femtosecond)
        explicit_context = self.alanine_explicit_state.create_context(integrator)
        explicit_context.setPositions(self.alanine_explicit_positions)
        sampler_state = SamplerState.from_context(explicit_context)

        integrator.step(10)
        assert not self.is_sampler_state_equal_context(sampler_state, explicit_context)
        sampler_state.apply_to_context(explicit_context)
        assert self.is_sampler_state_equal_context(sampler_state, explicit_context)


# =============================================================================
# TEST COMPOUND STATE
# =============================================================================

class TestCompoundThermodynamicState(object):

    class DummyState(object):
        """A state that keeps track of a useless system parameter."""

        standard_dummy_parameter = 1.0

        def __init__(self, dummy_parameter):
            self._dummy_parameter = dummy_parameter

        @property
        def dummy_parameter(self):
            return self._dummy_parameter

        @dummy_parameter.setter
        def dummy_parameter(self, value):
            self._dummy_parameter = value

        @classmethod
        def turn_to_standard_system(cls, system):
            try:
                cls.set_dummy_parameter(system, cls.standard_dummy_parameter)
            except TypeError:  # No parameter to set.
                pass

        def set_system_state(self, system):
            self.set_dummy_parameter(system, self.dummy_parameter)

        def check_system_consistency(self, system):
            dummy_parameter = TestCompoundThermodynamicState.get_dummy_parameter(system)
            if dummy_parameter != self.dummy_parameter:
                raise ValueError

        @staticmethod
        def is_context_compatible(context):
            parameters = context.getState(getParameters=True).getParameters()
            if 'dummy_parameters' in parameters.keys():
                return True
            else:
                return False

        def apply_to_context(self, context):
            context.setParameter('dummy_parameter', self.dummy_parameter)

        @classmethod
        def add_dummy_parameter(cls, system):
            """Add to system a CustomBondForce with a dummy parameter."""
            force = openmm.CustomBondForce('dummy_parameter')
            force.addGlobalParameter('dummy_parameter', cls.standard_dummy_parameter)
            system.addForce(force)

        @staticmethod
        def _find_dummy_force(system):
            for force in system.getForces():
                if isinstance(force, openmm.CustomBondForce):
                    for parameter_id in range(force.getNumGlobalParameters()):
                        parameter_name = force.getGlobalParameterName(parameter_id)
                        if parameter_name == 'dummy_parameter':
                            return force, parameter_id

        @classmethod
        def set_dummy_parameter(cls, system, value):
            force, parameter_id = cls._find_dummy_force(system)
            force.setGlobalParameterDefaultValue(parameter_id, value)

    @classmethod
    def get_dummy_parameter(cls, system):
        force, parameter_id = cls.DummyState._find_dummy_force(system)
        return force.getGlobalParameterDefaultValue(parameter_id)

    @classmethod
    def setup_class(cls):
        cls.pressure = 1.01325*unit.bars
        cls.temperature = 300*unit.kelvin

        cls.dummy_parameter = cls.DummyState.standard_dummy_parameter + 1.0
        cls.dummy_state = cls.DummyState(cls.dummy_parameter)

        alanine_explicit = testsystems.AlanineDipeptideExplicit().system
        cls.DummyState.add_dummy_parameter(alanine_explicit)
        cls.alanine_explicit = alanine_explicit

    def test_dynamic_inheritance(self):
        """ThermodynamicState is inherited dinamically."""
        thermodynamic_state = ThermodynamicState(self.alanine_explicit,
                                                 self.temperature)
        compound_state = CompoundThermodynamicState(thermodynamic_state, [])

        assert isinstance(compound_state, ThermodynamicState)

        # Attributes are correctly read.
        assert hasattr(compound_state, 'pressure')
        assert compound_state.pressure is None
        assert hasattr(compound_state, 'temperature')
        assert compound_state.temperature == self.temperature

        # Properties and attributes are correctly set.
        new_temperature = self.temperature + 1.0*unit.kelvin
        compound_state.pressure = self.pressure
        compound_state._temperature = new_temperature
        assert compound_state._barostat.getDefaultPressure() == self.pressure
        assert compound_state.temperature == new_temperature

    def test_constructor_set_state(self):
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.temperature)

        assert self.get_dummy_parameter(thermodynamic_state.system) != self.dummy_parameter
        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])
        assert self.get_dummy_parameter(compound_state.system) == self.dummy_parameter

    def test_property_forwarding(self):
        """Forward properties to IComposableStates and update system."""
        dummy_state = self.DummyState(self.dummy_parameter + 1.0)
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.temperature)
        compound_state = CompoundThermodynamicState(thermodynamic_state, [dummy_state])

        # Properties are correctly read and set, and
        # the system is updated to the new value.
        assert compound_state.dummy_parameter != self.dummy_parameter
        assert self.get_dummy_parameter(compound_state.system) != self.dummy_parameter
        compound_state.dummy_parameter = self.dummy_parameter
        assert compound_state.dummy_parameter == self.dummy_parameter
        assert self.get_dummy_parameter(compound_state.system) == self.dummy_parameter

        # Default behavior for attribute error and monkey patching.
        with nose.tools.assert_raises(AttributeError):
            compound_state.temp
        compound_state.temp = 0
        assert 'temp' in compound_state.__dict__

    def test_property_system(self):
        """CompoundThermodynamicState.system setting."""
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.temperature)
        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])

        # Setting an inconsistent system for the dummy raises an error.
        system = compound_state.system
        self.DummyState.set_dummy_parameter(system, self.dummy_parameter + 1.0)
        with nose.tools.assert_raises(ValueError):
            compound_state.system = system

    def test_method_turn_to_standard_system(self):
        """CompoundThermodynamicState.turn_to_standard_system method."""
        alanine_explicit = copy.deepcopy(self.alanine_explicit)
        thermodynamic_state = ThermodynamicState(alanine_explicit, self.temperature)
        thermodynamic_state.pressure = self.pressure
        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])

        system = thermodynamic_state.system
        assert ThermodynamicState._find_barostat(system) is not None
        assert self.get_dummy_parameter(system) == self.dummy_parameter
        compound_state.turn_to_standard_system(system)
        assert ThermodynamicState._find_barostat(system) is None
        assert self.get_dummy_parameter(system) == self.DummyState.standard_dummy_parameter

        # We still haven't computed the ThermodynamicState system hash
        # (pre-condition). Check that the standard system hash is correct.
        assert thermodynamic_state._cached_standard_system_hash is None
        standard_hash = openmm.XmlSerializer.serialize(system).__hash__()
        assert standard_hash == compound_state._standard_system_hash

        # Check that is_state_compatible work.
        undummied_alanine = testsystems.AlanineDipeptideExplicit().system
        incompatible_state = ThermodynamicState(undummied_alanine, self.temperature)
        assert not compound_state.is_state_compatible(incompatible_state)

        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        context = incompatible_state.create_context(integrator)
        assert not compound_state.is_context_compatible(context)

    def test_method_apply_to_context(self):
        """CompoundThermodynamicState.apply_to_context() method."""
        dummy_parameter = self.DummyState.standard_dummy_parameter
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.temperature)
        self.DummyState.set_dummy_parameter(thermodynamic_state.system, dummy_parameter)

        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        context = thermodynamic_state.create_context(integrator)
        assert context.getParameter('dummy_parameter') == dummy_parameter
        assert ThermodynamicState._find_barostat(context.getSystem()) is None

        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])
        compound_state.pressure = 1.0*unit.atmosphere  # Add barostat.
        compound_state.apply_to_context(context)
        assert context.getParameter('dummy_parameter') == self.dummy_parameter
        assert ThermodynamicState._find_barostat(context.getSystem()) is not None
