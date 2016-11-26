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
from simtk import unit
from openmmtools import testsystems

from yank.states import *


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class InconsistentThermodynamicState(ThermodynamicState):
    """ThermodynamicState that does not run consistency checks on init.

    It is useful to test private methods used to check for consistency.

    """
    def __init__(self, system=None, temperature=None, pressure=None):
        self._system = system
        self._temperature = temperature
        self._pressure = pressure


# =============================================================================
# TEST THERMODYNAMIC STATE
# =============================================================================

class TestThermodynamicState(object):

    @classmethod
    def setup_class(cls):
        """Create the test systems used in the test suite."""
        cls.temperature = 300*unit.kelvin
        cls.pressure = 1.0*unit.atmosphere
        cls.toluene_vacuum = testsystems.TolueneVacuum().system

        # A system correctly barostated
        cls.barostated_toluene = copy.deepcopy(cls.toluene_vacuum)
        barostat = openmm.MonteCarloBarostat(cls.pressure, cls.temperature)
        cls.barostated_toluene.addForce(barostat)

        # A system with two identical MonteCarloBarostats
        cls.multiple_barostat_toluene = copy.deepcopy(cls.barostated_toluene)
        barostat = openmm.MonteCarloBarostat(cls.pressure, cls.temperature)
        cls.multiple_barostat_toluene.addForce(barostat)

        # A system with an unsupported MonteCarloAnisotropicBarostat
        cls.unsupported_barostat_toluene = copy.deepcopy(cls.toluene_vacuum)
        pressure_in_bars = cls.pressure / unit.bar
        anisotropic_pressure = openmm.Vec3(pressure_in_bars, pressure_in_bars,
                                           pressure_in_bars)
        barostat = openmm.MonteCarloAnisotropicBarostat(anisotropic_pressure,
                                                        cls.temperature)
        cls.unsupported_barostat_toluene.addForce(barostat)

        # A system a barostated at the incorrect temperature
        cls.incompatible_pressure_barostat_toluene = copy.deepcopy(cls.toluene_vacuum)
        barostat = openmm.MonteCarloBarostat(cls.pressure,
                                             cls.temperature + 1*unit.kelvin)
        cls.incompatible_pressure_barostat_toluene.addForce(barostat)

        # A system a barostat at the incorrect pressure
        cls.incompatible_temperature_barostat_toluene = copy.deepcopy(cls.toluene_vacuum)
        barostat = openmm.MonteCarloBarostat(cls.pressure + 0.1*unit.atmosphere,
                                             cls.temperature)
        cls.incompatible_temperature_barostat_toluene.addForce(barostat)

    def test_method_find_barostat(self):
        """ThermodynamicState._find_barostat() method."""
        barostated_system = copy.deepcopy(self.barostated_toluene)
        barostat = ThermodynamicState._find_barostat(barostated_system)
        assert isinstance(barostat, openmm.MonteCarloBarostat)

        # Raise exception if multiple or unsupported barostats found
        multiple_system = self.multiple_barostat_toluene
        unsupported_system = self.unsupported_barostat_toluene

        test_cases = [(multiple_system, ThermodynamicsException.MULTIPLE_BAROSTATS),
                      (unsupported_system, ThermodynamicsException.UNSUPPORTED_BAROSTAT)]
        for system, err_code in test_cases:
            with nose.tools.assert_raises(ThermodynamicsException) as cm:
                ThermodynamicState._find_barostat(system)
            assert cm.exception.code == err_code

    def test_method_is_barostat_consistent(self):
        """ThermodynamicState._is_barostat_consistent() method."""
        temperature = 300*unit.kelvin
        pressure = 1.0*unit.atmosphere
        state = InconsistentThermodynamicState(None, temperature, pressure)

        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        assert state._is_barostat_consistent(barostat)
        barostat = openmm.MonteCarloBarostat(pressure + 0.2*unit.atmosphere, temperature)
        assert not state._is_barostat_consistent(barostat)
        barostat = openmm.MonteCarloBarostat(pressure, temperature + 10*unit.kelvin)
        assert not state._is_barostat_consistent(barostat)

    def test_method_configure_barostat(self):
        """ThermodynamicState._configure_barostat() method."""
        barostated_system = copy.deepcopy(self.barostated_toluene)
        temperature = self.temperature + 10.0*unit.kelvin
        pressure = self.pressure + 0.2*unit.atmosphere
        state = InconsistentThermodynamicState(barostated_system, temperature, pressure)

        state._configure_barostat(barostated_system)
        barostat = state._find_barostat(barostated_system)
        assert state._is_barostat_consistent(barostat)

    def test_method_add_barostat(self):
        """ThermodynamicState._add_barostat() method."""
        toluene_system = copy.deepcopy(self.toluene_vacuum)
        state = InconsistentThermodynamicState(temperature=self.temperature, pressure=self.pressure)
        assert state._find_barostat(toluene_system) is None  # Test pre-condition

        state._add_barostat(toluene_system)
        barostat = state._find_barostat(toluene_system)
        assert isinstance(barostat, openmm.MonteCarloBarostat)
        assert state._is_barostat_consistent(barostat)

    def test_constructor_npt_incompatible_systems(self):
        """Exception is raised on construction with NPT-incompatible systems."""
        TE = ThermodynamicsException  # shortcut
        test_cases = [(self.toluene_vacuum, TE.NO_BAROSTAT),
                      (self.multiple_barostat_toluene, TE.MULTIPLE_BAROSTATS),
                      (self.unsupported_barostat_toluene, TE.UNSUPPORTED_BAROSTAT),
                      (self.incompatible_pressure_barostat_toluene, TE.INCONSISTENT_BAROSTAT),
                      (self.incompatible_temperature_barostat_toluene, TE.INCONSISTENT_BAROSTAT)]
        for system, err_code in test_cases:
            with nose.tools.assert_raises(TE) as cm:
                ThermodynamicState(system=system, temperature=self.temperature,
                                   pressure=self.pressure)
            assert cm.exception.code == err_code

    def test_constructor_force_barostat(self):
        test_cases = [self.toluene_vacuum,
                      self.incompatible_pressure_barostat_toluene,
                      self.incompatible_temperature_barostat_toluene]

        for system in test_cases:
            old_hash = openmm.XmlSerializer.serialize(system).__hash__()

            # Force ThermodynamicState to add a barostat.
            state = ThermodynamicState(system=system, temperature=self.temperature,
                                       pressure=self.pressure, force_system_state=True)

            # The new system has now a compatible barostat.
            barostat = ThermodynamicState._find_barostat(state.system)
            assert isinstance(barostat, openmm.MonteCarloBarostat)
            assert state._is_barostat_consistent(barostat)

            # The original system is unaltered.
            new_hash = openmm.XmlSerializer.serialize(system).__hash__()
            assert new_hash == old_hash
