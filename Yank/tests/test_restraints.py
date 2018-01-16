#!/usr/bin/python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Test restraints module.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import os
import math
import copy

import numpy as np
import netCDF4 as netcdf
from simtk import openmm, unit
import openmmtools as mmtools
from openmmtools import testsystems, states
import nose
from nose.plugins.attrib import attr

import yank.restraints
from yank import experiment, analyze, Topography


# =============================================================================================
# UNIT TESTS
# =============================================================================================

class HostGuestNoninteracting(testsystems.HostGuestVacuum):
    """CB7:B2 host-guest system in vacuum with no nonbonded interactions.

    Parameters
    ----------
    Same as HostGuestVacuum

    Examples
    --------
    Create host:guest system with no nonbonded interactions.
    >>> testsystem = HostGuestVacuumNoninteracting()
    >>> system, positions = testsystem.system, testsystem.positions

    Properties
    ----------
    receptor_atoms : list of int
        Indices of receptor atoms
    ligand_atoms : list of int
        Indices of ligand atoms

    """
    def __init__(self, **kwargs):
        super(HostGuestNoninteracting, self).__init__(**kwargs)

        # Store receptor and ligand atom indices
        self.receptor_atoms = range(0, 126)
        self.ligand_atoms = range(126, 156)

        # Remove nonbonded interactions
        force_indices = {self.system.getForce(index).__class__.__name__: index
                         for index in range(self.system.getNumForces())}
        self.system.removeForce(force_indices['NonbondedForce'])

    @staticmethod
    def build_test_case():
        """Create a new ThermodynamicState, SamplerState and Topography."""
        # Create a test system
        t = HostGuestNoninteracting()

        # Create states and topography encoding the info to determine the parameters.
        topography = Topography(t.topology, ligand_atoms='resname B2')
        sampler_state = states.SamplerState(positions=t.positions)
        thermodynamic_state = states.ThermodynamicState(system=t.system, temperature=300.0*unit.kelvin)
        return thermodynamic_state, sampler_state, topography


expected_restraints = {
    'Harmonic': yank.restraints.Harmonic,
    'FlatBottom': yank.restraints.FlatBottom,
    'Boresch': yank.restraints.Boresch,
}

restraint_test_yaml = """
---
options:
  minimize: no
  verbose: yes
  output_dir: %(output_directory)s
  number_of_iterations: %(number_of_iter)s
  nsteps_per_iteration: 100
  temperature: 300*kelvin
  pressure: null
  anisotropic_dispersion_cutoff: null
  platform: OpenCL

solvents:
  vacuum:
    nonbonded_method: PME
    nonbonded_cutoff: 0.59 * nanometer

systems:
  ship:
    phase1_path: [data/benzene-toluene-standard-state/standard_state_complex.inpcrd, data/benzene-toluene-standard-state/standard_state_complex.prmtop]
    phase2_path: [data/benzene-toluene-standard-state/standard_state_complex.inpcrd, data/benzene-toluene-standard-state/standard_state_complex.prmtop]
    ligand_dsl: resname ene
    solvent: vacuum

protocols:
  absolute-binding:
    complex:
      alchemical_path:
        lambda_restraints:     [0.0, 0.25, 0.5, 0.75, 1.0]
        lambda_electrostatics: [0.0, 0.00, 0.0, 0.00, 0.0]
        lambda_sterics:        [0.0, 0.00, 0.0, 0.00, 0.0]
    solvent:
      alchemical_path:
        lambda_electrostatics: [1.0, 1.0]
        lambda_sterics:        [1.0, 1.0]

experiments:
  system: ship
  protocol: absolute-binding
  restraint:
    type: %(restraint_type)s
"""


def general_restraint_run(options):
    """
    Generalized restraint simulation run to test free energy = standard state correction.

    options : Dict. A dictionary of substitutions for restraint_test_yaml
    """
    with mmtools.utils.temporary_directory() as output_directory:
        # TODO refactor this to use AlchemicalPhase API rather than a YAML script.
        options['output_directory'] = output_directory
        # run both setup and experiment
        yaml_builder = experiment.ExperimentBuilder(restraint_test_yaml % options)
        yaml_builder.run_experiments()
        # Estimate Free Energies
        ncfile_path = os.path.join(output_directory, 'experiments', 'complex.nc')
        ncfile = netcdf.Dataset(ncfile_path, 'r')
        Deltaf_ij, dDeltaf_ij = analyze.estimate_free_energies(ncfile)
        # Correct the sign for the fact that we are adding vs removing the restraints
        DeltaF_simulated = Deltaf_ij[-1, 0]
        dDeltaF_simulated = dDeltaf_ij[-1, 0]
        DeltaF_restraints = ncfile.groups['metadata'].variables['standard_state_correction'][0]
        ncfile.close()

    # Check if they are close
    assert np.allclose(DeltaF_restraints, DeltaF_simulated, rtol=dDeltaF_simulated)


@attr('slow')  # Skip on Travis-CI
def test_harmonic_free_energy():
    """
    Test that the harmonic restraint simulated free energy equals the standard state correction
    """
    options = {'number_of_iter': '500',
               'restraint_type': 'Harmonic'}
    general_restraint_run(options)


@attr('slow')  # Skip on Travis-CI
def test_flat_bottom_free_energy():
    """
    Test that the harmonic restraint simulated free energy equals the standard state correction
    """
    options = {'number_of_iter': '500',
               'restraint_type': 'FlatBottom'}
    general_restraint_run(options)


@attr('slow')  # Skip on Travis-CI
def test_Boresch_free_energy():
    """
    Test that the harmonic restraint simulated free energy equals the standard state correction
    """
    # These need more samples to converge
    options = {'number_of_iter': '1000',
               'restraint_type': 'Boresch'}
    general_restraint_run(options)


def test_harmonic_standard_state():
    """
    Test that the expected harmonic standard state correction is close to our approximation

    Also ensures that PBC bonds are being computed and disabled correctly as expected
    """
    LJ_fluid = testsystems.LennardJonesFluid()

    # Create Harmonic restraint.
    restraint = yank.restraints.create_restraint('Harmonic', restrained_receptor_atoms=1)

    # Determine other parameters.
    ligand_atoms = [3, 4, 5]
    topography = Topography(LJ_fluid.topology, ligand_atoms=ligand_atoms)
    sampler_state = states.SamplerState(positions=LJ_fluid.positions)
    thermodynamic_state = states.ThermodynamicState(system=LJ_fluid.system,
                                                    temperature=300.0 * unit.kelvin)
    restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    spring_constant = restraint.spring_constant

    # Compute standard-state volume for a single molecule in a box of size (1 L) / (avogadros number)
    liter = 1000.0 * unit.centimeters ** 3  # one liter
    box_volume = liter / (unit.AVOGADRO_CONSTANT_NA * unit.mole)  # standard state volume
    analytical_shell_volume = (2 * math.pi / (spring_constant * thermodynamic_state.beta))**(3.0/2)
    analytical_standard_state_G = - math.log(box_volume / analytical_shell_volume)
    restraint_standard_state_G = restraint.get_standard_state_correction(thermodynamic_state)
    np.testing.assert_allclose(analytical_standard_state_G, restraint_standard_state_G)


# ==============================================================================
# RESTRAINT PARAMETER DETERMINATION
# ==============================================================================

def test_partial_parametrization():
    """The automatic restraint parametrization doesn't overwrite user values."""
    # Create states and identify ligand/receptor.
    test_system = testsystems.HostGuestVacuum()
    topography = Topography(test_system.topology, ligand_atoms='resname B2')
    sampler_state = states.SamplerState(positions=test_system.positions)
    thermodynamic_state = states.ThermodynamicState(test_system.system,
                                                    temperature=300.0*unit.kelvin)

    # Test case: (restraint_type, constructor_kwargs)
    test_cases = [
        ('Harmonic', dict(spring_constant=2.0*unit.kilojoule_per_mole/unit.nanometer**2,
                          restrained_receptor_atoms=[5])),
        ('FlatBottom', dict(well_radius=1.0*unit.angstrom, restrained_ligand_atoms=[130])),
        ('Boresch', dict(restrained_ligand_atoms=[130, 131, 136],
                         K_r=1.0*unit.kilojoule_per_mole/unit.angstroms**2))
    ]

    for restraint_type, kwargs in test_cases:
        state = copy.deepcopy(thermodynamic_state)
        restraint = yank.restraints.create_restraint(restraint_type, **kwargs)

        # Test-precondition: The restraint has undefined parameters.
        with nose.tools.assert_raises(yank.restraints.RestraintParameterError):
            restraint.restrain_state(state)

        # The automatic parametrization maintains user values.
        restraint.determine_missing_parameters(state, sampler_state, topography)
        for parameter_name, parameter_value in kwargs.items():
            assert getattr(restraint, parameter_name) == parameter_value

        # The rest of the parameters has been determined.
        restraint.get_standard_state_correction(state)

        # The force has been configured correctly.
        restraint.restrain_state(state)
        system = state.system
        for force in system.getForces():
            # RadiallySymmetricRestraint between two single atoms.
            if isinstance(force, openmm.CustomBondForce):
                particle1, particle2, _ = force.getBondParameters(0)
                assert particle1 == restraint.restrained_receptor_atoms[0]
                assert particle2 == restraint.restrained_ligand_atoms[0]
            # Boresch restraint.
            elif isinstance(force, openmm.CustomCompoundBondForce):
                particles, _ = force.getBondParameters(0)
                assert particles == tuple(restraint.restrained_receptor_atoms + restraint.restrained_ligand_atoms)


def restraint_selection_template(topography_ligand_atoms=None,
                                 restrained_receptor_atoms=None,
                                 restrained_ligand_atoms=None,
                                 topography_regions=None):
    """The DSL atom selection works as expected."""
    test_system = testsystems.HostGuestVacuum()
    topography = Topography(test_system.topology, ligand_atoms=topography_ligand_atoms)
    if topography_regions is not None:
        for region, selection in topography_regions.items():
            topography.add_region(region, selection)
    sampler_state = states.SamplerState(positions=test_system.positions)
    thermodynamic_state = states.ThermodynamicState(test_system.system,
                                                    temperature=300.0 * unit.kelvin)

    # Iniialize with DSL and without processing the string raises an error.
    restraint = yank.restraints.Harmonic(spring_constant=2.0 * unit.kilojoule_per_mole / unit.nanometer ** 2,
                                         restrained_receptor_atoms=restrained_receptor_atoms,
                                         restrained_ligand_atoms=restrained_ligand_atoms)
    with nose.tools.assert_raises(yank.restraints.RestraintParameterError):
        restraint.restrain_state(thermodynamic_state)

    # After parameter determination, the indices of the restrained atoms are correct.
    restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    assert len(restraint.restrained_receptor_atoms) == 14
    assert len(restraint.restrained_ligand_atoms) == 30

    # The bond force is configured correctly.
    restraint.restrain_state(thermodynamic_state)
    system = thermodynamic_state.system
    for force in system.getForces():
        if isinstance(force, openmm.CustomCentroidBondForce):
            assert force.getBondParameters(0)[0] == (0, 1)
            assert len(force.getGroupParameters(0)[0]) == 14
            assert len(force.getGroupParameters(1)[0]) == 30
    assert isinstance(force, openmm.CustomCentroidBondForce)  # We have found a force.


def test_restraint_dsl_selection():
    """The DSL atom selection works as expected."""
    restraint_selection_template(topography_ligand_atoms='resname B2',
                                 restrained_receptor_atoms="(resname CUC) and (name =~ 'O[0-9]+')",
                                 restrained_ligand_atoms='resname B2')


def test_restraint_region_selection():
    """Test that the region atom selection works as expected"""
    restraint_selection_template(topography_ligand_atoms='resname B2',
                                 restrained_receptor_atoms='choice_res_residue and the_oxygen',
                                 restrained_ligand_atoms='choice_lig_residue',
                                 topography_regions={'choice_lig_residue': 'resname B2',
                                                     'choice_res_residue': 'resname CUC',
                                                     'the_oxygen': "name =~ 'O[0-9]+'"})


def test_restraint_region_dsl_mix():
    """Test that the region atom selection works as expected"""
    restraint_selection_template(topography_ligand_atoms='resname B2',
                                 restrained_receptor_atoms='choice_res_residue and the_oxygen',
                                 restrained_ligand_atoms='resname B2',
                                 topography_regions={'choice_lig_residue': 'resname B2',
                                                     'choice_res_residue': 'resname CUC',
                                                     'the_oxygen': "name =~ 'O[0-9]+'"})


# ==============================================================================
# RESTRAINT FACTORY FUNCTIONS
# ==============================================================================

def test_available_restraint_classes():
    """Test to make sure expected restraint classes are available."""
    available_restraint_classes = yank.restraints.available_restraint_classes()
    available_restraint_types = yank.restraints.available_restraint_types()

    # We shouldn't have `None` (from the base class) as an available type
    assert None not in available_restraint_classes
    assert None not in available_restraint_types

    for restraint_type, restraint_class in expected_restraints.items():
        msg = "Failed comparing restraint type '%s' with %s" % (restraint_type, str(available_restraint_classes))
        assert restraint_type in available_restraint_classes, msg
        assert available_restraint_classes[restraint_type] is restraint_class, msg
        assert restraint_type in available_restraint_types, msg


def test_restraint_dispatch():
    """Test dispatch of various restraint types."""
    thermodynamic_state, sampler_state, topography = HostGuestNoninteracting.build_test_case()
    for restraint_type, restraint_class in expected_restraints.items():
        # Add restraints and determine parameters.
        thermo_state = copy.deepcopy(thermodynamic_state)
        restraint = yank.restraints.create_restraint(restraint_type)
        restraint.determine_missing_parameters(thermo_state, sampler_state, topography)

        # Check that we got the right restraint class.
        assert restraint.__class__.__name__ == restraint_type
        assert restraint.__class__ is restraint_class


def test_restraint_force_group():
    "The restraint force should be placed in its own force group for optimization."
    thermodynamic_state, sampler_state, topography = HostGuestNoninteracting.build_test_case()
    for restraint_type, restraint_class in expected_restraints.items():
        # Add restraints and determine parameters.
        thermo_state = copy.deepcopy(thermodynamic_state)
        restraint = yank.restraints.create_restraint(restraint_type)
        restraint.determine_missing_parameters(thermo_state, sampler_state, topography)
        restraint.restrain_state(thermo_state)

        # Find the force group of the restraint force.
        system = thermo_state.system
        for force_idx, force in enumerate(system.getForces()):
            try:
                parameter_name = force.getGlobalParameterName(0)
            except AttributeError:
                continue
            if parameter_name == 'lambda_restraints':
                restraint_force_idx = force_idx
                restraint_force_group = force.getForceGroup()
                break

        # No other force should have the same force group.
        for force_idx, force in enumerate(system.getForces()):
            if force_idx != restraint_force_idx:
                assert force.getForceGroup() != restraint_force_group


# ==============================================================================
# RESTRAINT STATE
# ==============================================================================

class TestRestraintState(object):
    """Test class RestraintState."""

    @classmethod
    def setup_class(cls):
        lysozyme = testsystems.LysozymeImplicit()
        system, positions = lysozyme.system, lysozyme.positions
        thermodynamic_state = states.ThermodynamicState(system, 300*unit.kelvin)
        sampler_state = states.SamplerState(positions)
        topography = Topography(lysozyme.topology, ligand_atoms='resname TMP')
        cls.lysozyme_test_case = (thermodynamic_state, sampler_state, topography)

    def get_restraint_cases(self):
        for cls_name, cls in yank.restraints.available_restraint_classes().items():
            # Create restraint and automatically determine parameters.
            restraint = cls()
            thermodynamic_state, sampler_state, topography = copy.deepcopy(self.lysozyme_test_case)
            restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)

            # Apply restraint.
            restraint.restrain_state(thermodynamic_state)

            # Create compound state to control the strength of the restraint.
            restraint_state = yank.restraints.RestraintState(lambda_restraints=1.0)
            compound_state = states.CompoundThermodynamicState(thermodynamic_state=thermodynamic_state,
                                                               composable_states=[restraint_state])
            yield compound_state

    def test_apply_to_system(self):
        """The System parameters are updated when lambda_restraints is set on the compound state."""
        for compound_state in self.get_restraint_cases():
            # Test pre-condition.
            assert compound_state.lambda_restraints == 1.0

            # Changing the attribute changes the internal representation of a system.
            compound_state.lambda_restraints = 0.5
            for force, parameter_id in compound_state._get_system_forces_parameters(compound_state.system):
                assert force.getGlobalParameterDefaultValue(parameter_id) == 0.5

    def test_apply_to_context(self):
        """The Context parameters are updated when the compound state is applied."""
        for compound_state in self.get_restraint_cases():
            compound_state.lambda_restraints = 0.5

            integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
            context = compound_state.create_context(integrator)
            assert context.getParameter('lambda_restraints') == 0.5

            compound_state.lambda_restraints = 0.0
            compound_state.apply_to_context(context)
            assert context.getParameter('lambda_restraints') == 0.0
            del context, integrator

    def test_compatibility(self):
        """States differing only by the strength of the restraint are compatible."""
        unrestrained_system = self.lysozyme_test_case[0].system

        for compound_state in self.get_restraint_cases():
            compound_state.lambda_restraints = 1.0
            compatible_state = copy.deepcopy(compound_state)
            compatible_state.lambda_restraints = 0.0
            assert compound_state.is_state_compatible(compatible_state)

            # Trying to assign a System without a Restraint raises an error.
            with nose.tools.assert_raises(yank.restraints.RestraintStateError):
                compound_state.system = unrestrained_system

    def test_find_force_groups_to_update(self):
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        for compound_state in self.get_restraint_cases():
            context = compound_state.create_context(copy.deepcopy(integrator))

            # Find the restraint force group.
            system = context.getSystem()
            force, _ = next(yank.restraints.RestraintState._get_system_forces_parameters(system))
            force_group = force.getForceGroup()

            # No force group should be updated if we don't move.
            assert compound_state._find_force_groups_to_update(context, compound_state) == set()

            # We need to update the force if the current state changes.
            compound_state2 = copy.deepcopy(compound_state)
            compound_state2.lambda_restraints = 0.5
            assert compound_state._find_force_groups_to_update(context, compound_state2) == {force_group}


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    test_restraint_dispatch()
