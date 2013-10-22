#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Markov chain Monte Carlo simulation framework.

DESCRIPTION

This module provides a framework for equilibrium sampling from a given thermodynamic state of
a biomolecule using a Markov chain Monte Carlo scheme.

CAPABILITIES
* molecular dynamics [assumed to be free of integration error]
* hybrid Monte Carlo
* generalized hybrid Monte Carlo
* sidechain rotamer switching
* constant-pH

NOTES

This is still in development.

REFERENCES

[1] Jun S. Liu. Monte Carlo Strategies in Scientific Computing. Springer, 2008.

EXAMPLES

Construct a simple MCMC simulation using Langevin dynamics moves.

>>> # Create a test system
>>> import testsystems
>>> [system, coordinates] = testsystems.AlanineDipeptideImplicit()
>>> # Create a move set containing just a molecular dynamics move.
>>> move_set = set() # set of moves to choose from
>>> move = LangevinDynamicsMove(system)
>>> move.weight = 1.0 # relative weight of move
>>> move.collision_rate = 5.0 / unit.picosecond # set collision rate
>>> move.timestep = 2.0 * units.femtoseconds # set the timestep
>>> move.nsteps = 500 # set the number of steps per move
>>> move_set.insert(move) # add move to move set
>>> # Create MCMC sampler
>>> sampler = MCMCSampler(system, move_set)
>>> niterations = 10 # set number of iterations to run
>>> sampler.run(niterations)

TODO

* Split this into a separate package, with individual files for each move type.

COPYRIGHT AND LICENSE

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import math
import numpy
import copy
import time

import simtk
import simtk.chem.openmm as openmm
import simtk.unit as units

#=============================================================================================
# REVISION CONTROL
#=============================================================================================

__version__ = "$Revision: 883 $" # TODO: Turn on keyword interpolation.

#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

#=============================================================================================
# Exceptions
#=============================================================================================

class NotImplementedException(Exception):
    """
    Exception denoting that the requested feature has not yet been implemented.

    """

class ParameterException(Exception):
    """
    Exception denoting that a parameter has been incorrectly furnished.

    """

#=============================================================================================
# Monte Carlo Move abstract base class
#=============================================================================================

class MarkovChainMonteCarloMove(object):
    """
    Markov chain Monte Carlo (MCMC) move abstract base class.

    Markov chain Monte Carlo (MCMC) simulations are constructed from a set of derived objects.
    
    """
    
    pass

#=============================================================================================
# Markov chain Monte Carlo sampler
#=============================================================================================

class MCMCSampler(object):
    """
    Markov chain Monte Carlo sampler.

    """

    def __init__(self, thermodynamic_state, move_set):
        """
        Initialize a Markov chain Monte Carlo sampler.

        ARGUMENTS

        thermodynamic_state (ThermodynamicState) - thermodynamic state to sample during MCMC run
        move_set (set of MarkovChainMonteCarloMove objects) - moves to attempt during MCMC run

        """

        # Store thermodynamic state.
        self.thermodynamic_state = thermodynamic_state

        # Store the move set.
        self.move_types = list()
        self.move_probabilities = list()
        for move in move_set:
            self.move_types.append(move)
            self.move_probabilities.append(weight.weight)
        # Normalize weights.
        self.move_probabilities = numpy.array(self.move_probabilities)
        self.move_probabilities[:] /= self.move_probabilities.sum()

    def run(self, niterations):
        """
        Run the sampler.
        
        """
        pass

#=============================================================================================
# Langevin dynamics move
#=============================================================================================

class LangevinDynamicsMove(MarkovChainMonteCarloMove):
    """
    Langevin dynamics segment as a (pseudo) Monte Carlo move.

    This move assigns a velocity from the Maxwell-Boltzmann distribution and executes a number
    of Maxwell-Boltzmann steps to propagate dynamics.  This is not a *true* Monte Carlo move,
    in that the generation of the correct distribution is only exact in the limit of infinitely
    small timestep; in other words, the discretization error is assumed to be negligible. Use
    HybridMonteCarloMove instead to ensure the exact distribution is generated.

    """

    def __init__(self, reference_system):
        """

        """

        # Set defaults
        self.collision_rate = 90.0 / units.picosecond
        self.timestep = 2.0 * units.femtosecond
        self.number_of_steps_per_iteration = 500

        # Initialize base class with System.
        MarkovChainMonteCarloMove.__init__(self, reference_system)

        return

    def _assign_Maxwell_Boltzmann_velocities(self, system, temperature):
        """Generate Maxwell-Boltzmann velocities.

        @param system the system for which velocities are to be assigned
        @type simtk.chem.openmm.System or System
        
        @param temperature the temperature at which velocities are to be assigned
        @type Quantity with units of temperature

        @return velocities drawn from the Maxwell-Boltzmann distribution at the appropriate temperature
        @returntype (natoms x 3) numpy array wrapped in Quantity with units of velocity

        TODO

        This could be sped up by introducing vector operations.

        """

        # Get number of atoms
        natoms = system.getNumParticles()

        # Create storage for velocities.        
        velocities = units.Quantity(numpy.zeros([natoms, 3], numpy.float32), units.nanometer / units.picosecond) # velocities[i,k] is the kth component of the velocity of atom i
  
        # Compute thermal energy and inverse temperature from specified temperature.
        kT = kB * temperature # thermal energy
        beta = 1.0 / kT # inverse temperature
  
        # Assign velocities from the Maxwell-Boltzmann distribution.
        for atom_index in range(natoms):
            mass = system.getParticleMass(atom_index) # atomic mass
            sigma = units.sqrt(kT / mass) # standard deviation of velocity distribution for each coordinate for this atom
            for k in range(3):
                velocities[atom_index,k] = sigma * numpy.random.normal()

        # Return velocities
        return velocities

    def run(self, state, positions):
        """
        Apply move to the specified system and coordinates.

        ARGUMENTS

        state (ThermodynamicState) - thermodynamic state
        positions (Quantity of numpy array) - current positions
        
        RETURNS

        new_positions

        """
        
        # Create integrator.
        integrator = openmm.LangevinIntegrator(state.temperature, self.collision_rate, self.timestep)

        # Create context.
        context = Context(state.system, integrator)
        
        # Set coordinates.
        context.setPositions(positions)

        # Assign Maxwell-Boltzmann velocities.
        velocities = self._assign_Maxwell_Boltzmann_velocities(state.system, state.temperature)
        context.setVelocities(velocities)
        
        # Run dynamics.
        integrator.step(self.number_of_steps_per_iteration)
        
        # Get coordinates.
        openmm_state = context.getState(getPositions=True)
        new_positions = openmm_state.getPositions(asNumpy=True)

        # Clean up.
        del context, integrator, openmm_state

        return new_positions
    
#=============================================================================================
# Hybrid Monte Carlo move
#=============================================================================================

class HybridMonteCarloMove(MarkovChainMonteCarloMove):
    """
    Langevin dynamics segment as a (pseudo) Monte Carlo move.

    This move assigns a velocity from the Maxwell-Boltzmann distribution and executes a number
    of Maxwell-Boltzmann steps to propagate dynamics.  This is not a *true* Monte Carlo move,
    in that the generation of the correct distribution is only exact in the limit of infinitely
    small timestep; in other words, the discretization error is assumed to be negligible. Use
    HybridMonteCarloMove instead to ensure the exact distribution is generated.

    """

    def __init__(self, reference_system):
        """

        """

        # Set defaults
        self.timestep = 2.0 * units.femtosecond
        self.number_of_steps_per_iteration = 500

        return


    def _assign_Maxwell_Boltzmann_velocities(self, system, temperature):
        """Generate Maxwell-Boltzmann velocities.

        @param system the system for which velocities are to be assigned
        @type simtk.chem.openmm.System or System
        
        @param temperature the temperature at which velocities are to be assigned
        @type Quantity with units of temperature

        @return velocities drawn from the Maxwell-Boltzmann distribution at the appropriate temperature
        @returntype (natoms x 3) numpy array wrapped in Quantity with units of velocity

        TODO

        This could be sped up by introducing vector operations.

        """

        # Get number of atoms
        natoms = system.getNumParticles()

        # Create storage for velocities.        
        velocities = units.Quantity(numpy.zeros([natoms, 3], numpy.float32), units.nanometer / units.picosecond) # velocities[i,k] is the kth component of the velocity of atom i
  
        # Compute thermal energy and inverse temperature from specified temperature.
        kT = kB * temperature # thermal energy
        beta = 1.0 / kT # inverse temperature
  
        # Assign velocities from the Maxwell-Boltzmann distribution.
        for atom_index in range(natoms):
            mass = system.getParticleMass(atom_index) # atomic mass
            sigma = units.sqrt(kT / mass) # standard deviation of velocity distribution for each coordinate for this atom
            for k in range(3):
                velocities[atom_index,k] = sigma * numpy.random.normal()

        # Return velocities
        return velocities

    def run(self, state, positions):
        """
        Apply move to the specified system and coordinates.

        ARGUMENTS

        state (ThermodynamicState) - thermodynamic state
        positions (Quantity of numpy array) - current positions
        
        RETURNS

        new_positions

        """
        
        # Create integrator.
        integrator = openmm.VerletIntegrator(self.timestep)

        # Create context.
        context = Context(state.system, integrator)
        
        # Set coordinates.
        context.setPositions(positions)

        # Assign Maxwell-Boltzmann velocities.
        velocities = self._assign_Maxwell_Boltzmann_velocities(state.system, state.temperature)
        context.setVelocities(velocities)

        # Compute initial total energy.
        total_energy = 
        
        # Run dynamics.
        integrator.step(self.number_of_steps_per_iteration)
        
        # Get coordinates.
        openmm_state = context.getState(getPositions=True)
        new_positions = openmm_state.getPositions(asNumpy=True)

        # Clean up.
        del context, integrator, openmm_state

        return new_positions

#=============================================================================================
# Monte Carlo volume move for constant-pressure simulation
#=============================================================================================

class MonteCarloVolumeMove(MarkovChainMonteCarloMove):
    """
    Monte Carlo volume move for constant-pressure simulation.

    NOTES

    Gaussian displacements in volume are attempted, and the box geometry scaled isotropically.
    Because the Gaussian tails extend into a region where the box volume could be negative, this region is excluded from draws
    and Metropolis-Hastings is used to ensure proper satisfaction of detailed balace.

    If the old volume is Vold and the new volume Vnew, with DeltaV = (Vnew - Vold) the Gaussian random variate with std dev sigma,
    the Metropolis-Hastings acceptance probability can be shown to be

    P_accept = [P(Vnew)/P(Vold)] [P(Vold | Vnew)/P(Vnew | Vold)]

    where

    P(Vnew)/P(Vold) = exp[- beta (U(Vnew) - U(Vold) + p DeltaV ]

    and

    P(Vold | Vnew)/P(Vnew | Vold) = (Vnew/Vold)^N [1 + erf(Vnew/sigma/sqrt(2))] / [1 + erf(Vold/sigma/sqrt(2))]                                              

    Molecular scaling is used to scale the molecular centers of mass when the box is rescaled.

    During pre-equilibration, the Gaussian displacement standard deviation mcvolume_sigma is adjusted heuristically
    to obtain an acceptance probability of ~ 50% if the 'mcvolume_adjust_sigma' flag is set to .TRUE.
    This option MUST be switched off during equilibration and production or else detailed balance will be violated.

    This method must be combined with a canonical (NVT) sampling method to produce the isothermal-isobaric (NPT) ensemble.

    REFERENCES

    TODO:

    Introduce 'tune' function?

    """

    def __init__(self, reference_system):
        """
        Initialize Monte Carlo volume move scheme.

        """

        # Set default options.
        self.proposal_method = 'gaussian' # volume change proposal method 'gaussian' or 'uniform'
        self.sigma = 0.001 * self._volume(reference_system) # volume change proposal magnitude
        self.ntrials = 10 # number of volume move attempts per iteration
        self.method = 'atomic-scaling' # method for volume move: 'atomic-scaling' is supported now, 'molecular scaling' later

        self.adjust_move_size = False # whether move size should be adjusted or not
        self.adjust_interval = 1 # interval between move size adjustments, in cycles
        self.target_acceptance = 0.5 # target value for acceptance rate for adjusting moves
        self.sigma_inflation_factor = 1.50 # factor for scaling move size up
        self.sigma_deflation_factor = 0.95 # factor for scaling move size down

        # Build a list of molecules, in case we use molecular scaling.
        self.molecules = reference_system.enumerateMolecules()

        # Reset statistics.
        self.nattempted = 0 # number of moves attempted since last update
        self.naccepted = 0 # number of moves accepted since last move size adjustment adjustment
            
        return

    def _volume(box_vectors):
        """
        Compute the volume of the given box vectors.

        ARGUMENTS

        box_vectors - a list of numpy 3-vectors with units

        RETURNS

        volume (Quantity) - the box volume, in units of length**3

        """
        
        # Compute volume of parallelepiped.
        import numpy.linalg
        [a,b,c] = self.box_vectors
        A = numpy.array([a/a.unit, b/a.unit, c/a.unit])
        volume = numpy.abs(A) * a.unit**3
        return volume        

    def run(self, state, positions):
        """
        Apply move to the specified system and coordinates.

        ARGUMENTS

        state (ThermodynamicState) - thermodynamic state
        positions (Quantity of numpy array) - current positions
        
        RETURNS

        new_positions

        """

        # Compute reduced potential for initial coordinates.
        old_reduced_potential = state.reduced_potential(positions)
        old_volume = self._volume(state.system.getPeriodicBoxVectors)
        
        for attempt in range(self.nattempts):
            if verbose: print "MC volume attempt %d / %d" % (attempt, self.nattempts)

            # Choose volume perturbation.
            DeltaV = 0.0
            if self.volume_move_method == 'gaussian':
                DeltaV = numpy.random.randn() * self.volume_move_size
            elif self.volume_move_method == 'uniform':
                DeltaV = (2.0 * numpy.random.rand() - 1.0) * self.volume_move_size
            else:
                raise Exception("Unknown volume move method '%s'." % self.volume_move_method)

            # Compute new box volume.
            volume = old_volume + DeltaV # new box volume
            
            # Determine scaling factor for box vectors and atomic or molecular positions.
            scale_factor = ((volume + DeltaV) / volume)**(1.0/3.0)

            # Scale box.
            [a,b,c] = state.system.getPeriodicBoxVectors()
            state.system.setPeriodicBoxVectors(a * scale_factor, b * scale_factor, c * scale_factor)

            # Scale atomic positions.
            if self.scaling_method = 'atomic':
                positions = old_positions * scale_factor
                nscale = state.system.natoms # number of atoms
            elif self.scaling_method = 'molecular':
                for molecule in state.system.iterateMolecules():
                    com = old_positions[molecule.atom_indices].mean() # center of mass position
                    translation = com * (scale_factor - 1.0) # translation shift to apply to all molecular coordinates
                    positions[molecule.atom_indices] = old_positions[molecule.atom_indices] + translation
                nscale = state.system.nmolecules # number of molecules, used in proposal probability

            # Compute log proposal probability
            log_proposal_ratio = float(nscale) * math.log(volume / old_volume)

            # Evaluate energy change.
            reduced_potential = state.reduced_potential(positions)

            # Compute log probability change.
            Delta_logP = - (reduced_potential - old_reduced_potential) + log_proposal_ratio

            # Accept or reject.
            nattempted += 1
            if (Delta_logP >= 0.0) or (numpy.random.rand() < math.exp(Delta_logP)):
                # Accept.
                naccepted += 1

                # Update energy
                reduced_potential_old = reduced_potential

                if self.verbose: print "rejected"
            else:
                # Reject.

                # Restore box vectors.
                state.system.setPeriodicBoxVectors(a, b, c)

                # Restore positions.
                for iatom in range(state.system.getNumParticles()):
                    positions[i] = old_positions[i]

                if self.verbose: print "accepted"

        return new_positions

#=============================================================================================
# Constant pH
#=============================================================================================

class ProtonationStateMove(MarkovChainMonteCarloMove):
    """
    Protonation state move for constant-pH simulation.

    This move type implements the constant-pH dynamics of Mongan and Case [1].

    REFERENCES

    [1] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born
    implicit solvent. J Comput Chem 25:2038, 2004.

    TODO

    * Should we generalize this to multiple tautomers too?
    * Should we have separate classes for protein constant pH an ligand pH/tautomer?
    

    """

    def __init__(self, reference_system, metadata, titratable_groups):
        """
        Initialize a constant-pH move.

        ARGUMENTS

        reference_system
        metadata - information about atom names, residues, and connectivity
        titratable_groups - dictionary of titratable groups, their charges, and reference pKas
        
        """

        # Set defaults.
        self.nattempts = 10 # number of protonation state change attempts per call
        self.double_proposal_probability = 0.1 # probability of proposing two simultaneous protonation state changes
        
        return

    def select_protonation_state(self, titratable_group_index, titration_state_index, state):
        """
        Change the titration state of the designated group for the provided state.

        """

        # Here, we modify the charges for the atoms in state.system that correspond to the designated titratable_group.

        atom_indices
        new_charges

        for index in atom_indices:
            state.system.charges[index] = new_charges[index]

        # Add a @property decorator to System that allows us to index the charges on all atoms in the system?
        # This would have to appropriately modify both NonbondedForce and OBCGBSAForce force entries, as well as special exclusions which depend on whether system is being decoupled or annihilated.
        # Would it be easier to regenerate from modified AMBER prmtop data representation at this point?  The problem is that this strategy would not work with other forcefields.    
            
        return

    def run(self, state, positions):
        """
        Apply move to the specified system and coordinates.

        ARGUMENTS

        state (ThermodynamicState) - thermodynamic state
        positions (Quantity of numpy array) - current positions
        
        RETURNS

        new_positions

        """

        # Make several attempts to change protonation state.
        for attempt in range(self.nattempts):
            
        # Each attempt requires we change the *charges* in the system    
        # Use state.reduced_potential() to compute the reduced potential after a modification attempt.        

        return new_positions

    # Also provide utility methods to help compute reference free energies and protonated/deprotonated charges for new molecules.
    # Would provide OEMol entries for each protonation state with AM1-BCC charges and known pKas or tautomer ratios, and we would compute free energies of each state.
    
#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
