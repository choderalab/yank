from repex import testsystems
from yank import oldrepex
import simtk.unit as units
import tempfile

testsystem = testsystems.AlanineDipeptideImplicit()
[reference_system, positions] = [testsystem.system, testsystem.positions]
# Copy reference system.
systems = [reference_system for index in range(10)]
# Create temporary file for storing output.



file = tempfile.NamedTemporaryFile() # temporary file for testing
store_filename = file.name
# Create reference state.
reference_state = oldrepex.ThermodynamicState(reference_system, temperature=298.0*units.kelvin)
# Create simulation.
simulation = oldrepex.HamiltonianExchange(reference_state, systems, positions, store_filename)
simulation.number_of_iterations = 2 # set the simulation to only run 2 iterations
simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
simulation.nsteps_per_iteration = 50 # run 50 timesteps per iteration
simulation.minimize = False
# Run simulation.
simulation.run() # run the simulation
