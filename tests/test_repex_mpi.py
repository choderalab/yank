# Parallel tempering simulation of alanine dipeptide in implicit solvent (replica exchange among temperatures.
# (This is just an illustrative example; use ParallelTempering class for actual production parallel tempering simulations.)

import simtk.openmm
import numpy
import sys
import os

import yank
from yank.oldrepex import ThermodynamicState, ReplicaExchange

# Create test system.
print "Creating test systems..."
import testsystems
[system, coordinates] = testsystems.AlanineDipeptideImplicit()
# Create thermodynamic states for parallel tempering with exponentially-spaced schedule.
import simtk.unit as units
import math
nreplicas = 6 # number of temperature replicas
T_min = 298.0 * units.kelvin # minimum temperature
T_max = 600.0 * units.kelvin # maximum temperature
T_i = [ T_min + (T_max - T_min) * (math.exp(float(i) / float(nreplicas-1)) - 1.0) / (math.e - 1.0) for i in range(nreplicas) ]
states = [ ThermodynamicState(system=system, temperature=T_i[i]) for i in range(nreplicas) ]
import tempfile
file = tempfile.NamedTemporaryFile() # use a temporary file for testing -- you will want to keep this file, since it stores output and checkpoint data

# Select platform: one of 'Reference' (CPU-only), 'Cuda' (NVIDIA Cuda), or 'OpenCL' (for OS X 10.6 with OpenCL OpenMM compiled)
platform = simtk.openmm.Platform.getPlatformByName("OpenCL")    
platform = simtk.openmm.Platform.getPlatformByName("Cuda")    

# Set up device to bind to.
print "Selecting MPI communicator and selecting a GPU device..."
from mpi4py import MPI # MPI wrapper
hostname = os.uname()[1]
ngpus = 6 # number of GPUs per system
comm = MPI.COMM_WORLD # MPI communicator
deviceid = comm.rank % ngpus # select a unique GPU for this node assuming block allocation (not round-robin)
platform.setPropertyDefaultValue('CudaDeviceIndex', '%d' % deviceid) # select Cuda device index
platform.setPropertyDefaultValue('OpenCLDeviceIndex', '%d' % deviceid) # select OpenCL device index
print "node '%s' deviceid %d / %d, MPI rank %d / %d" % (hostname, deviceid, ngpus, comm.rank, comm.size)
# Make sure random number generators have unique seeds.
seed = numpy.random.randint(sys.maxint - comm.size) + comm.rank
numpy.random.seed(seed)

# Set up replica exchange simulation.
simulation = ReplicaExchange(states, coordinates, file.name, mpicomm=comm) # initialize the replica-exchange simulation
simulation.verbose = True
simulation.number_of_iterations = 100 # set the simulation to only run 2 iterations
simulation.timestep = 2.0 * units.femtoseconds # set the timestep for integration
simulation.nsteps_per_iteration = 500 # run 500 timesteps per iteration
simulation.platform = platform
simulation.run() # run the simulation

