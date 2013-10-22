#!/usr/local/bin/env python

# Test context creation in parallel.

import simtk.openmm as openmm
import simtk.unit as units

# Initialize MPI
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# Set platform to GPU id.
gpu_platform_name = 'CUDA'
platform = openmm.Platform.getPlatformByName(gpu_platform_name)
deviceid = rank
cuda_temp_directory = 'CUDA%d' % deviceid
platform.setPropertyDefaultValue('OpenCLDeviceIndex', '%d' % deviceid) # select OpenCL device index
platform.setPropertyDefaultValue('CudaDeviceIndex', '%d' % deviceid) # select Cuda device index
platform.setPropertyDefaultValue('CudaTempDirectory', cuda_temp_directory) # select Cuda device index
print "rank %d/%d platform %s deviceid %d" % (rank, size, gpu_platform_name, deviceid)

# Make CUDA temp directories.
import os, os.path
if not os.path.exists(cuda_temp_directory):
    os.makedirs(cuda_temp_directory)

# Create test system.
import testsystems
[system, positions] = testsystems.LennardJonesFluid()

# Create Context.
import time
temperature = 300.0 * units.kelvin
collision_rate = 9.1 / units.picoseconds
timestep = 1.0 * units.femtoseconds
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
print "rank %d/%d creating context..." % (rank, size)
initial_time = time.time()
context = openmm.Context(system, integrator, platform)
print "rank %d/%d context created in %.3f s" % (rank, size, time.time() - initial_time)
MPI.COMM_WORLD.barrier()

# Run dynamics.
nsteps = 500
niterations = 1000
context.setPositions(positions)
for iteration in range(niterations):
    integrator.step(nsteps)
    state = context.getState(getEnergy=True)
    print "rank %d/%d : iteration %5d | potential %12.3f kcal/mol" % (rank, size, iteration, state.getPotentialEnergy() / units.kilocalories_per_mole)
    MPI.COMM_WORLD.barrier()




