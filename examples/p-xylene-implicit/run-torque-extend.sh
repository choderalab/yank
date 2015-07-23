#!/bin/bash
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=24:00:00
#
# join stdout and stderr
#PBS -j oe
#
# spool output immediately
#PBS -k oe
#
# specify queue
#PBS -q gpu
#
# nodes: number of 8-core nodes
#   ppn: how many cores per node to use (1 through 8)
#       (you are always charged for the entire node)
#PBS -l nodes=1:ppn=4:gpus=4:shared
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N p-xylene-implicit

cd $PBS_O_WORKDIR

# Run the simulation with verbose output:
echo "Running simulation via MPI..."
build_mpirun_configfile  --mpitype=conda "yank run --store=output --verbose --phase=complex-implicit --iterations=5000 --mpi"
mpirun -configfile configfile

# Analyze the data
echo "Analyzing data..."
yank analyze --store=output
