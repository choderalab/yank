#!/bin/bash
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=72:00:00
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
#PBS -l nodes=2:ppn=4:gpus=4
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N freesolv-validation

cd $PBS_O_WORKDIR

# Run the simulation
echo "Running simulation via MPI..."
build_mpirun_configfile "yank script --yaml=freesolv.yaml"
mpirun -configfile configfile
