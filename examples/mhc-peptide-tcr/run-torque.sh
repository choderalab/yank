#!/bin/bash
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=12:00:00
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
# nodes: number of nodes
#   ppn: how many cores per node to use
#PBS -l nodes=7:ppn=4:gpus=4:shared
##PBS -l procs=16,gpus=1:shared
##PBS -l procs=8,gpus=1:shared
#
# memory requirements
#PBS -l mem=8GB
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N mhc-peptide-tcr

if [ -n "$PBS_O_WORKDIR" ]; then
    cd $PBS_O_WORKDIR
fi

# Record GPU file contents
cat $PBS_GPUFILE

# Set up and run simulation in serial mode.

if [ ! -e output ]; then
    echo "Making output directory..."
    mkdir output
fi

# Clean up any leftover files
echo "Cleaning up previous simulation..."
yank cleanup --store=output

# Set up calculation.
echo "Setting up binding free energy calculation..."
python mhc-peptide-tcr.py

# Make sure files are synced.
sync
sleep 5
sync

# Run the simulation with verbose output:
echo "Running simulation via MPI..."
#build_mpirun_configfile --mpitype=conda "yank run --store=output --verbose --mpi --platform CUDA --precision=double"
build_mpirun_configfile --mpitype=conda "yank run --store=output --verbose --mpi"

# Make sure files are synced.
sync
sleep 5
sync

# Note configfile contents
cat configfile
echo ""

mpirun -configfile configfile
date


