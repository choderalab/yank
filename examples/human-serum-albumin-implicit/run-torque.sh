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
#PBS -l nodes=2:ppn=4:gpus=4:shared
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N hsa-aspirin-implicit

cd $PBS_O_WORKDIR

# Set defaults
export NITERATIONS=${NITERATIONS:=1000}
export NEQUILITERATIONS=${NEQUILITERATIONS:=0}

if [ ! -e output ]; then
    echo "Making output directory..."
    mkdir output
fi

# Clean up any leftover files
echo "Cleaning up previous simulation..."
yank cleanup --store=output

# Set up calculation.
echo "Setting up binding free energy calculation..."
yank prepare binding amber --setupdir=setup --ligand="resname MOL" --store=output --iterations=$NITERATIONS --restraints=Harmonic --gbsa=OBC2 --temperature="300*kelvin" --minimize --equilibrate=$NEQUILITERATIONS --verbose

# Run the simulation with verbose output:
echo "Running simulation via MPI..."
build_mpirun_configfile  --mpitype=conda "yank run --store=output --verbose --mpi"
mpirun -configfile configfile

# Analyze the data
echo "Analyzing data..."
yank analyze --store=output
