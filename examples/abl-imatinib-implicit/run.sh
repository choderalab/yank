#!/bin/bash

#
# Abl binding to imatinib in implicit solvent.
#

# Set defaults
export NITERATIONS=${NITERATIONS:=1000}

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
yank prepare binding amber --setupdir=setup --ligand="resname MOL" --store=output --iterations=$NITERATIONS --restraints=harmonic --gbsa=OBC2 --temperature=300*kelvin --minimize --verbose

# Run the simulation with verbose output:
echo "Running simulation..."
yank run --store=output --verbose

