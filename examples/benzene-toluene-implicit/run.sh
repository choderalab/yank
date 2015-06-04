#!/bin/bash

#
# Benzene-toluene example run script (serial mode)
#

# Set defaults
export NITERATIONS=${NITERATIONS:=1000}

# Create output directory.
if [ ! -e output ]; then
    echo "Making output directory..."
    mkdir output
fi

# Clean up any leftover files
echo "Cleaning up previous simulation..."
yank cleanup --store=output

# Set up calculation.
echo "Setting up binding free energy calculation..."
yank prepare binding amber --setupdir=setup --ligname=BEN --store=output --iterations=$NITERATIONS --restraints=harmonic --gbsa=OBC2 --temperature="300*kelvin" --verbose

# Run the simulation with verbose output:
echo "Running simulation..."
yank run --store=output --verbose

