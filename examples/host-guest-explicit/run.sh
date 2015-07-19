#!/bin/bash

#
# host-guest association in explicit solvent
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
yank prepare binding amber --setupdir=setup --ligand="resname MOL" --store=output --iterations=$NITERATIONS --nbmethod=CutoffPeriodic --temperature="300*kelvin" --pressure="1*atmosphere" --minimize --verbose

# Run the simulation with verbose output:
echo "Running simulation..."
yank run --store=output --verbose

# Analyze the data
echo "Analyzing data..."
yank analyze --store=output
