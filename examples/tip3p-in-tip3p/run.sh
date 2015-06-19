#!/bin/bash

#
# TIP3P hydration free energy example run script (serial mode)
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
yank prepare binding amber --setupdir=setup --ligand="resname LIG" --store=output --iterations=$NITERATIONS --nbmethod=CutoffPeriodic --temperature="300*kelvin" --pressure="1*atmosphere" --verbose

# Run the simulation with verbose output:
echo "Running simulation..."
yank run --store=output --verbose

