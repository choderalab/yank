#!/bin/bash

#
# Association free energy of sodium and chloride ions example run script (serial mode)
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
yank prepare binding amber --setupdir=setup --ligname=Na+ --store=output --iterations=$NITERATIONS --nbmethod=CutoffPeriodic --temperature=300*kelvin --pressure=1*atmospheres --verbose

# Run the simulation with verbose output:
echo "Running simulation..."
yank run --store=output --verbose

