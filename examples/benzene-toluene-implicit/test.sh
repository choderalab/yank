#!/bin/bash

if [ ! -e output ]; then
    echo "Making output directory..."
    mkdir output
fi

# Clean up any leftover files
echo "Cleaning up previous simulation..."
yank cleanup --store=output

# Set up calculation.
echo "Setting up binding free energy calculation..."
yank prepare binding amber --ligname=BEN --store=output --iterations=1 --restraints=harmonic --gbsa=OBC2 --temperature=300*kelvin --verbose --iterations=10

# Run the simulation with verbose output:
echo "Running simulation..."
yank run --store=output --verbose --online-analysis

