#!/bin/bash

#
# MHC-peptide-TCR complex
# Using Yank Python API
#

# Set defaults
export NITERATIONS=${NITERATIONS:=100}

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
python mhc-peptide-tcr.py

# Run the simulation with verbose output:
echo "Running simulation..."
yank run --store=output --verbose

# Analyze the data
echo "Analyzing data..."
yank analyze --store=output

