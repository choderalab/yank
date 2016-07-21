#!/bin/bash

#
# p-xylene binding to T4 lysozyme L99A example run script (serial mode)
#

# Run the simulation with verbose output:
echo "Running simulation..."
yank script --yaml=yank.yaml

# Analyze the data
echo "Analyzing data..."
yank analyze --store=experiments
