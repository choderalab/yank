#!/bin/bash

#
# TIP3P hydration free energy example run script (serial mode)
#

# Run the simulation
echo "Running simulation..."
yank script --yaml=yank.yaml

# Analyze the data
echo "Analyzing data..."
yank analyze --store=experiments
