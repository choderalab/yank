#!/bin/bash

#
# HSA binding of kinase inhibitors with YANK
#

# Run the simulation with verbose output:
echo "Running simulation..."
yank script --yaml=yank.yaml

# Analyze the data
echo "Analyzing data..."
yank analyze --store=experiments
