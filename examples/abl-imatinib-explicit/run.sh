#!/bin/bash

#
# Abl binding to imatinib in explicit solvent.
#

# Run the simulation with verbose output:
echo "Running simulation..."
yank script --yaml=yank.yaml

# Analyze the data
echo "Analyzing data..."
yank analyze --store=experiments
