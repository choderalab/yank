#!/bin/bash

# Run the simulation with verbose output:
echo "Running simulation..."
build_mpirun_configfile "yank script --yaml=yank.yaml"
mpirun -configfile configfile
date



