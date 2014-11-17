#!/bin/bash

# Create AMBER prmtop/inpcrd files.
echo "Creating AMBER prmtop/inpcrd files..."
rm -f leap.log {vacuum,solvent,complex}.{inpcrd,prmtop,pdb}
tleap -f setup.leap.in > setup.leap.out


