#!/bin/bash

# Create AMBER prmtop/inpcrd files.
echo "Creating AMBER prmtop/inpcrd files..."
rm -f leap.log receptor.{inpcrd,prmtop,pdb} ligand.{inpcrd,prmtop,pdb} complex.{inpcrd,prmtop,pdb}
tleap -f setup.leap.in > setup.leap.out

