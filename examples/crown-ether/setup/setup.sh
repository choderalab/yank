#!/bin/bash

# Parameterize 18-crown-6 from Tripos mol2.
antechamber -pf yes -fi mol2 -i 18-crown-6.tripos.mol2 -fo mol2 -o 18-crown-6.gaff.mol2
parmchk -i 18-crown-6.gaff.mol2 -o 18-crown-6.frcmod -f mol2

# Parameterize phenol from Tripos mol2.
antechamber -pf yes -fi mol2 -i phenol.tripos.mol2 -fo mol2 -o phenol.gaff.mol2
parmchk -i phenol.gaff.mol2 -o phenol.frcmod -f mol2

# Create 18-crown-6:phenol system.
rm -f leap.log receptor.{crd,prmtop,pdb} ligand.{crd,prmtop,pdb}
tleap -f setup.leap.in
