#!/bin/bash

# Parameterize benzene from Tripos mol2.
antechamber -fi mol2 -i benzene.tripos.mol2 -fo mol2 -o benzene.gaff.mol2
parmchk -i benzene.gaff.mol2 -o benzene.frcmod -f mol2

# Parameterize toluene from Tripos mol2.
antechamber -fi mol2 -i toluene.tripos.mol2 -fo mol2 -o toluene.gaff.mol2
parmchk -i toluene.gaff.mol2 -o toluene.frcmod -f mol2

# Create benzene-toluene system.
rm -f leap.log receptor.{crd,prmtop,pdb} ligand.{crd,prmtop,pdb}
tleap -f setup.leap.in
