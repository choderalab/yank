#!/bin/tcsh

# Parameterize benzene from Tripos mol2.
antechamber -fi mol2 -i benzene.mol2 -fo mol2 -o benzene.mol2
parmchk -i benzene.gaff.mol2 -o benzene.frcmod -f mol2

# Parameterize toluene from Tripos mol2.
antechamber -fi mol2 -i toluene.mol2 -fo mol2 -o toluene.mol2
parmchk -i toluene.gaff.mol2 -o toluene.frcmod -f mol2

# Create benzene-toluene system.
rm -f leap.log receptor.{crd,prmtop,pdb} ligand.{crd,prmtop,pdb}
tleap -f setup.leap.in

# Create PDB files.
cat receptor.crd | ambpdb -p receptor.prmtop > receptor.pdb
cat ligand.crd | ambpdb -p ligand.prmtop > ligand.pdb
cat complex.crd | ambpdb -p complex.prmtop > complex.pdb

