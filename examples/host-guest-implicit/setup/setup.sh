#!/bin/bash

#set up the ligand molecule:
echo "Generating Tripos mol2 file of ligand from IUPAC or common name..."
rm -f ligand.tripos.mol2
python generate-mol2-from-name.py --name p-xylene --outfile ligand.tripos.mol2

#prepare the receptor for simulation
echo "Parameterizing receptor molecule using antechamber"
#antechamber -fi mol2 -i receptor.tripos.mol2 -fo mol2 -o receptor.gaff.mol2 -c bcc -pl 15
parmchk -i receptor.gaff.mol2 -o receptor.gaff.frcmod -f mol2

#parameterize the ligand
echo "Parameterizing the ligand molecule using antechamber"
antechamber -fi mol2 -i ligand.tripos.mol2 -fo mol2 -o ligand.gaff.mol2 -c bcc
parmchk -i ligand.gaff.mol2 -o ligand.gaff.frcmod -f mol2

#create AMBER inputs with tleap
echo "Creating AMBER inputs with tleap"
rm -f leap.log receptor.{inpcrd,prmtop,pdb} ligand.{inpcrd,prmtop,pdb} complex.{inpcrd,prmtop,pdb}
tleap -f setup.leap.in > setup.leap.out
