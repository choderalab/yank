#!/bin/tcsh

# Generate ligand Tripos mol2 file from IUPAC or common name.
python generate-mol2-from-name.py --name p-xylene --outfile ligand.tripos.mol2

# Copy protein PDB file.
cp 181L.pdb receptor-source.pdb

# Parameterize ligand from Tripos mol2.
antechamber -fi mol2 -i ligand.tripos.mol2 -fo mol2 -o ligand.gaff.mol2
parmchk -i ligand.gaff.mol2 -o ligand.frcmod -f mol2

# Create AMBER prmtop/inpcrd files.
rm -f leap.log receptor.{inpcrd,prmtop,pdb} ligand.{inpcrd,prmtop,pdb} complex.{inpcrd,prmtop,pdb}
tleap -f setup.leap.in

# Create PDB files.
cat receptor.inpcrd | ambpdb -p receptor.prmtop > receptor.pdb
cat ligand.inpcrd | ambpdb -p ligand.prmtop > ligand.pdb
cat complex.inpcrd | ambpdb -p complex.prmtop > complex.pdb

