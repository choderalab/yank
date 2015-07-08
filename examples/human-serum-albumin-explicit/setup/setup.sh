#!/bin/bash

# Generate ligand Tripos mol2 file from IUPAC or common name.
# NOTE: Requires OpenEye toolkit be installed.
echo "Generating Tripos mol2 file of ligand from IUPAC or common name..."
rm -f ligand.tripos.mol2
python generate-mol2-from-name.py --name aspirin --outfile ligand.tripos.mol2

# Prepare receptor by filling in missing atoms and expunging waters and ions.
# NOTE: Requires pdbfixer tool be installed and PDBFIXER_HOME environment variable set.
echo "Preparing receptor by adding missing atoms..."
rm -f receptor.pdbfixer.pdb
#pdbfixer 2I30.pdb --add-residues --keep-heterogens=none --add-atoms=heavy --ph=7.4 --replace-nonstandard --output=receptor.pdbfixer.pdb # something is wrong
pdbfixer 2I2Z.pdb --keep-heterogens=none --add-atoms=heavy --ph=7.4 --replace-nonstandard --output=receptor.pdbfixer.pdb

# Change all CYS to CYX
sed 's/CYS/CYX/' < receptor.pdbfixer.pdb > receptor.pdbfixer.CYX.pdb

# Parameterize ligand from Tripos mol2.
echo "Parameterizing ligand with GAFF and AM1-BCC charges..."
antechamber -fi mol2 -i ligand.tripos.mol2 -fo mol2 -o ligand.gaff.mol2 -c bcc
parmchk2 -i ligand.gaff.mol2 -o ligand.gaff.frcmod -f mol2

# Create AMBER prmtop/inpcrd files.
echo "Creating AMBER prmtop/inpcrd files..."
rm -f leap.log receptor.{inpcrd,prmtop,pdb} ligand.{inpcrd,prmtop,pdb} complex.{inpcrd,prmtop,pdb}
tleap -f setup.leap.in > setup.leap.out
