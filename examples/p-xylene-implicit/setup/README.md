# Set up p-xylene binding to T4 lysozyme L99A using AmberTools

## Manifest
* `setup.tcsh` - shell script to run whole setup pipeline to regenerate YANK input files
     Requires OpenEye Python toolkits, pdbfixer, and AmberTools
* `generate-mol2-from-name.py` - Python script to generate ligand Tripos mol2 file from IUPAC or common name
     Requires OpenEye Python toolkits.
* `setup.leap.in` - input file for AmberTools tleap
* `181L.pdb` - source PDB file for benzene bound to T4 lysozyme L99A
* `187L.pdb` - source PDB file for p-xylene bound to T4 lysozyme L99A
* `p-xylene.tripos.mol2` - p-xylene in Tripos mol2 format
* `p-xylene.gaff.mol2` - p-xylene in GAFF mol2 format
* `p-xylene.gaff.frcmod` - AMBER frcmod file for p-xylene GAFF parameters
* `ligand.{inpcrd,prmtop,pdb}` - ligand AMBER inpcrd/prmtop/pdb files 
* `receptor.{inpcrd,prmtop,pdb}` - receptor  AMBER inpcrd/prmtop/pdb files
* `complex.{inpcrd,prmtop,pdb}` - complex AMBER inpcrd/prmtop/pdb files
* `run.tcsh` - shell script to run YANK simulation

