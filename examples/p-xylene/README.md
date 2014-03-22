# p-xylene binding to T4 lysozyme L99A in implicit solvent example

## Description

The ligand is parameterized using the GAFF forcefield with the AM1-BCC charge model.

The receptor (taken from 181L.pdb) is parameterized with the AMBER parm99sb-ildn forcefield.

The mbondi2 radii are used, with OBC GBSA in YANK.

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

## Requirements

### Setting up a new ligand or protein
* OpenEye Python toolkits: http://www.eyesopen.com/toolkits
* `pdbfixer`: https://github.com/peastman/pdbfixer
* AmberTools: http://ambermd.org/AmberTools-get.html

### Running a simulation
* YANK and dependencies
* `mpi4py` (if parallel execution is desired): http://mpi4py.scipy.org/

## Usage

### Set up the system from scratch (not necessary unless source PDB or ligand files are modified):
```tcsh
./setup.tcsh
```

### Run simulation
```tcsh
python ../../yank/yank.py --receptor_prmtop receptor.prmtop --ligand_prmtop ligand.prmtop --complex_prmtop complex.prmtop --complex_crd complex.crd --restraints flat-bottom --randomize_ligand --iterations 1000 --verbose 
```

