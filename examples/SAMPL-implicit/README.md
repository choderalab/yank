# SAMPL guest molecules binding to CB7 in implicit solvent

## Description

The ligand is parameterized using the GAFF forcefield with the AM1-BCC charge model.

The receptor, CB7, is also parameterized using GAFF and AM1-BCC with a maximum path length of 15.

The mbondi2 radii are used, with OBC GBSA in YANK.


## Usage

### Set up the system from scratch (not necessary unless source PDB or ligand files are modified):
```tcsh
./setup.tcsh
```

## Running the simulation.

Set up the simulation to alchemically decouple benzene, putting all the output files in `output/`:
```tcsh
yank setup binding amber --setupdir=setup --ligname=MOL --store=output --iterations=1000 --restraints=harmonic --gbsa=OBCs --temperature=300*kelvin --minimize --verbose
```

Run the simulation with verbose output:
```tcsh
yank run --store=output --verbose
```

Clean up and delete all simulation files:
```tcsh
yank cleanup --store=output
```
## Citation

The compounds are from the following paper:

"Absolute and relative binding affinity of cucurbit[7]uril towards a series of cationic guests"
Liping Cao and Lyle Isaacs, 2013
