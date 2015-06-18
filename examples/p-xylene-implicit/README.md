# p-xylene binding to T4 lysozyme L99A in implicit solvent example

## Description

The ligand is parameterized using the GAFF forcefield with the AM1-BCC charge model.

The receptor (taken from 181L.pdb) is parameterized with the AMBER parm99sb-ildn forcefield.

The mbondi2 radii are used, with OBC GBSA in YANK.

## Usage

### Set up the system from scratch (not necessary unless source PDB or ligand files are modified):
```tcsh
./setup.tcsh
```

## Running the simulation.

Set up the simulation to alchemically decouple benzene, putting all the output files in `output/`:
```tcsh
yank prepare binding amber --setupdir=setup --ligand="resname MOL" --store=output --iterations=1000 --restraints=harmonic --gbsa=OBC2 --temperature="300*kelvin" --minimize --verbose
```

Run the simulation with verbose output:
```tcsh
yank run --store=output --verbose
```

Clean up and delete all simulation files:
```tcsh
yank cleanup --store=output
```

