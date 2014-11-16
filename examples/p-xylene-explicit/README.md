# p-xylene binding to T4 lysozyme L99A in TIP3P solvent with reaction-field electrostatics example

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
yank setup binding amber --setupdir=setup --ligname=BEN --store=output --iterations=1000 --nbmethod=CutoffPeriodic --temperature="300*kelvin" --pressure="1*atmosphere" --minimize --verbose
```

Run the simulation with verbose output:
```tcsh
yank run --store=output --verbose
```

Clean up and delete all simulation files:
```tcsh
yank cleanup --store=output
```

