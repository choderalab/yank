# TIP3P water in water using reaction field example.

## Description


## Usage

### Set up the system from scratch (not necessary unless source PDB or ligand files are modified):
```tcsh
./setup.tcsh
```

## Running the simulation.

Set up the simulation to alchemically decouple water, putting all the output files in `output/`:
```tcsh
yank setup binding amber --setupdir=setup --ligname=LIG --store=output --iterations=1000 --nbmethod=CutoffPeriodic --temperature="300*kelvin" --pressure="1*atmosphere" --minimize --verbose
```

Run the simulation with verbose output:
```tcsh
yank run --store=output --verbose
```

Clean up and delete all simulation files:
```tcsh
yank cleanup --store=output
```

