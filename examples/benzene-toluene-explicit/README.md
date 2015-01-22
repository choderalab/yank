# benzene-toluene association in explicit solvent

This example computes the binding affinity of benzene with toluene in TIP3P water.

A harmonic restraint is used to keep the two molecules from drifting far away from each other.

## Running the simulation.

Set up the simulation to alchemically decouple benzene, putting all the output files in `output/`:
```tcsh
yank prepare binding amber --setupdir=setup --ligname=BEN --store=output --iterations=1000 --restraints=harmonic --temperature=300*kelvin --pressure=1*atmospheres --minimize --verbose
```

Run the simulation with verbose output:
```tcsh
yank run --store=output --verbose
```

Clean up and delete all simulation files:
```tcsh
yank cleanup --store=output
```

