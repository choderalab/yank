# p-xylene binding to T4 lysozyme L99A in implicit solvent example

## Description

The ligand is parameterized using the GAFF forcefield with the AM1-BCC charge model.

The receptor (taken from 181L.pdb) is parameterized with the AMBER parm99sb-ildn forcefield.

The mbondi2 radii are used, with OBC GBSA in YANK.

## Usage

Automatically set up the system and run the calculation
```bash
yank script --yaml=yank.yaml
```

Analyze the data
```bash
yank analyze --store=experiments
```

