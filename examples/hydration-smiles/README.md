# Hydration free energies of small molecules from SMILES

**This example requires the OpenEye toolkit installed** to generate the molecules from SMILES strings.

## Description
Hydration free energy calculations in TIP3P water box at 300K and 1atm of a subset of 12 molecules in the [FreeSolv database](https://github.com/choderalab/FreeSolv).

## Running the simulation.
On the console, `cd` into this folder and type:
```tcsh
yank script --yaml=freesolv.yaml
```

## Running only few molecules in the database
Modify the YAML script and substitute the line
```yaml
        select: all
```
with
```yaml
        select: 0
```
to run the molecule described in the first line of the database, or
```yaml
        select: !Combinatorial [0, 5]
```
to run the molecules in the 1st and 6th lines of the database. Then run the script as before
```tcsh
yank script --yaml=freesolv.yaml
```
