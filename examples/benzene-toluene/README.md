# benzene-toluene association in implicit solvent

## Running the simulation.

```tcsh
rm *.nc ; python ../../yank/yank.py --receptor_prmtop receptor.prmtop --ligand_prmtop ligand.prmtop --complex_prmtop complex.prmtop --complex_crd complex.crd --restraints flat-bottom --randomize_ligand --iterations 2 --verbose
```
