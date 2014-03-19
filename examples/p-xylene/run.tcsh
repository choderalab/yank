#!/bin/tcsh

# Run in serial mode.
python yank.py --serial --complex_prmtop complex.prmtop --receptor_prmtop receptor.prmtop --ligand_prmtop ligand.prmtop --complex_crd complex.crd --output . --verbose --iterations 10000 --randomize_ligand

# Run in MPI mode using 4 GPUs and 2 CPUs.
#mpirun -np 6 python yank.py --mpi --complex_prmtop complex.prmtop --receptor_prmtop receptor.prmtop --ligand_prmtop ligand.prmtop --complex_crd complex.crd --output . --verbose --iterations 10000 --randomize_ligand --ngpus 4 --ncpus 2

