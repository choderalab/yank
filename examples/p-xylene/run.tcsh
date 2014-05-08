#!/bin/tcsh

# Run in serial mode.
#python ../../yank/yank.py --receptor_prmtop receptor.prmtop --ligand_prmtop ligand.prmtop --complex_prmtop complex.prmtop --complex_crd complex.inpcrd --restraints harmonic --randomize_ligand --iterations 5 --verbose --platform CUDA

# Run in MPI mode.
mpirun -rmk pbs python ../../yank/yank.py --receptor_prmtop receptor.prmtop --ligand_prmtop ligand.prmtop --complex_prmtop complex.prmtop --complex_crd complex.inpcrd --restraints harmonic --randomize_ligand --iterations 500 --verbose --mpi --platform OpenCL --gpus_per_node 4

