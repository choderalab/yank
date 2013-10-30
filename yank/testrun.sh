#!/bin/tcsh

canopy
openmm-git

# T4 lysozyme L99A test
setenv JOBDIR "examples/p-xylene"

# Simple benzene-toluene test.
#setenv JOBDIR "examples/benzene-toluene"

# Clean up data files.
#rm -f ${JOBDIR}/*.nc

# Clean out cache.
#rm -rf ~/.nv
#sync

setenv | grep TMPDIR

# SERIAL
#python yank.py --complex_prmtop $JOBDIR/complex.prmtop --receptor_prmtop $JOBDIR/receptor.prmtop --ligand_prmtop $JOBDIR/ligand.prmtop --complex_crd $JOBDIR/complex.crd --output $JOBDIR --verbose --iterations 10000 --randomize_ligand

# PARALLEL
mpirun -np 4 python yank.py --mpi --complex_prmtop $JOBDIR/complex.prmtop --receptor_prmtop $JOBDIR/receptor.prmtop --ligand_prmtop $JOBDIR/ligand.prmtop --complex_crd $JOBDIR/complex.crd --output $JOBDIR --verbose --iterations 10000 --randomize_ligand

