#!/bin/tcsh
#
# Simple YANK test script to be run from yank/yank/ directory (with examples/ subdirectory present).

# Simple benzene-toluene test.
#setenv JOBDIR "examples/benzene-toluene"

# More expensive T4 lysozyme L99A test
# (uncomment this to override other test)
setenv JOBDIR "examples/p-xylene"

# Clean up data files before run.
rm -f ${JOBDIR}/*.nc

# SERIAL EXECUTION
python yank.py --complex_prmtop $JOBDIR/complex.prmtop --receptor_prmtop $JOBDIR/receptor.prmtop --ligand_prmtop $JOBDIR/ligand.prmtop --complex_crd $JOBDIR/complex.crd --output $JOBDIR --verbose --iterations 40 --randomize_ligand

# PARALLEL EXECUTION (requires mpi4py and MPI)
#mpirun -np 4 python yank.py --mpi --complex_prmtop $JOBDIR/complex.prmtop --receptor_prmtop $JOBDIR/receptor.prmtop --ligand_prmtop $JOBDIR/ligand.prmtop --complex_crd $JOBDIR/complex.crd --output $JOBDIR --verbose --iterations 40 --randomize_ligand

