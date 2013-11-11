#!/usr/local/bin/env python

# Submit jobs to cluster using slurm.

import os
import os.path
import commands

pbs_template = """\
#!/bin/tcsh
#  Batch script for mpirun job on cbio cluster.
#
#SBATCH --job-name=%(jobname)s
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --output=%(jobname)s-%%j.stdout
#SBATCH --error=%(jobname)s-%%j.stderr
#SBATCH --gres=gpu:4
##SBATCH --constraint=gtx680

setenv | grep SLURM

canopy
openmm

date

setenv YANKHOME /cbio/jclab/home/chodera/yank

# Select job directory.
setenv JOBDIR ${YANKHOME}/test-systems/T4-lysozyme-L99A/amber-gbsa/amber-gbsa/%(molecule)s

# Set YANK directory.
setenv YANKDIR ${YANKHOME}/src/

date

# Change to job directory
cd $JOBDIR

# Clean up old working files
rm -f *.nc

# Set PYTHONPATH
setenv PYTHONPATH ${YANKDIR}:${PYTHONPATH}

# Run YANK
mpirun -bootstrap slurm python $YANKDIR/yank.py --mpi --receptor_prmtop receptor.prmtop --ligand_prmtop ligand.prmtop --complex_prmtop complex.prmtop --complex_crd complex.crd --output . --restraints flat-bottom --iterations 10000 --verbose

date
"""

def isletter(c):
   if ((c >= 'a') and (c <= 'z')) or ((c >= 'A') and (c <= 'Z')):
      return True
   return False   

# T4 lysozyme L99A
molecules = ['1-methylpyrrole', '2-fluorobenzaldehyde', 'benzene', 'ethylbenzene', 'indole', 'n-butylbenzene', 'n-propylbenzene', 'p-xylene', 'thieno_23c_pyridine', '12-dichlorobenzene', '23-benzofuran', 'benzenedithiol', 'indene', 'isobutylbenzene', 'n-methylaniline', 'o-xylene', 'phenol', 'toluene']
#molecules = ['p-xylene', 'benzene', 'phenol', 'indole', '12-dichlorobenzene']
#molecules = ['p-xylene', 'benzene', 'phenol', '12-dichlorobenzene', '1-methylpyrrole']
#molecules = ['p-xylene']

# FKBP
#molecules = ['L12', 'L14', 'L20', 'LG2', 'LG3', 'LG5', 'LG6', 'LG8', 'LG9']

# Chk1 kinase
#molecules = ['molec000001', 'molec000002', 'molec000003']

# Kim's congeneric series
#molecules += ['molec000023', 'molec000034', 'molec000073', 'molec000089', 'molec000096']

for molecule in molecules:
   print molecule

   # Make sure job name begins with a letter
   jobname = molecule
   
   # Form PBS script
   pbs = pbs_template % vars()
   #print pbs

   # Construct directory.
   filename = '../test-systems/T4-lysozyme-L99A/amber-gbsa/amber-gbsa/%(molecule)s/run.pbs' % vars()
   outfile = open(filename, 'w')
   outfile.write(pbs)
   outfile.close()

   # Submit to PBS
   output = commands.getoutput('qsub %(filename)s' % vars());
   print output
