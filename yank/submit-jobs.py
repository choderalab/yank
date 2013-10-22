#!/usr/local/bin/env python

# Submit jobs to cluster

import os
import os.path
import commands

pbs_template = """\
#!/bin/tcsh
#  Sample Batch Script for MPI job on Torque.
#
#
#
#  Submit this script using the command: qsub <script_name>
#
#  Use the "qstat" command to check the status of a job.
#
# The following are embedded QSUB options. The syntax is #PBS (the # does
# _not_  denote that the lines are commented out so do not remove).
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=8:00:00
#
# nodes: number of 16-core nodes
# ppn: how many cores per node to use (1 through 16)
#       (you are always charged for the entire node)
#PBS -l nodes=1:ppn=6:gpu=4:shared
#
# export all my environment variables to the job
#PBS -V
#
# job name (default = name of script file)
#PBS -N %(jobname)s
#
# specify queue
##PBS -q normal
#
# filename for standard output (default = <job_name>.o<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
#PBS -o ${HOME}/code/yank/test-systems/T4-lysozyme-L99A/amber-gbsa/amber-gbsa/%(molecule)s/stdout.1
#
# filename for standard error (default = <job_name>.e<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
#PBS -e ${HOME}/code/yank/test-systems/T4-lysozyme-L99A/amber-gbsa/amber-gbsa/%(molecule)s/stderr.1
#
# End of embedded QSUB options
#
set echo               # echo commands before execution; use for debugging
#

set JOBID=`echo $PBS_JOBID | cut -d'.' -f1`

# Select number of processors.
setenv NP `wc -l ${PBS_NODEFILE} | cut -d'/' -f1`

# Select job directory.
setenv JOBDIR ${YANKHOME}/test-systems/T4-lysozyme-L99A/amber-gbsa/amber-gbsa/%(molecule)s

# Set YANK directory.
setenv YANKDIR ${YANKHOME}/src/
cd $YANKDIR

date

# Change to job directory
#cd $JOBDIR

# Clean up old working files
rm -f stdout* stderr*
rm -f *.nc

# Set PYTHONPATH
setenv PYTHONPATH ${YANKDIR}:${PYTHONPATH}

# Flat-bottom restraint.
#mpirun -launcher ssh -rmk pbs -np ${NP} -hostfile ${PBS_NODEFILE} python $YANKDIR/yank.py --receptor_prmtop receptor.prmtop --ligand_prmtop ligand.prmtop --complex_prmtop complex.prmtop --complex_crd complex.crd --output . --mpi --restraints flat-bottom --iterations 2000

# Harmonic restraint
mpirun -launcher ssh -rmk pbs -np ${NP} -hostfile ${PBS_NODEFILE} python $YANKDIR/yank.py --receptor_prmtop receptor.prmtop --ligand_prmtop ligand.prmtop --complex_prmtop complex.prmtop --complex_crd complex.crd --output . --mpi --restraints flat-bottom --iterations 2000

date
"""

def isletter(c):
   if ((c >= 'a') and (c <= 'z')) or ((c >= 'A') and (c <= 'Z')):
      return True
   return False   

# T4 lysozyme L99A
#molecules = ['1-methylpyrrole', '2-fluorobenzaldehyde', 'benzene', 'ethylbenzene', 'indole', 'n-butylbenzene', 'n-propylbenzene', 'p-xylene', 'thieno_23c_pyridine', '12-dichlorobenzene', '23-benzofuran', 'benzenedithiol', 'indene', 'isobutylbenzene', 'n-methylaniline', 'o-xylene', 'phenol', 'toluene']
#molecules = ['p-xylene', 'benzene', 'phenol', 'indole', '12-dichlorobenzene']
molecules = ['benzene', 'phenol', '12-dichlorobenzene', '1-methylpyrrole']
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
   if not isletter(molecule[0]):
      jobname = 'Q' + jobname
   # Truncate job name to 15 characters for batch queue system
   jobname = jobname[0:15]
   #print jobname
   
   # Form PBS script
   pbs = pbs_template % vars()

   # Construct directory.
   filename = '../test-systems/T4-lysozyme-L99A/amber-gbsa/amber-gbsa/%(molecule)s/run.pbs' % vars()
   outfile = open(filename, 'w')
   outfile.write(pbs)
   outfile.close()

   # Submit to PBS
   output = commands.getoutput('qsub %(filename)s' % vars());
   print output
   

   
   
   
