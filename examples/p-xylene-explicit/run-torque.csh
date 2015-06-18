#!/bin/tcsh
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=8:00:00
#
# join stdout and stderr
#PBS -j oe
#
# spool output immediately
#PBS -k oe
#
# specify queue
#PBS -q gpu
#
# nodes: number of 8-core nodes
#   ppn: how many cores per node to use (1 through 8)
#       (you are always charged for the entire node)
#PBS -l nodes=1:ppn=4:gpus=4:shared:gtxtitan
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N p-xylene
#
# specify email
#PBS -M jchodera@gmail.com
#
# mail settings
#PBS -m n
#
# filename for standard output (default = <job_name>.o<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
#PBS -o /cbio/jclab/projects/musashi/yank.jchodera/examples/p-xylene/torque.out

cd $PBS_O_WORKDIR

date
yank prepare binding amber --setupdir=setup --ligand="resname MOL" --store=output --iterations=1000 --nbmethod=CutoffPeriodic --temperature="300*kelvin" --pressure="1*atmosphere" --minimize --verbose
date

