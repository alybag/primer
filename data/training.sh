#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH -J "myjob"   # job name
#SBATCH --mail-user=alybag@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
cd /fslhome/alybag/training/bin

/fslhome/alybag/training/bin/output0
#/fslhome/alybag/training/bin/output1
#/fslhome/alybag/training/bin/output2
#/fslhome/alybag/training/bin/output3
#/fslhome/alybag/training/bin/output4
#/fslhome/alybag/training/bin/output5
#/fslhome/alybag/training/bin/output6
