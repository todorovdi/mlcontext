#!/bin/bash
#SBATCH --job-name=exec    # Job name
##SBATCH --output=../../slurmout/out_%j.out   # Standard output and error log
#SBATCH --output=../../slurmout/out_%A_%a.out   # Standard output and error log
#SBATCH --error=../../slurmout/out_%A_%a.out   # Standard output and error log

## hpc, htc or gpu.  Normally htc is stronger but waiting line is bigger
#SBATCH --partition=hpc              # Partition choice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=15G                    # Memory in MB per default
##SBATCH --time=1-00:00:00             # 7 days by default on htc partition
#SBATCH --time=00:20:00             # 7 days by default on htc partition
#SBATCH --cpus-per-task=16
##SBATCH --cores-per-socket=32

# I get only 77 jobs running at the same time on hpc. At least when I as for 32 procs

#SBATCH --mail-user=todorovdi@gmail.com   # Where to send mail
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --array=0-19

# Commands to be submitted:
JOBID=$SLURM_JOB_ID
ID=$SLURM_ARRAY_TASK_ID
echo "$JOBID"_"$ID"
#SHIFT_ID=0


HOME_DIR=/pbs/home/d/dtodorov
CODE_DIR=/pbs/home/d/dtodorov/memerr/code
source $HOME_DIR/.bashrc

echo "DATA_MEMORY_ERRORS_STAB_AND_STOCH=$DATA_MEMORY_ERRORS_STAB_AND_STOCH"

#ipython $CODE_DIR/exec.py
#ipython $CODE_DIR/exec_pre.py $ID
ipython $CODE_DIR/exec_HPC.py $ID
