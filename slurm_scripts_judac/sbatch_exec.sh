#!/bin/bash
#SBATCH --account=icei-hbp-2020-0012
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=batch

#SBATCH --job-name=lyon_exec    # Job name
#SBATCH --partition=batch              # Partition choice

##SBATCH --output=../../slurmout/out_%j.out   # Standard output and error log
###/p/project/icei-hbp-2020-0012

#SBATCH --output=/p/project/icei-hbp-2020-0012/slurmout/ly_memerr_out_%A_%a.out   # Standard output and error log
#SBATCH --error=/p/project/icei-hbp-2020-0012/slurmout/ly_memerr_out_%A_%a.out   # Standard output and error log

## hpc, htc or gpu.  Normally htc is stronger but waiting line is bigger
#SBATCH --mem=15G                    # Memory in MB per default
##SBATCH --time=1-00:00:00             # 7 days by default on htc partition
##SBATCH --time=00:40:00             # 7 days by default on htc partition
##SBATCH --time=00:20:00             # 7 days by default on htc partition
##SBATCH --time=05:40:00             # 7 days by default on htc partition
#SBATCH --time=10:40:00             # 7 days by default on htc partition

# I get only 77 jobs running at the same time on hpc. At least when I as for 32 procs

#SBATCH --mail-user=todorovdi@gmail.com   # Where to send mail
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --array=0-255


# Commands to be submitted:
JOBID=$SLURM_JOB_ID
ID=$SLURM_ARRAY_TASK_ID
echo "$JOBID"_"$ID"
#SHIFT_ID=0

jutil env activate -p icei-hbp-2020-0012

module load Stages/2022
module load GCC
#module load R
module load Python/3.9.6
module load matplotlib

source $CODE_MEMORY_ERRORS/__workstart_HPC.sh

# was needed on in2p3
#HOME_DIR=/pbs/home/d/dtodorov
#CODE_DIR=/pbs/home/d/dtodorov/memerr/code
#source $HOME_DIR/.bashrc
## /p/home/jusers/todorov1/jusuf

#echo "DATA_MEMORY_ERRORS_STAB_AND_STOCH=$DATA_MEMORY_ERRORS_STAB_AND_STOCH"
#echo "PYTHONPATH=$PYTHONPATH"
#echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
#
#export PYTHONPATH
#export LD_LIBRARY_PATH
#
#echo 'trying python'
#python -c "import PIL"
#echo 'trying ipython'
#ipython -c "import PIL"

#ipython $CODE_DIR/exec.py
#ipython $CODE_DIR/exec_pre.py $ID
#ipython $CODE_MEMORY_ERRORS/exec_HPC.py $ID


EXIT_IF_ANY_FAILS=0
NFAILS=0
NRUNS=0

RUNSTRINGS_FN="$CODE_MEMORY_ERRORS/_runstrings.txt"

export NUMBA_CACHE_DIR=$LOCALSCRATCH

#which ipython
#exit 0

SHIFT_ID=0
#EFF_ID=$((ID+SHIFT_ID))
#echo "Running job array number: ${ID} (effctive_id = $EFF_ID) on $HOSTNAME,  $SLURM_JOB_ID, $SLURM_ARRAY_JOB_ID"
#$OSCBAGDIS_DATAPROC_CODE/run/srun_exec_runstr.sh $RUNSTRINGS_FN $SLURM_ARRAY_JOB_ID $EFF_ID

MAXJOBS=256 # better this than 64, otherwise more difficult on the level of indtool

NUMRS=`wc -l $RUNSTRINGS_FN | awk '{print $1;}'`
echo "Start now"
echo "SBATCH TYPE: CPU MULTIRUN"
while [ $NUMRS -gt $SHIFT_ID ]; do
  EFF_ID=$((ID+SHIFT_ID))
  if [ $EFF_ID -ge $NUMRS ]; then
    break
  fi
  echo "Running job array number: ${ID} (effctive_id = $EFF_ID) on $HOSTNAME,  $SLURM_JOB_ID, $SLURM_ARRAY_JOB_ID"
  #python -c "1/0"

  #$OSCBAGDIS_DATAPROC_CODE/run/srun_exec_runstr.sh $RUNSTRINGS_FN $SLURM_ARRAY_JOB_ID $EFF_ID $ID
  #ipython --ipython-dir=$LOCALSCRATCH $CODE_MEMORY_ERRORS/exec_HPC.py $EFF_ID

  #############################   Execute 
  sedind=$(( $EFF_ID + 1 ))
  runstr=`sed -n "$sedind"p $RUNSTRINGS_FN`
  runstr="$runstr --SLURM_job_id $SLURM_JOB_ID" 
  echo "$runstr"
  # execute
  $runstr
  EXCODE=$?
  #############################    


  #EXCODE=0
  echo "---!!!--- Current run error code: $EXCODE"
  if [[ $EXCODE -ne 0 ]]; then 
    NFAILS=$((NFAILS + 1))
    echo "NFAILS=$NFAILS"
  fi

  if [[ $EXCODE -ne 0 ]] && [[ $EFF_ID -eq 0 ]]; then
    echo "Exiting due to bad error code in test :("
    exit $EXCODE
  fi

  if [[ $EXCODE -ne 0 ]] && [[ $EXIT_IF_ANY_FAILS -ne 0 ]]; then
    echo "Exiting due to bad error code :("
    exit $EXCODE
  fi
  SHIFT_ID=$((SHIFT_ID + MAXJOBS))

  echo "FINISHED job array number: ${ID} (effctive_id = $EFF_ID) on $HOSTNAME,  $SLURM_JOB_ID, $SLURM_ARRAY_JOB_ID"
  NRUNS=$((NRUNS + 1))

  echo "----------------"
  echo "----------------"
  echo "----------------"
  echo "----------------"
  ##########################  ONLY FOR RUNNING OF INDIVID JOBS
  #break
  ##########################
done


echo "---!!!--- END OF EVERYTHING: failed $NFAILS of $NRUNS "
echo "---!!!--- End error code: $NFAILS"
exit $NFAILS  # added afterrunning job 233971
