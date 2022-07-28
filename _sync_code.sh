#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "Need one arg"
  exit 1
fi
 
DRY_RUN_FLAG=""
RUNTYPE=$1
if [[ $RUNTYPE == "dry" ]]; then
  echo " ----------  DRY RUN  ------------"
  DRY_RUN_FLAG="n"
elif [[ $RUNTYPE == "normal" ]]; then
  echo " ----------  NORMAL RUN  ------------"
else
  echo " ----------  WRONG CMD OPTION  ------------"
  exit 1
fi

run="python3 _rsync_careful.py"
LOCAL_DIR="/home/demitau/memerr_code"
#FLAGS="-rtvh$DRY_RUN_FLAG --progress"
# for dry run
#FLAGS="-rtvzn --progress"; echo "DRY RUN!!"

DIRECT_SSH=0
if [ $DIRECT_SSH -ne 0 ]; then
  echo "Using rsync -e ssh"
  SSH_FLAG="-e ssh -z"
  CLUSTER_PROJECT_DIR="$SSH_HOSTNAME:$CLUSTER_PROJECT_PATH"
  CLUSTER_CODE="$SSH_HOSTNAME:$CLUSTER_CODE_PATH"
  SLEEP="sleep 1s"
  echo "not implemented; need to change _rsync_careful"; exit 1
else
  numfiles=`ls $mountpath | wc -l`
  MQR=`mountpoint -q "$mountpath"`
  while [ $numfiles -eq 0 ] || ! mountpoint -q "$mountpath"; do
    echo "not mounted! trying to remount; numfiles=$numfiles MQR=$MQR"
    sudo umount -l $mountpath # would not work if I run on cron
    #sshfs $SSH_HOSTNAME:/sps/crnl/todorov 
    sshfs $SSH_HOSTNAME:$CLUSTER_USER_PATH $mountpath
    #exit 1
    sleep 3s
    numfiles=`ls $mountpath | wc -l`
    MQR=`mountpoint -q "$mountpath"`
  done

  echo "Using mounted sshfs"
  SSH_FLAG=""
  CLUSTER_PROJECT_DIR="$mountpath/memerr"
  CLUSTER_CODE="$CLUSTER_PROJECT_DIR/code"
  SLEEP=""
fi


echo "  sync souce code"
# if I put *.{py,sh} here, then it does not get parsed well by python
$run --mode:$RUNTYPE --exclude="*HPC*.py" --exclude="*_orig.*" --exclude="_sync*.sh" "$LOCAL_DIR/*.py"  "$CLUSTER_CODE/"
$run --mode:$RUNTYPE --exclude="*HPC*.sh" --exclude="*_orig.*" --exclude="_sync*.sh" "$LOCAL_DIR/*.sh"  "$CLUSTER_CODE/"
if [ $? -ne 0 ]; then
  exit $?
fi
#exit 0
$SLEEP
echo "  sync figure making code"
subdir=figure
$run --mode:$RUNTYPE --exclude="*HPC*.py" "$LOCAL_DIR/$subdir/*.py"  "$CLUSTER_CODE/$subdir"
#$run $FLAGS $SSH_FLAG   
echo "  sync params"
subdir=params
$run --mode:$RUNTYPE  "--exclude *HPC*.ini" "$LOCAL_DIR/$subdir/*.ini"  $CLUSTER_CODE/$subdir
$SLEEP
#subdir=jupyter_debug
#$run --mode:$RUNTYPE $LOCAL_DIR/$subdir  $CLUSTER_CODE/$subdir
#subdir=jupyter_release
#$run --mode:$RUNTYPE $LOCAL_DIR/$subdir  $CLUSTER_CODE/$subdir


echo "  REV sync souce code"
$run --mode:$RUNTYPE  "$CLUSTER_CODE/*_HPC.py"  $LOCAL_DIR/
$SLEEP
echo "  REV sync scripts"
$run --mode:$RUNTYPE  "$CLUSTER_CODE/sbatch*.sh"  $LOCAL_DIR/slurm_scripts_"$SSH_HOSTNAME"
$run --mode:$RUNTYPE  "--exclude sbatch*.sh" "$CLUSTER_CODE/*.sh"  $LOCAL_DIR

echo "  REV sync params"
subdir=params
$run --mode:$RUNTYPE  "--exclude sbatch*.sh" "$CLUSTER_CODE/$subdir/*HPC*.ini"  $LOCAL_DIR/$subdir
#$SLEEP
#sshfs $SSH_HOSTNAME:/pbs/home/d/dtodorov /home/demitau/remote_in2p3/

#$run -rtvz --progress --exclude *.npy --exclude *.ds --exclude *results* $DATA_QUENTIN/data2 /home/demitau/remote_in2p3/memerr/data2


# echo "  sync souce code"
# $run $FLAGS $SSH_FLAG --exclude="*_orig.*" --exclude="_sync*.sh" $LOCAL_DIR/*.{py,sh}  $CLUSTER_CODE/
# if [ $? -ne 0 ]; then
#   exit $?
# fi
# $SLEEP
# echo "  sync figure making code"
# subdir=figure
# $run $FLAGS $SSH_FLAG $LOCAL_DIR/$subdir/*.py  $CLUSTER_CODE/$subdir
# $SLEEP
# $run $FLAGS $SSH_FLAG $LOCAL_DIR/*.ipynb  $CLUSTER_CODE/jupyter
# #subdir=run
# 
# 
# echo "  REV sync souce code"
# $run $FLAGS $SSH_FLAG $CLUSTER_CODE/*_HPC.py  $LOCAL_DIR/*.py
# $SLEEP
# echo "  REV sync souce code"
# $run $FLAGS $SSH_FLAG $CLUSTER_CODE/*.sh  $LOCAL_DIR/*.{py,sh}
# $SLEEP
