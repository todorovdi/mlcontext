#!/bin/bash
CLUSTER_USER_PATH="judac:/p/project/icei-hbp-2020-0012/lyon"
CLUSTER_PROJECT_PATH="$CLUSTER_USER_PATH/memerr"
CLUSTER_CODE_PATH="$CLUSTER_PROJECT_PATH/code"
SSH_HOSTNAME=judac
mountpath="$HOME/ju_lyon"


DRY_RUN_FLAG=""
LOCAL_DIR="/home/demitau/memerr_code"
FLAGS="-rtvz$DRY_RUN_FLAG --progress"
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

LOCAL=1
REMOTE=1

if [ $LOCAL -gt 0 ]; then
  echo " self clean jupyter"
  sd=jupyter_debug
  mkdir $LOCAL_DIR/"$sd"_cleaned  
  python -m nbconvert --ClearOutputPreprocessor.enabled=True $LOCAL_DIR/$sd/*.ipynb --to notebook --output-dir=$LOCAL_DIR/"$sd"_cleaned   

  ##
  sd=jupyter_release
  mkdir $LOCAL_DIR/"$sd"_cleaned  
  python -m nbconvert --ClearOutputPreprocessor.enabled=True $LOCAL_DIR/$sd/*.ipynb --to notebook --output-dir=$LOCAL_DIR/"$sd"_cleaned  
fi


if [ $REMOTE -gt 0 ]; then
  echo "  rev rsync jupyter debug"
  sd=jupyter_debug
  sd_HPC="$sd"_cleaned
  sd_loc="$sd"_HPC_cleaned
  mkdir $LOCAL_DIR/$sd_loc  
  mkdir $CLUSTER_CODE/$sd_HPC
  python -m nbconvert --ClearOutputPreprocessor.enabled=True $CLUSTER_CODE/$sd/*.ipynb --to notebook --output-dir=$CLUSTER_CODE/$sd_HPC
  rsync $FLAGS $SSH_FLAG $CLUSTER_CODE/$sd_HPC/*.ipynb $LOCAL_DIR/$sd_loc  

  echo "  rev rsync jupyter release"
  sd=jupyter_release
  sd_HPC="$sd"_cleaned
  sd_loc="$sd"_HPC_cleaned
  mkdir $LOCAL_DIR/$sd_loc  
  mkdir $CLUSTER_CODE/$sd_HPC
  python -m nbconvert --ClearOutputPreprocessor.enabled=True $CLUSTER_CODE/$sd/*.ipynb --to notebook --output-dir=$CLUSTER_CODE/$sd_HPC
  rsync $FLAGS $SSH_FLAG $CLUSTER_CODE/$sd_HPC/*.ipynb $LOCAL_DIR/$sd_loc
fi
