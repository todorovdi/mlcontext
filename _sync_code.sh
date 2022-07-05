#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "Need one arg"
  exit 1
fi
 
DRY_RUN_FLAG=""
if [[ $1 == "dry" ]]; then
  echo " ----------  DRY RUN  ------------"
  DRY_RUN_FLAG="n"
elif [[ $1 == "normal" ]]; then
  echo " ----------  NORMAL RUN  ------------"
else
  echo " ----------  WRONG CMD OPTION  ------------"
  exit 1
fi

LOCAL_DIR="/home/demitau/memerr_code"
FLAGS="-rtvhz$DRY_RUN_FLAG --progress"
# for dry run
#FLAGS="-rtvzn --progress"; echo "DRY RUN!!"

CLUSTER_PROJECT_PATH="/pbs/home/d/dtodorov/memerr"
CLUSTER_CODE_PATH="$CLUSTER_PROJECT_PATH/code"

DIRECT_SSH=0
if [ $DIRECT_SSH -ne 0 ]; then
  echo "Using rsync -e ssh"
  SSH_FLAG="-e ssh"
  CLUSTER="cluster_in2p3:$CLUSTER_CODE_PATH"
  CLUSTER_BASE="cluster_in2p3:$CLUSTER_PROJECT_PATH"
  SLEEP="sleep 1s"
else
  mountpath="$HOME/remote_in2p3"
  numfiles=`ls $mountpath | wc -l`
  MQR=`mountpoint -q "$mountpath"`
  while [ $numfiles -eq 0 ] || ! mountpoint -q "$mountpath"; do
    echo "not mounted! trying to remount; numfiles=$numfiles MQR=$MQR"
    sudo umount -l $mountpath # would not work if I run on cron
    sshfs judac:/sps/crnl/todorov $mountpath
    #exit 1
    sleep 3s
    numfiles=`ls $mountpath | wc -l`
    MQR=`mountpoint -q "$mountpath"`
  done

  echo "Using mounted sshfs"
  SSH_FLAG=""
  CLUSTER="$HOME/remote_in2p3/memerr/code"
  CLUSTER_BASE="$HOME/remote_in2p3/memerr"
  SLEEP=""
fi


echo "  sync souce code"
rsync $FLAGS $SSH_FLAG --exclude="*_orig.*" $LOCAL_DIR/*.{py,sh}  $CLUSTER/
$SLEEP
echo "  sync figure making code"
subdir=figure
rsync $FLAGS $SSH_FLAG $LOCAL_DIR/$subdir/*.py  $CLUSTER/$subdir
$SLEEP
rsync $FLAGS $SSH_FLAG $LOCAL_DIR/*.ipynb  $CLUSTER/jupyter
#subdir=run


#sshfs cluster_in2p3:/pbs/home/d/dtodorov /home/demitau/remote_in2p3/

#rsync -rtvz --progress --exclude *.npy --exclude *.ds --exclude *results* $DATA_QUENTIN/data2 /home/demitau/remote_in2p3/memerr/data2
