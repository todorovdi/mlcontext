#!/bin/bash
#sshfs cluster_in2p3:/pbs/home/d/dtodorov /home/demitau/remote_in2p3/
#sshfs cluster_in2p3:/sps/crnl/todorov  /home/demitau/remote_in2p3_data/
# remote2 is for data

rsync -rtvzh --progress --exclude *.meg4 --exclude *.npy --exclude *.ds --exclude *results* $DATA_QUENTIN/data2 $HOME/remote_in2p3_data/memerr/

