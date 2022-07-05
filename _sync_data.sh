#!/bin/bash
#sshfs cluster_in2p3:/pbs/home/d/dtodorov /home/demitau/remote_in2p3/
#sshfs cluster_in2p3:/sps/crnl/todorov  /home/demitau/remote2_in2p3/

rsync -rtvzh --progress --exclude *.meg4 --exclude *.npy --exclude *.ds --exclude *results* $DATA_QUENTIN/data2 /home/demitau/remote2_in2p3/memerr/

