#!/bin/bash

rsync -rtvzh --progress --exclude *.meg4 --exclude *.npy --exclude *.ds --exclude *results* $DATA_QUENTIN/data2 $HOME/ju_oscbagdis/lyon_tmp

