#!/bin/bash
if git rev-parse --git-dir > /dev/null 2>&1; then
  echo 'Git found'
  GITHERE=1

  # save current code version and make it transferable to HPC (we don't have git there)
  v=`git tag | tail -1` 
  h=`git rev-parse --short HEAD`
  #echo "$v, hash=$h" > last_code_ver_synced_with_HPC.txt
  # it's better to have in both dirs because in MEG room it will search for this file in the current dir
  #touch $CODE_MEMORY_ERRORS/exper_protoco/last_commit_info.txt 
  echo "git_tag=$v, hash=$h" > $CODE_MEMORY_ERRORS/exper_protocol/last_commit_info.txt
  #touch $CODE_MEMORY_ERRORS/last_commit_info.txt 
  echo "git_tag=$v, hash=$h" > $CODE_MEMORY_ERRORS/last_commit_info.txt
fi

