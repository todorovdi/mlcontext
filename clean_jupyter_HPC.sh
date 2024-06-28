#!/bin/bash
LOCAL_DIR="$CODE_MEMORY_ERRORS"

echo " self clean jupyter"
#sd=jupyter_debug
#mkdir $LOCAL_DIR/"$sd"_cleaned  
dirs=("jupyter_behav_bonaiuto" "jupyter_behav_cc" "jupyter_behav_NIH" "jupyter_MEG_cc" "jupyter_MEG_NIH" "jupyter_model_beh")
for sd in "${dirs[@]}"; do
  #sdfull="$LOCAL_DIR/$sd"
  sdfull="$LOCAL_DIR/jupyter_dirty/$sd"
  sdfull_res="$LOCAL_DIR/jupyter_cleaned/$sd"
  echo "processing $sdfull to $sdfull_res"
  mkdir -p "$sdfull_res"
  python -m nbconvert --ClearOutputPreprocessor.enabled=True $sdfull/*.ipynb --to notebook --output-dir=$sdfull_res   

  ./setmtimes.sh "$sdfull" "$sdfull_res" 
done
