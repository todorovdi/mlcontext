#!/bin/bash
# source and destination directories
src_dir="$1"
dst_dir="$2"

# Loop through each file in the source directory
for src_file in "$src_dir"/*; do
    # Extract the filename from the source file path
    filename=$(basename "$src_file")
    
    # Construct the corresponding destination file path
    dst_file="$dst_dir/$filename"
    
    # Check if the destination file exists
    if [ -e "$dst_file" ]; then
        # Get the access and modification times from the source file
        src_atime=$(stat -c %X "$src_file") # Access time
        src_mtime=$(stat -c %Y "$src_file") # Modification time
        dst_atime=$(stat -c %X "$dst_file") # Access time
        dst_mtime=$(stat -c %Y "$dst_file") # Modification time

        src_mtime_hr=$(date -d @$src_mtime +"%Y-%m-%d %H:%M:%S")
        dst_mtime_hr=$(date -d @$dst_mtime +"%Y-%m-%d %H:%M:%S")
        
        echo "source mtime: $src_mtime_hr  of  $src_file"
        echo "dst    mtime: $dst_mtime_hr  of  $dst_file" 
        
        confirm="y"
        ## if uncommented, then ask for user confirmation
        #read -p "Do you want to transfer the mtime and atime from source to destination? (y/n): " confirm

        if [ "$confirm" = "y" ]; then
            # Get the access time from the source file
            src_atime=$(stat -c %X "$src_file") # Access time

            # Apply the access time to the destination file
            touch -a -t $(date -d @$src_atime +%Y%m%d%H%M.%S) "$dst_file"
            
            # Apply the modification time to the destination file
            touch -m -t $(date -d @$src_mtime +%Y%m%d%H%M.%S) "$dst_file"
            
            echo "Timestamps transferred successfully."
        else
            echo "Transfer skipped for $dst_file."
        fi


    else
        echo "Destination file $dst_file does not exist."
    fi
done

