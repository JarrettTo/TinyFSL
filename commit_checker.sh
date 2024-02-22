#!/bin/bash

# Maximum file size in bytes (100MB)
MAX_SIZE_BYTES=100000000

# Get the list of files under the camgoz_model folder
files=$(find sign_squeezenet_model -type f)

for file in $files; do
    echo "Now checking the file '$file'."
    
    # Get the size of the file
    size=$(wc -c < "$file")
    # Check if the file size exceeds the limit
    if [ $size -gt $MAX_SIZE_BYTES ]; then
        echo "Error: File '$file' is larger than 100MB and cannot be committed."
    else
        # If the file is below the size limit, add it to the index
        git add "$file"
    fi
done

# Commit the files added to the index with the specified message
git commit -m "Add files under the size limit in sign_squeezenet_model folder"
