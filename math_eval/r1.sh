#!/bin/bash
file_names=(
    "eval.json"
)

for file_name in "${file_names[@]}"; do
    echo "Processing file: $file_name"
    python r1.py --file_name "$file_name"
    echo "Finished processing: $file_name"
    echo "---------------------------"
done

echo "All files processed!"