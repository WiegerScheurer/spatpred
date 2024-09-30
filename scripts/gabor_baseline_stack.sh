#!/bin/bash

# Run the unpredictability ridge regression script for peripheral patches


# Iterate over the first argument from 0 to 29750 in steps of 500
for arg1 in $(seq 0 1000 72000); do
# for arg1 in $(seq 0 1000 29000); do
# for arg1 in $(seq 5000 500 6500); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1000))
    # Run the Python script with the current arguments
    # python ./scripts/get_pred.py $arg1 $arg2 'subj01'
    python ./scripts/gabor_baseline.py $arg1 $arg2 --filetag "all_imgs_sf4_dir6"
done
