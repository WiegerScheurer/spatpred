#!/bin/bash

# Run the encoding ridge regression script for all the 

for subj_num in {02..08}; do
    echo "Running classic DNN encoding ridge regression for subj$subj_num"
    /home/rfpred/scripts/run_cnn_ridge.py subj$subj_num
done