#!/bin/bash

# Run the unpredictability ridge regression script for peripheral patches

eccentricity=2.0


# I had a bug, add 90 for future analyses
# for angle in 90 210 330; do
for angle in 330; do

    for subj_num in {01..08}; do
        echo "Running peripheral unpredictability ridge regression for subj$subj_num"
        # /home/rfpred/scripts/run_pred_ridge_peri.py subj$subj_num $eccentricity $angle --mean_unpred
        /home/rfpred/scripts/run_pred_ridge_peri.py subj$subj_num $eccentricity $angle
    done
done