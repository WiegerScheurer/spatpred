#!/bin/bash

# This script runs the encoding_feats.py script to iterate over all the images.

# Define the values for pca_fit_batch and n_comps
pca_fit_batch=500
n_comps=500

# Loop over the integer vector for cnn_layer
# for cnn_layer in 1 4 7 9 11; do
# for cnn_layer in 2 5 9 12 16 19 22 26 29 32 36 39 42; do

# selected_indices = [0, 2, 5, 10, 17, 21, 24, 28] #### These are the vgg-16 non batchnorm layers I use for unpredfeats

for cnn_layer in 0 2 5 10 17 21 24 28; do
    # Print the start time
    echo "Start time for cnn_layer $cnn_layer: $(date)"

    # Run the Python script with the current arguments
    ./scripts/encoding_feats_vgg.py $pca_fit_batch $n_comps $cnn_layer

    # Print the end time
    echo "End time for cnn_layer $cnn_layer: $(date)"
done