#!/bin/bash

# This script chains the acquisition, storage, and analysis scripts for the AlexNet encoding features of the NSD images.

# Run the encoding_stack.sh script
source ./scripts/encoding_stack.sh

# Store the output of the encoding_stack.sh script in a format ready to be regressed
./scripts/store_enc_feats.py

# Run the regressions for the encoding features
./scripts/run_cnn_ridge.py