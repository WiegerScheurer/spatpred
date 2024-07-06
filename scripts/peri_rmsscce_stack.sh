#!/bin/bash

eccentricity=1.2

for angle in 90 210 330; do
    echo "Computing the local contrast features for the peripheral patch at $angle degrees, with $eccentricity eccentricity"
    /home/rfpred/scripts/peri_rmsscce_comp.py 1.2 $angle
done