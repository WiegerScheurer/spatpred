#!/bin/bash

# This script runs the get_cnnfeats.py script to iterate over all the images for a defined subject. 
# It is needed because otherwise the .py script would overload the cache and crash the terminal. 

#!/bin/bash
# conda deactivate
source activate /home/rfpred/envs/rfenv
# # Iterate over the first argument from 0 to 29000 in steps of 1000
# for arg1 in $(seq 0 1000 29000); do
#     # Calculate arg2 as arg1 + 250
#     arg2=$((arg1 + 1000))
#     # Run the Python script with the current arguments
#     /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 1
# done


# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 1200 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 0
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 1200 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 2
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 600 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 3
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 600 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 5
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 600 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 6
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 600 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 8
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 600 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 9
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 600 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 10
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 600 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 11
done

# Iterate over the first argument from 0 to 29000 in steps of 1000
for arg1 in $(seq 0 600 28800); do
    # Calculate arg2 as arg1 + 250
    arg2=$((arg1 + 1200))

    # Run the Python script with the current arguments
    /home/rfpred/scripts/get_cnnfeats.py $arg1 $arg2 'subj01' 12
done
