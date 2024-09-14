#!/usr/bin/env python3

# Script to compute the gabor pyramid filter outputs for the NSD stimuli

import os
import sys
import importlib
import moten
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from skimage import color
from scipy.stats import zscore as zs

print(sys.path)

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

from classes.voxelsieve import VoxelSieve
from classes.natspatpred import NatSpatPred
NSP = NatSpatPred()
NSP.initialise()

predparser = argparse.ArgumentParser(description='Get the predictability estimates for a range of images of a subject')

predparser.add_argument('start', type=int, help='The starting index of the images to get the predictability estimates for')
predparser.add_argument('end', type=int, help='The ending index of the images to get the predictability estimates for')
predparser.add_argument('--subject', type=str, help='The subject to get the predictability estimates for', default=None)

args = predparser.parse_args()

from funcs.gaborpyr import (
    isotropic_gaussian,
    circle_stim,
    random_dot,
    show,
    cut_corners,
    make_checker,
    plot_filter_locations,
    plot_filter_outputs,
    normalize_output,
    select_filters,
)


subject = args.subject
pixels = 425
degrees = 8.4
pix_per_deg = pixels / degrees

cirkel = circle_stim((425, 425), (1 * pix_per_deg), (212, 212))

gauss = isotropic_gaussian(dims=(425,425), sigma=pix_per_deg/4) # Justify this 2.5 factor

checker_stim = make_checker(dims=(425,425), checkercenter=(212,212), scales=3, scaling_factor=3, checker_size=50, stride=0)

fig, axes = plt.subplots(1, 3, figsize=(15,6))

for img_no, img in enumerate([gauss, checker_stim, checker_stim * gauss]):
    axes[img_no].imshow(img, cmap='gist_gray')
    axes[img_no].axis("off")
plt.tight_layout()

gauss_check_stack = np.stack([gauss, checker_stim * gauss], axis=0)

# Original spatfreqs = [0.25, 0.5, 1, 2] in cycles per image (so cycles per 8.4 degrees)
# If I want to transform this to cycles per degree, I need to divide by 8.4
spat_freqs = [4.2, 8.4, 16.8, 33.6]

checkpyramid = moten.pyramids.StimulusStaticGaborPyramid(stimulus=gauss_check_stack,
                                                spatial_frequencies=[4.2, 8.4, 16.8, 33.6], # 1, 2, 4, 8 cycles per degree
                                                # spatial_frequencies=[33.6], # 1, 2, 4, 8 cycles per degree
                                                # spatial_orientations=(0, 45, 90, 135),
                                                spatial_orientations=tuple(range(0, 180, 20)),
                                                sf_gauss_ratio=1, # ratio of spatial frequency to gaussian s.d.
                                                max_spatial_env=(1/8.4), # max sd of gaussian envelope
                                                filter_spacing=.5,
                                                include_edges=False, # Should be false, we're not interested in the edges
                                                spatial_phase_offset=0,)

checkpyramid.view.nfilters

file_exists = os.path.isfile(f"{NSP.own_datapath}/visfeats/gabor_pyramid/gauss_checker_output.npy")

if file_exists:
    print("Loading the filter selection from file")
    gauss_output = np.load(f"{NSP.own_datapath}/visfeats/gabor_pyramid/gauss_checker_output.npy")
else:
    gauss_output = checkpyramid.project_stimulus(gauss_check_stack)

# Figure out how many filters there are per spatial frequency
filters_per_freq= []
for sf in spat_freqs:

    all_filters = checkpyramid.view.filters
    count = sum(1 for d in all_filters if d.get('spatial_freq') == sf)
    filters_per_freq.append(count)
    
filters_per_freq

output_norm, filters_per_freq_sel, filter_selection, filter_selection_dictlist = (
    select_filters(
        pyramid=checkpyramid,
        output=gauss_output,
        imgs=gauss_check_stack,
        img_no=1,
        spat_freqs=spat_freqs,
        filters_per_freq=filters_per_freq,
        percentile_cutoff=99.9,
        plot=False,
        verbose=True,
    )
)

# The indices for the filters that are within the patch
filter_indices = np.where(filter_selection == True)[0]

# Now we can project the NSD images

start_img = args.start
end_img = args.end

# FOR NOW I DEFINE THE SUBJECT BECAUSE I WANT TO CHECK THE PERFORMANCE
# OF THE MODEL BEFORE COMPUTING IT FOR THE 73K IMAGES
imgs,_ = NSP.stimuli.rand_img_list(n_imgs=(end_img-start_img), 
                                   asPIL=False, 
                                   add_masks=False, 
                                   select_ices=NSP.stimuli.imgs_designmx()[args.subject][start_img:end_img])

img_list = []

print("Converting images to luminance channel")
for img_no, img in enumerate(imgs):

    # Convert RGB image to LAB colour space
    lab_image = color.rgb2lab(imgs[img_no])

    # First channel [0] is Luminance, second [1] is green-red, third [2] is blue-yellow
    lumimg = lab_image[
        :, :, 0
    ]  # Extract the L channel for luminance values, assign to input array

    img_list.append(lumimg)

imgstack = np.array(img_list)


print(f"Building pyramid for images {start_img} to {end_img}")
nsdpyramid = moten.pyramids.StimulusStaticGaborPyramid(stimulus=imgstack,
                                                spatial_frequencies=[4.2, 8.4, 16.8, 33.6], # 1, 2, 4, 8 cycles per degree
                                                # spatial_frequencies=[33.6], # 1, 2, 4, 8 cycles per degree
                                                # spatial_orientations=(0, 45, 90, 135),
                                                spatial_orientations=tuple(range(0, 180, 20)),
                                                sf_gauss_ratio=1, # ratio of spatial frequency to gaussian s.d.
                                                max_spatial_env=(1/8.4), # max sd of gaussian envelope
                                                filter_spacing=.5,
                                                include_edges=False, # Should be false, we're not interested in the edges
                                                spatial_phase_offset=0,
)

print(f"Ended up with a total filter count of: {nsdpyramid.view.nfilters}")

nsd_output = nsdpyramid.project_stimulus(imgstack, filters=filter_selection_dictlist)

nsd_output_norm = normalize_output(nsd_output, len(spat_freqs), filters_per_freq)

os.makedirs(f"{NSP.own_datapath}/visfeats/gabor_pyramid/batches", exist_ok=True)
np.save(f"{NSP.own_datapath}/visfeats/gabor_pyramid/batches/gabor_baseline_{start_img}_{end_img}.npy", nsd_output_norm)

print(f"Saved the output to {NSP.own_datapath}/visfeats/gabor_pyramid/batches/gabor_baseline_{start_img}_{end_img}.npy")