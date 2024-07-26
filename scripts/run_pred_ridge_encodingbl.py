#!/usr/bin/env python3

import os
import sys

os.environ["OMP_NUM_THREADS"] = "5"
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import argparse

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

from unet_recon.inpainting import UNet
from funcs.analyses import univariate_regression

import importlib
from importlib import reload
import classes.natspatpred
import unet_recon.inpainting
importlib.reload(classes.natspatpred)
importlib.reload(unet_recon.inpainting)
from unet_recon.inpainting import UNet
from classes.natspatpred import NatSpatPred, VoxelSieve

NSP = NatSpatPred()
NSP.initialise()

predparser = argparse.ArgumentParser(description='Get the predictability estimates for a range of images of a subject')

predparser.add_argument('subject', type=str, help='The subject')

args = predparser.parse_args()

rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(verbose=False)
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)

############ CONSTRAINED VOXEL SELECTION Y-MATRIX ################

subject = args.subject
max_size = 2
min_size = .15
patchbound = 1
min_nsd_R2 = 0
min_prf_R2 = 0
# fixed_n_voxels = 170

voxeldict = {}
n_voxels = []
for roi in rois:
    print_attr = True if roi == rois[len(rois)-1] else False
    voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
                                subject=subject, 
                                roi=roi,
                                patchloc='central', 
                                max_size=max_size, 
                                min_size=min_size, 
                                patchbound=patchbound, 
                                min_nsd_R2=min_nsd_R2, 
                                min_prf_R2=min_prf_R2,
                                print_attributes=print_attr,
                                fixed_n_voxels=None)
    n_voxels.append(len(voxeldict[roi].size))
    
max_n_voxels = np.min(n_voxels)

ydict = {}
for roi in rois:
    ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
    print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')

# layers = [1, 4, 7, 9, 11] # These are the possible layers, but most sensible to
# Use the first one, as then it is also a low-level features baseline.

# Use the alexnet feature maps as the baseline model
# baseline_layers = [1, 4, 7, 9, 11]
baseline_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
baseline_layer = "all"
# baseline_layers = [0, 5, 12, 19, 26]# 12, 14, 17, 19, 21, 24, 26, 28]

modeltype = "VGG" # Same as vggfull but I saved the encoding feats under VGG and unpred feats under vggfull for clarity (this is irony)
# for baseline_layer in baseline_layers:

Xbl = NSP.stimuli.alex_featmaps(baseline_layers, subject, plot_corrmx=False, smallpatch=True, modeltype=modeltype)[:ydict["V1"].shape[0]]

# which_cnn = 'vgg-b'
# which_cnn = 'alexnet'
# which_cnn = 'alexnet_new'
which_cnn = "vggfull"
# n_layers = 5 if which_cnn == 'alexnet' else 6
n_layers = len(baseline_layers)


Xpred = NSP.stimuli.unpred_feats(cnn_type=which_cnn, content=True, style=False, ssim=False, pixel_loss=False, 
                                L1=False, MSE=True, verbose=True, outlier_sd_bound=5, subject=subject)[:ydict["V1"].shape[0]]

X = np.hstack((Xbl, Xpred[:, 0].reshape(-1, 1)))

print(f'Xpred has these dimensions: {Xpred.shape}')
    
if which_cnn == 'alexnet_new': # Remove this later, only for clarity of files
    which_cnn = 'alexnet'
    
for layer in range(0, Xpred.shape[1]):
    feat = f'{which_cnn}_lay{layer}'
    X_unpred = Xpred[:,layer].reshape(-1,1)
    X = np.hstack((Xbl, X_unpred))
    print(f'X has these dimensions: {X.shape}')
    
    reg_df = NSP.analyse.analysis_chain_slim(subject=subject,
                            ydict=ydict,
                            voxeldict=voxeldict,
                            X=X,
                            alpha=.1,
                            cv=5,
                            rois=rois,
                            X_alt=Xbl, # The baseline model
                            fit_icept=False,
                            save_outs=True,
                            regname=feat,
                            plot_hist=False,
                            alt_model_type="baseline model",
                            save_folder=f'unpred_encodingbl_{modeltype}_{baseline_layer}',
                            X_str=f'{feat} model')
