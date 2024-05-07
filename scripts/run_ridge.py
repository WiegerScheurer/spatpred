#!/usr/bin/env python3

import os
import sys

from regex import F

os.environ["OMP_NUM_THREADS"] = "5"
import os
import sys
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns
import nibabel as nib
import pickle
import torchvision.models as models
import nibabel as nib
import h5py
import scipy.stats.mstats as mstats
import copy

from nilearn import plotting
from scipy.ndimage import binary_dilation
from PIL import Image
from importlib import reload
from scipy.io import loadmat
from matplotlib.ticker import MultipleLocator, NullFormatter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from colorama import Fore, Style
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from tqdm import tqdm
from matplotlib.lines import Line2D
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from typing import Dict, Tuple, Union
from scipy.special import softmax

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

from unet_recon.inpainting import UNet
from funcs.analyses import univariate_regression

import importlib
from importlib import reload
import funcs.natspatpred
import unet_recon.inpainting
importlib.reload(funcs.natspatpred)
importlib.reload(unet_recon.inpainting)
from unet_recon.inpainting import UNet
from funcs.natspatpred import NatSpatPred, VoxelSieve

NSP = NatSpatPred()
NSP.initialise()

# TODO: also return the cor_scores for the uninformative x matrix and create brainplots where
# the r-values are plotted on the brain for both the informative and uninformative x matrices
# Or well, more importantly find a way to visualise how they compare, because otherwise it's
# badly interpretable.

rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(verbose=False)
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)

############ CONSTRAINED VOXEL SELECTION Y-MATRIX ################
##### ALSO RUN THIS FOR THE PRED FEATS SEPARATELY WITHOUT THE BASELINE #########

# subject = 'subj01'
# max_size = 2
# min_size = .2
# patchbound = 1.5
# min_nsd_R2 = 20
# min_prf_R2 = 0

# voxeldict = {}
# for roi in rois:
#     print_attr = True if roi == rois[len(rois)-1] else False
#     voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
#                                 subject=subject, 
#                                 roi=roi, 
#                                 max_size=max_size, 
#                                 min_size=min_size, 
#                                 patchbound=patchbound, 
#                                 min_nsd_R2=min_nsd_R2, 
#                                 min_prf_R2=min_prf_R2,
#                                 print_attributes=print_attr,
#                                 all_voxels=False)

# ydict = {}
# for roi in rois:
#     ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
#     print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')

# # Define the baseline model, which is the rms, ce and sc_l features.
# # TODO: Also include alexnet featmaps?
# baseline_strings = ['rms', 'ce', 'sc_l']

# Xbl = np.hstack((NSP.stimuli.baseline_feats(baseline_strings[0]), 
#                NSP.stimuli.baseline_feats(baseline_strings[1]), 
#                NSP.stimuli.baseline_feats(baseline_strings[2])))

    
# Xpred = NSP.stimuli.unpred_feats(content=True, style=False, ssim=False, pixel_loss=False, L1=False, MSE=True, verbose=True)

# for layer in range(0, 5):
#     Xalex = Xpred[:,layer].reshape(-1,1)
#     X = np.hstack((Xbl, Xalex))
#     print(f'X has these dimensions: {X.shape}')
#     X_shuf = np.copy(X)
#     np.random.shuffle(X_shuf)

#     obj = NSP.analyse.analysis_chain(subject=subject,
#                                      ydict=ydict, 
#                                      X=X, # The baseline model + current unpredictability layer
#                                      alpha=10, 
#                                      voxeldict=voxeldict, 
#                                      cv=5, 
#                                      rois=rois, 
#                                      X_uninformative=Xbl, # The baseline model
#                                      fit_icept=False, 
#                                      save_outs=True,
#                                      regname=f'unpred_lay{layer}')
    
#     # This is for the relative R scores.
#     rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))
#     rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

#     # plot the relative R scores
#     NSP.analyse.plot_brain(prf_dict, 
#                            roi_masks, 
#                            subject, 
#                            NSP.utils.cap_values(np.copy(rel_scores_np), 0, 10), 
#                            False, 
#                            save_img=True, 
#                            img_path=f'/home/rfpred/imgs/reg/unpred_lay{layer}_regcorplot.png')

#     # This is for the betas
#     plot_bets = np.hstack((obj[:,:3], obj[:,5].reshape(-1,1)))
#     plot_bets_np = NSP.utils.coords2numpy(plot_bets, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

#     # plot the betas
#     NSP.analyse.plot_brain(prf_dict, 
#                            roi_masks, 
#                            subject, 
#                            NSP.utils.cap_values(np.copy(plot_bets_np), 0, 10), 
#                            False, 
#                            save_img=True, 
#                            img_path=f'/home/rfpred/imgs/reg/unpred_lay{layer}_regbetaplot.png')


################### FULL VISUAL CORTEX (V1, V2, V3, V4) REGRESSIONS ###############

subject = 'subj01'

voxeldict = {}
for roi in rois:
    print_attr = True if roi == rois[len(rois)-1] else False
    voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
                                subject=subject, 
                                roi=roi, 
                                print_attributes=print_attr,
                                all_voxels=True)


# subject = 'subj01'
# max_size = 2
# min_size = .5
# patchbound = 1.5
# min_nsd_R2 = 20
# min_prf_R2 = 0

# voxeldict = {}
# for roi in rois:
#     print_attr = True if roi == rois[len(rois)-1] else False
#     voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
#                                 subject=subject, 
#                                 roi=roi, 
#                                 max_size=max_size, 
#                                 min_size=min_size, 
#                                 patchbound=patchbound, 
#                                 min_nsd_R2=min_nsd_R2, 
#                                 min_prf_R2=min_prf_R2,
#                                 print_attributes=print_attr,
#                                 all_voxels=False)

ydict = {}

for roi in rois:
    ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
    print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')
    
for layer in range(2, 5):
    print(f'Running regression for layer: {layer}')
    
    X = NSP.stimuli.unet_featmaps(list_layers=[layer], scale='full') # Get X matrix
    print(f'X has these dimensions: {X.shape}')
    X_shuf = np.copy(X) # Get control X matrix which is a shuffled version of original X matrix
    np.random.shuffle(X_shuf)

    obj = NSP.analyse.analysis_chain(subject=subject,
                                     ydict=ydict, 
                                     X=X, 
                                     alpha=10, 
                                     voxeldict=voxeldict, 
                                     cv=5, 
                                     rois=rois, 
                                     X_uninformative=X_shuf, 
                                     fit_icept=False, 
                                     save_outs=True,
                                     regname=f'alexunet_layer{layer}')
    
    rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))

    rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

    NSP.analyse.plot_brain(prf_dict, 
                           roi_masks, 
                           subject, 
                           NSP.utils.cap_values(np.copy(rel_scores_np), None, None), 
                           False, 
                           save_img=True, 
                           img_path=f'/home/rfpred/imgs/reg/alexunet_layer{layer}_regcorplot.png')







# baseline_strings = ['rms', 'ce', 'sc_l']
    
# for feat in baseline_strings:
#     print(f'Running regression for baseline feature: {feat}')
#     X = NSP.stimuli.baseline_feats(feat)
#     print(f'X has these dimensions: {X.shape}')
#     X_shuf = np.copy(X)
#     np.random.shuffle(X_shuf)

#     obj = NSP.analyse.analysis_chain(subject=subject,
#                                      ydict=ydict, 
#                                      X=X, 
#                                      alpha=10, 
#                                      voxeldict=voxeldict, 
#                                      cv=5, 
#                                      rois=rois, 
#                                      X_uninformative=X_shuf, 
#                                      fit_icept=False, 
#                                      save_outs=True,
#                                      regname=feat)
    
#     # This is for the relative R scores.
#     rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))
#     rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)


#     # This is for the betas
#     plot_bets = np.hstack((obj[:,:3], obj[:,5].reshape(-1,1)))
#     plot_bets_np = NSP.utils.coords2numpy(plot_bets, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)
    
#     NSP.analyse.plot_brain(prf_dict, roi_masks, subject, NSP.utils.cap_values(np.copy(plot_bets_np), 0, 10), False, save_img=True, img_path=f'/home/rfpred/imgs/reg/{feat}_regcorplot.png')

# X = np.hstack((NSP.stimuli.baseline_feats(baseline_strings[0]), 
#                NSP.stimuli.baseline_feats(baseline_strings[1]), 
#                NSP.stimuli.baseline_feats(baseline_strings[2])))

# X_shuf = np.copy(X)
# np.random.shuffle(X_shuf)

# obj = NSP.analyse.analysis_chain(subject=subject,
#                                  ydict=ydict, 
#                                  X=X, 
#                                  alpha=10, 
#                                  voxeldict=voxeldict, 
#                                  cv=5, 
#                                  rois=rois, 
#                                  X_uninformative=X_shuf, 
#                                  fit_icept=False, 
#                                  save_outs=True,
#                                  regname='')

# rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))

# rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

# NSP.analyse.plot_brain(prf_dict, roi_masks, subject, NSP.utils.cap_values(np.copy(rel_scores_np), 0, 2), False, save_img=True, img_path='/home/rfpred/imgs/reg/bl_triple_regcorplot.png')

print('Het zit er weer op kameraad')



