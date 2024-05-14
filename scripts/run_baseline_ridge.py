#!/usr/bin/env python3

import os
import sys

os.environ["OMP_NUM_THREADS"] = "10"
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
from scipy import stats
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
# voxeldict = {}
# for roi in rois:
#     print_attr = True if roi == rois[len(rois)-1] else False
#     voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
#                                 subject=subject, 
#                                 roi=roi,
#                                 print_attributes=print_attr,
#                                 fixed_n_voxels='all')

# ydict = {}
# for roi in rois:
#     ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
#     print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')



# for layer in range(0, Xpred.shape[1]):
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
#                                      cv=10, 
#                                      rois=rois, 
#                                      X_uninformative=Xbl, # The baseline model
#                                      fit_icept=False, 
#                                      save_outs=True,
#                                      regname=f'{which_cnn}_unpred_lay{layer}')
    
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
#                            img_path=f'/home/rfpred/imgs/reg/{which_cnn}_unpred_lay{layer}_regcorplot.png')

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
#                            img_path=f'/home/rfpred/imgs/reg/{which_cnn}_unpred_lay{layer}_regbetaplot.png')


# for layer in range(0,Xpred.shape[1]):
#     delta_r_layer = pd.read_pickle(f'{NSP.own_datapath}/{subject}/brainstats/{which_cnn}_unpred_lay{layer}_delta_r.pkl').values[0].flatten()
#     if layer == 0:
#         all_delta_r = delta_r_layer
#     else:
#         all_delta_r = np.vstack((all_delta_r, delta_r_layer))
        
# df = (pd.DataFrame(all_delta_r, columns = rois))
# print(df)

# plt.clf()

# # Reset the index of the DataFrame to use it as x-axis
# df.reset_index(inplace=True)

# # Melt the DataFrame to long-form or tidy format
# df_melted = df.melt('index', var_name='ROI', value_name='b')

# # Create the line plot
# sns.lineplot(x='index', y='b', hue='ROI', data=df_melted, marker='o')

# plt.xticks(range(5))  # Set x-axis ticks to be integers from 0 to 4
# plt.xlabel(f'{which_cnn} Layer')
# plt.ylabel('Delta R Value')
# plt.title(f'Delta R Value per {which_cnn} Layer')

# # Save the plot
# plt.savefig(f'{NSP.own_datapath}/{subject}/brainstats/{which_cnn}_unpred_delta_r_plot.png')

# plt.show()
# plt.clf()

################### FULL VISUAL CORTEX (V1, V2, V3, V4) REGRESSIONS ###############

subject = 'subj01'
tag = '_nieuw'
voxeldict = {}
for roi in rois:
    print_attr = True if roi == rois[len(rois)-1] else False
    voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
                                subject=subject, 
                                roi=roi,
                                print_attributes=print_attr,
                                fixed_n_voxels='all')

ydict = {}
for roi in rois:
    ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
    print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')

# baseline_strings = ['rms', 'ce', 'sc_l']

baseline_strings = ['rms', 'ce', 'sc']

rms = NSP.stimuli.get_rms(subject)
sc = NSP.stimuli.get_scce(subject, 'sc')
ce = NSP.stimuli.get_scce(subject, 'ce')
baseline = pd.concat([rms, sc, ce], axis=1)

for feat_no in range(0,4):    
    feat = baseline_strings[feat_no] if feat_no < 3 else baseline_strings
    print(f'Running regression for baseline feature: {feat}')
    X = baseline[feat].values.reshape(-1,1) if feat_no < 3 else baseline[feat].values
    print(f'X has these dimensions: {X.shape}')
    X_shuf = np.copy(X)
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
                                     regname=f'{feat}{tag}')
    
    rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))

    rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

    # Plot relative R scores
    NSP.analyse.plot_brain(prf_dict, 
                           roi_masks, 
                           subject, 
                           NSP.utils.cap_values(np.copy(rel_scores_np), None, None), 
                           False, 
                           save_img=True, 
                           img_path=f'/home/rfpred/imgs/reg/{feat}_regcorplot{tag}.png')

    # This is for the betas
    plot_bets = np.hstack((obj[:,:3], obj[:,5].reshape(-1,1)))
    plot_bets_np = NSP.utils.coords2numpy(plot_bets, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

    # plot the betas
    NSP.analyse.plot_brain(prf_dict, 
                           roi_masks, 
                           subject, 
                           NSP.utils.cap_values(np.copy(plot_bets_np), None, None), 
                           False, 
                           save_img=True, 
                           img_path=f'/home/rfpred/imgs/reg/{feat}_regbetaplot{tag}.png')


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



