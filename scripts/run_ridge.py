#!/usr/bin/env python3

import os
import sys

# from regex import F

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

subject = 'subj01'
max_size = 2
min_size = .2
patchbound = 1.5
min_nsd_R2 = 25
min_prf_R2 = 0
fixed_n_voxels = 515

voxeldict = {}
for roi in rois:
    print_attr = True if roi == rois[len(rois)-1] else False
    voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
                                subject=subject, 
                                roi=roi, 
                                max_size=max_size, 
                                min_size=min_size, 
                                patchbound=patchbound, 
                                min_nsd_R2=min_nsd_R2, 
                                min_prf_R2=min_prf_R2,
                                print_attributes=print_attr,
                                fixed_n_voxels=fixed_n_voxels)

ydict = {}
for roi in rois:
    ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
    print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')

# Define the baseline model, which is the rms, ce and sc_l features.
# TODO: Also include alexnet featmaps?
baseline_strings = ['rms', 'ce', 'sc_l']

Xbl = np.hstack((NSP.stimuli.baseline_feats(baseline_strings[0]), 
               NSP.stimuli.baseline_feats(baseline_strings[1]), 
               NSP.stimuli.baseline_feats(baseline_strings[2])))

    
Xpred = NSP.stimuli.unpred_feats(content=True, style=False, ssim=False, pixel_loss=False, L1=True, MSE=False, verbose=True, outlier_sd_bound=5)

for layer in range(0, 5):
    Xalex = Xpred[:,layer].reshape(-1,1)
    X = np.hstack((Xbl, Xalex))
    print(f'X has these dimensions: {X.shape}')
    X_shuf = np.copy(X)
    np.random.shuffle(X_shuf)

    obj = NSP.analyse.analysis_chain(subject=subject,
                                     ydict=ydict, 
                                     X=X, # The baseline model + current unpredictability layer
                                     alpha=10, 
                                     voxeldict=voxeldict, 
                                     cv=5, 
                                     rois=rois, 
                                     X_uninformative=Xbl, # The baseline model
                                     fit_icept=False, 
                                     save_outs=True,
                                     regname=f'unpred_lay{layer}NEW')
    
    # This is for the relative R scores.
    rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))
    rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

    # plot the relative R scores
    NSP.analyse.plot_brain(prf_dict, 
                           roi_masks, 
                           subject, 
                           NSP.utils.cap_values(np.copy(rel_scores_np), 0, 10), 
                           False, 
                           save_img=True, 
                           img_path=f'/home/rfpred/imgs/reg/unpred_lay{layer}_regcorplotNEWER.png')

    # This is for the betas
    plot_bets = np.hstack((obj[:,:3], obj[:,5].reshape(-1,1)))
    plot_bets_np = NSP.utils.coords2numpy(plot_bets, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

    # plot the betas
    NSP.analyse.plot_brain(prf_dict, 
                           roi_masks, 
                           subject, 
                           NSP.utils.cap_values(np.copy(plot_bets_np), 0, 10), 
                           False, 
                           save_img=True, 
                           img_path=f'/home/rfpred/imgs/reg/unpred_lay{layer}_regbetaplotNEWER.png')


for layer in range(0,5):
    delta_r_layer = pd.read_pickle(f'{NSP.own_datapath}/{subject}/brainstats/unpred_lay{layer}NEW_delta_r.pkl').values[0].flatten()
    if layer == 0:
        all_delta_r = delta_r_layer
    else:
        all_delta_r = np.vstack((all_delta_r, delta_r_layer))
        
df = (pd.DataFrame(all_delta_r, columns = rois))
print(df)

plt.clf()

# Reset the index of the DataFrame to use it as x-axis
df.reset_index(inplace=True)

# Melt the DataFrame to long-form or tidy format
df_melted = df.melt('index', var_name='ROI', value_name='b')


# Create the line plot
sns.lineplot(x='index', y='b', hue='ROI', data=df_melted, marker='o')

plt.xticks(range(5))  # Set x-axis ticks to be integers from 0 to 4
plt.xlabel('Alexnet Layer')
plt.ylabel('Delta R Value')
plt.title('Delta R Value per Alexnet Layer')

# Save the plot
plt.savefig(f'{NSP.own_datapath}/{subject}/brainstats/unpred_delta_r_plot.png')

plt.show()
plt.clf()

voxel_assignment = {}

for layer in range(0,5): # Loop over the layers of the alexnet
    cordict, coords = NSP.analyse.load_regresults(subject, prf_dict, roi_masks, 'unpred', f'{str(layer)}NEW', plot_on_viscortex=False, plot_result='r', verbose=False)
    if layer == 0:
        all_betas = np.hstack((np.array(coords)[:,:3], np.array(coords)[:,5].reshape(-1,1)))
    else:
        all_betas = np.hstack((all_betas, np.array(coords)[:,5].reshape(-1,1)))
        
for n_roi, roi in enumerate(rois):
    n_roivoxels = len(cordict['X'][roi][0])
    
    if roi == 'V1':
        vox_of_roi = np.ones((n_roivoxels, 1))
    else:
        vox_of_roi = (np.vstack((vox_of_roi, (np.ones((n_roivoxels, 1))* (n_roi + 1))))).astype(int)

all_betas_voxroi = np.hstack((all_betas, vox_of_roi))[:,3:]
all_betas_voxroi[:,:5] = stats.zscore(all_betas_voxroi[:,:5], axis=0)

# Get the index of the maximum value in each row, excluding the last column
max_indices = np.argmax(all_betas_voxroi[:, :-1], axis=1)

# print(max_indices)

# Create a DataFrame from the array
df = pd.DataFrame(all_betas_voxroi, columns=[f'col_{i}' for i in range(all_betas_voxroi.shape[1])])

# Rename the last column to 'ROI'
df.rename(columns={df.columns[-1]: 'ROI'}, inplace=True)

# Add the max_indices as a new column
df['AlexNet layer'] = max_indices

# Convert the 'ROI' column to int for plotting
df['ROI'] = df['ROI'].astype(int)

# # Plot the distribution of max_indices for each ROI
# sns.countplot(x='max_index', hue='ROI', data=df)

# plt.show()

# # Plot the distribution of ROIs for each max_index
# sns.countplot(x='ROI', hue='max_index', data=df)

# plt.show()

# Calculate the proportions of max_indices within each ROI
df_prop = (df.groupby('ROI')['AlexNet layer']
             .value_counts(normalize=True)
             .unstack(fill_value=0))

# Plot the proportions using a stacked bar plot
df_prop.plot(kind='bar', stacked=True)


# Save the plot
plt.savefig(f'{NSP.own_datapath}/{subject}/brainstats/unpred_layassign.png')

plt.show()


################### FULL VISUAL CORTEX (V1, V2, V3, V4) REGRESSIONS ###############

# subject = 'subj01'

# voxeldict = {}
# for roi in rois:
#     print_attr = True if roi == rois[len(rois)-1] else False
#     voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
#                                 subject=subject, 
#                                 roi=roi, 
#                                 print_attributes=print_attr,
#                                 all_voxels=True)


# # subject = 'subj01'
# # max_size = 2
# # min_size = .5
# # patchbound = 1.5
# # min_nsd_R2 = 20
# # min_prf_R2 = 0

# # voxeldict = {}
# # for roi in rois:
# #     print_attr = True if roi == rois[len(rois)-1] else False
# #     voxeldict[roi] = VoxelSieve(NSP, prf_dict, roi_masks,
# #                                 subject=subject, 
# #                                 roi=roi, 
# #                                 max_size=max_size, 
# #                                 min_size=min_size, 
# #                                 patchbound=patchbound, 
# #                                 min_nsd_R2=min_nsd_R2, 
# #                                 min_prf_R2=min_prf_R2,
# #                                 print_attributes=print_attr,
# #                                 all_voxels=False)

# ydict = {}

# for roi in rois:
#     ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
#     print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')
    
# for layer in range(2, 5):
#     print(f'Running regression for layer: {layer}')
    
#     X = NSP.stimuli.unet_featmaps(list_layers=[layer], scale='full') # Get X matrix
#     print(f'X has these dimensions: {X.shape}')
#     X_shuf = np.copy(X) # Get control X matrix which is a shuffled version of original X matrix
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
#                                      regname=f'alexunet_layer{layer}')
    
#     rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))

#     rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

#     NSP.analyse.plot_brain(prf_dict, 
#                            roi_masks, 
#                            subject, 
#                            NSP.utils.cap_values(np.copy(rel_scores_np), None, None), 
#                            False, 
#                            save_img=True, 
#                            img_path=f'/home/rfpred/imgs/reg/alexunet_layer{layer}_regcorplot.png')



# # baseline_strings = ['rms', 'ce', 'sc_l']
    
# # for feat in baseline_strings:
# #     print(f'Running regression for baseline feature: {feat}')
# #     X = NSP.stimuli.baseline_feats(feat)
# #     print(f'X has these dimensions: {X.shape}')
# #     X_shuf = np.copy(X)
# #     np.random.shuffle(X_shuf)

# #     obj = NSP.analyse.analysis_chain(subject=subject,
# #                                      ydict=ydict, 
# #                                      X=X, 
# #                                      alpha=10, 
# #                                      voxeldict=voxeldict, 
# #                                      cv=5, 
# #                                      rois=rois, 
# #                                      X_uninformative=X_shuf, 
# #                                      fit_icept=False, 
# #                                      save_outs=True,
# #                                      regname=feat)
    
# #     # This is for the relative R scores.
# #     rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))
# #     rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)


# #     # This is for the betas
# #     plot_bets = np.hstack((obj[:,:3], obj[:,5].reshape(-1,1)))
# #     plot_bets_np = NSP.utils.coords2numpy(plot_bets, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)
    
# #     NSP.analyse.plot_brain(prf_dict, roi_masks, subject, NSP.utils.cap_values(np.copy(plot_bets_np), 0, 10), False, save_img=True, img_path=f'/home/rfpred/imgs/reg/{feat}_regcorplot.png')

# # X = np.hstack((NSP.stimuli.baseline_feats(baseline_strings[0]), 
# #                NSP.stimuli.baseline_feats(baseline_strings[1]), 
# #                NSP.stimuli.baseline_feats(baseline_strings[2])))

# # X_shuf = np.copy(X)
# # np.random.shuffle(X_shuf)

# # obj = NSP.analyse.analysis_chain(subject=subject,
# #                                  ydict=ydict, 
# #                                  X=X, 
# #                                  alpha=10, 
# #                                  voxeldict=voxeldict, 
# #                                  cv=5, 
# #                                  rois=rois, 
# #                                  X_uninformative=X_shuf, 
# #                                  fit_icept=False, 
# #                                  save_outs=True,
# #                                  regname='')

# # rel_obj = np.hstack((obj[:,:3], (obj[:,3] - obj[:,4]).reshape(-1,1)))

# # rel_scores_np = NSP.utils.coords2numpy(rel_obj, roi_masks[subject][f'{roi}_mask'].shape, keep_vals=True)

# # NSP.analyse.plot_brain(prf_dict, roi_masks, subject, NSP.utils.cap_values(np.copy(rel_scores_np), 0, 2), False, save_img=True, img_path='/home/rfpred/imgs/reg/bl_triple_regcorplot.png')

print('Het zit er weer op kameraad')



