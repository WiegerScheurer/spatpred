import os
import sys

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



print(sys.path)

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

import pickle
import matplotlib.pyplot as plt

def plot_scores(ydict, X, alpha, cv, rois, X_uninformative, fit_icept:bool=False):
    r_values = {}
    r_uninformative = {}
    cor_scores_dict = {}  # Dictionary to store cor_scores

    # Calculate scores for the given X
    for roi in rois:
        y = ydict[roi]
        model = NSP.analyse.run_ridge_regression(X, y, alpha=alpha, fit_icept=False)
        _, cor_scores = NSP.analyse.score_model(X, y, model, cv=cv)
        r_values[roi] = np.mean(cor_scores, axis=0)
        cor_scores_dict[roi] = cor_scores  # Save cor_scores to dictionary

        xyz = voxeldict[roi].xyz
        this_coords = np.hstack((xyz, np.array(r_values[roi]).reshape(-1,1)))
        if roi == 'V1':
            coords = this_coords
        else:
            coords = np.vstack((coords, this_coords))

    # Calculate scores for the uninformative X
    for roi in rois:
        y = ydict[roi]
        model = NSP.analyse.run_ridge_regression(X_uninformative, y, alpha=alpha, fit_icept=fit_icept)
        _, cor_scores = NSP.analyse.score_model(X_uninformative, y, model, cv=cv)
        r_uninformative[roi] = np.mean(cor_scores, axis=0)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Assuming rois is a list with at least 4 elements
    for i, roi in enumerate(rois[:4]):
        # Underlay with the histogram of r_uninformative[roi] values
        axs[i].hist(r_uninformative[roi], bins=40, edgecolor='black', alpha=0.5, label='Uninformative X')
        # Plot the histogram of r_values[roi] values in the i-th subplot
        axs[i].hist(r_values[roi], bins=40, edgecolor='black', alpha=0.5, label='X')
        axs[i].set_title(f'R values for {roi}')
        axs[i].legend()

    # Display the figure
    plt.tight_layout()
    plt.savefig('HEREplot.png')  # Save the plot to a file
    plt.show()

    # Save cor_scores to a file
    with open('HEREcor_scores.pkl', 'wb') as f:
        pickle.dump(cor_scores_dict, f)

    return coords

rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(verbose=False)
prf_dict = NSP.cortex.prf_dict(rois, roi_masks)

subject = 'subj01'
max_size = 1000
min_size = 0
patchbound = 1000
min_nsd_R2 = 0
min_prf_R2 = 0

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
                                print_attributes=print_attr)

ydict = {}
for roi in rois:
    ydict[roi] = NSP.analyse.load_y(subject=subject, roi=roi, voxelsieve=voxeldict[roi], n_trials='all').T
    print(f'{roi} y-matrix has dimensions: {ydict[roi].shape}')
    
Xrms = NSP.stimuli.baseline_feats('rms')
Xce = NSP.stimuli.baseline_feats('ce')
Xsc = NSP.stimuli.baseline_feats('sc_l') # the _l attachment is for 'large' -> computed feature for 5° radius patch

X = NSP.stimuli.unet_featmaps(list_layers=[3], scale='full')

X_shuf = np.copy(X)
np.random.shuffle(X_shuf)

obj = plot_scores(ydict, X, alpha=10, cv=5, rois=rois, X_uninformative=X_shuf, fit_icept=False)



