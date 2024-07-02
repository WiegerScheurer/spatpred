import copy
import os
import pickle
import random
import re
import sys
import time
from importlib import reload
from math import e, sqrt
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import fnmatch
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats.mstats as mstats
import seaborn as sns
import sklearn as sk
import torch
import torchvision.models as models
import yaml
import joblib
from arrow import get
from colorama import Fore, Style
from IPython.display import display
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter, MaxNLocator,
                               MultipleLocator, NullFormatter)
import matplotlib.patches as patches
from nilearn import plotting
from PIL import Image
from scipy import stats
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from scipy.special import softmax
from scipy.stats import zscore as zs
from skimage import color
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)
from tqdm.notebook import tqdm

# print('soepstengesl')

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

import lgnpy.CEandSC.lgn_statistics
from lgnpy.CEandSC.lgn_statistics import LGN, lgn_statistics, loadmat

from unet_recon.inpainting import UNet


class VoxelSieve:
    """
    A class used to represent a Voxel Sieve, filtering out all voxels that do not
    meet all requirements defined in the VoxelSieve initiation method.

    Attributes
    ----------
    size : ndarray
        The size of the pRFs.
    ecc : ndarray
        The eccentricity of the pRFs.
    angle : ndarray
        The angle of the pRFs.
    prf_R2 : ndarray
        The R2 value of the voxel from pRF.
    nsd_R2 : ndarray
        The R2 value of the voxel from NSD.
    vox_pick : ndarray
        Boolean array indicating which voxels meet the criteria.
    xyz : ndarray
        The x, y, z coordinates of the selected voxels.
    r2mask : ndarray
        A boolean array that sets the top n NSD R2 voxels to True in case 
            fixed_n_voxels is an integer. This is used in situations where
            a fixed amount of voxels is wanted for each ROI, for example to
            check whether differences in effect are due to the amount of voxels.

    Methods
    -------
    __init__(self, prf_dict: Dict, roi_masks: Dict, NSP, subject: str, roi: str, max_size: float, min_size: float, patchbound: float, min_nsd_R2: int, min_prf_R2: int)
        Initializes the VoxelSieve instance with the given parameters.
    """

    def _get_peri_bounds(self, patchbound:float, peripheral_center:Optional[tuple], peri_angle:Optional[float], peri_ecc:Optional[float], leniency:float=0.0, verbose:bool=True):
        # Convert angles from degrees to radians
        peri_angle_rad = np.deg2rad(peri_angle)

        # Convert polar coordinates to Cartesian coordinates
        peri_x = peri_ecc * np.cos(peri_angle_rad)
        peri_y = peri_ecc * np.sin(peri_angle_rad)
        peripheral_center = (peri_x, peri_y)

        # Compute the distance from each voxel to the center of the patch
        voxel_x = self.ecc * np.cos(np.deg2rad(self.angle))
        voxel_y = self.ecc * np.sin(np.deg2rad(self.angle))
        distances = np.sqrt((voxel_x - peripheral_center[0])**2 + (voxel_y - peripheral_center[1])**2)

        # Select the voxels for which the distance plus the size times (1 - leniency) is less than or equal to the patchbound
        selected_voxels = distances + self.size * (1 - leniency) <= patchbound

        return selected_voxels
    
    def vox_lim(self, cutoff:int):
        """Method to restrict the number of voxels after VoxelSieve class initiation.

        Args:
        - self (VoxelSieve instance): Existing voxel selection.
        - cutoff (int): The maximum number of voxels to be included in the voxel selection. 
        """        
        roi_voxels = np.ones(np.sum(self.vox_pick)).astype(bool)
        r2raw_arr = self.nsd_R2[roi_voxels]
        
        # Adjust the mask based on the top n NSD R2 voxels, so cutoffs don't cut off good voxels
        top_n = cutoff
        indices = np.argsort(r2raw_arr)
        topices = indices[-top_n:]
        r2mask = np.zeros_like(r2raw_arr, dtype=bool)
        r2mask[topices] = True
        
        # Apply the vox_pick mask to all the attributes with voxel specific data
        self.size = self.size[r2mask]
        self.ecc = self.ecc[r2mask]
        self.angle = self.angle[r2mask]
        self.prf_R2 = self.prf_R2[r2mask]
        self.nsd_R2 = self.nsd_R2[r2mask]
        self.sigmas = self.sigmas[r2mask]
        self.ycoor = self.ycoor[r2mask]
        self.xcoor = self.xcoor[r2mask]
        self.xyz = self.xyz[r2mask]
        self.vox_pick = np.zeros_like(self.vox_pick, dtype=bool)
        self.vox_pick[topices] = True
        
                
    def __init__(self, NSP, prf_dict:Dict, roi_masks:Dict, subject:str, roi:str, patchloc:str='central',
                 max_size:Optional[float]=None, min_size:Optional[float]=None, patchbound:Optional[float]=None, 
                 min_nsd_R2:Optional[int]=None, min_prf_R2:Optional[int]=None, print_attributes:bool=True, 
                 fixed_n_voxels:Optional[Union[str, int]]=None, 
                 peripheral_center:Optional[tuple]=None, peri_angle:Optional[float]=None, peri_ecc:Optional[float]=None, 
                 leniency:Optional[float]=None, verbose:bool=True):
        """ TODO: Write this 

        Args:
            NSP (_type_): _description_
            prf_dict (Dict): _description_
            roi_masks (Dict): _description_
            subject (str): _description_
            roi (str): _description_
            patchloc (str, optional): _description_. Defaults to 'central'.
            max_size (Optional[float], optional): _description_. Defaults to None.
            min_size (Optional[float], optional): _description_. Defaults to None.
            patchbound (Optional[float], optional): _description_. Defaults to None.
            min_nsd_R2 (Optional[int], optional): _description_. Defaults to None.
            min_prf_R2 (Optional[int], optional): _description_. Defaults to None.
            print_attributes (bool, optional): _description_. Defaults to True.
            fixed_n_voxels (Optional[Union[str, int]], optional): _description_. Defaults to None.
            peripheral_center (Optional[tuple], optional): _description_. Defaults to None.
            peri_angle (Optional[float], optional): _description_. Defaults to None.
            peri_ecc (Optional[float], optional): _description_. Defaults to None.
            leniency (Optional[float], optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to True.
        """
        self.patchbound = patchbound
        self.figdims = (425, 425) # Raw image size of NSD stimuli

        self.size = prf_dict[subject]['proc'][f'{roi}_mask']['size'][:,3]
        self.ecc = prf_dict[subject]['proc'][f'{roi}_mask']['eccentricity'][:,3]
        self.angle = prf_dict[subject]['proc'][f'{roi}_mask']['angle'][:,3]
        self.prf_R2 = prf_dict[subject]['proc'][f'{roi}_mask']['R2'][:,3]
        self.nsd_R2 = NSP.cortex.nsd_R2_dict(roi_masks, glm_type='hrf')[subject]['R2_roi'][f'{roi}_mask'][:,3]
        self.sigmas, self.ycoor, self.xcoor = NSP.cortex.calculate_pRF_location(self.size, self.ecc, self.angle, self.figdims)
        
        central_coords = int((self.figdims[0] + 1) / 2) # The x, and y for the central patch centre

        if fixed_n_voxels == 'all': # If all_voxels is True, all ROI voxels are selected regardless of the other parameters
            self.vox_pick = np.ones(len(self.size)).astype(bool)
            patch_x = patch_y = central_coords

        elif patchloc == 'central':
            patch_x = patch_y = central_coords
        #                      RF not too large    &     RF within patch boundary      &    RF not too small    &    NSD R2 high enough      &    pRF R2 high enough
            self.vox_pick = (self.size < max_size) & (self.ecc+self.size < patchbound) & (self.size > min_size) & (self.nsd_R2 > min_nsd_R2) & (self.prf_R2 > min_prf_R2)

        elif patchloc == 'peripheral':
            if peripheral_center is None and peri_angle and peri_ecc is not None:
                # patchloc_triangle_s = peri_ecc
                peri_y = round(peri_ecc * np.sin(np.radians(peri_angle)), 2)
                peri_x = round(peri_ecc * np.cos(np.radians(peri_angle)), 2)
                peripheral_center = (peri_x, peri_y)
                if verbose:
                    print(f'Peripheral center at {peripheral_center}')
            leniency = 0.0 if leniency is None else leniency
            self.vox_pick = self._get_peri_bounds(patchbound, peripheral_center, peri_angle, peri_ecc, leniency, verbose)
            patch_x = central_coords + peripheral_center[0] * (self.figdims[0]/8.4) # in pixels
            patch_y = central_coords + peripheral_center[1] * (self.figdims[0]/8.4)
        
        # Apply the vox_pick mask to all the attributes with voxel specific data
        self.patchcoords = (patch_x, patch_y) # Matrix indexing
        self.size = self.size[self.vox_pick]
        self.ecc = self.ecc[self.vox_pick]
        self.angle = self.angle[self.vox_pick]
        self.prf_R2 = self.prf_R2[self.vox_pick]
        self.nsd_R2 = self.nsd_R2[self.vox_pick]
        self.sigmas = self.sigmas[self.vox_pick]
        self.ycoor = self.ycoor[self.vox_pick]
        self.xcoor = self.xcoor[self.vox_pick]
        self.xyz = prf_dict[subject]['proc'][f'{roi}_mask']['size'][:, :3][self.vox_pick].astype(int)
        if fixed_n_voxels == "all":
            self.patchmask = np.ones((self.figdims[0], self.figdims[1])).astype(bool)
            self.patchbound = 10 # arbitrary, needs to be larger than img
        else:
            self.patchmask = np.flip(NSP.utils.make_circle_mask(self.figdims[0], patch_y, patch_x, patchbound * (425 / 8.4), fill='y'), 0).astype(bool)
        
        if type(fixed_n_voxels) == int:
            self.vox_lim(fixed_n_voxels)
        
        self.attributes = [attr for attr in dir(self) if not attr.startswith('_')] # Filter out both the 'dunder' and hidden methods
        
        print(f'{roi} voxels that fulfill requirements: {Fore.LIGHTWHITE_EX}{len(self.size)}{Style.RESET_ALL} out of {Fore.LIGHTWHITE_EX}{len(prf_dict[subject]["proc"][f"{roi}_mask"]["size"])}{Style.RESET_ALL}.')
        
        if print_attributes:
            print('\nClass contains the following attributes:')
            for attr in self.attributes:
                print(f"{Fore.BLUE} .{attr}{Style.RESET_ALL}")  
            print('\n')          