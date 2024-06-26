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



class Stimuli():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    # Function to show a randomly selected image of the nsd dataset
    def show_stim(self, img_no='random', small:bool=False, hide:bool=False, crop:bool=False):
        # Example code to show how to access the image files, these are all 73000 of them, as np.arrays
        # I keep it like this as it might be useful to also store the reconstructed images with the autoencoder
        # using a .hdf5 folder structure, but I can change this later on.
            
        stim_dir = f'{self.nsp.nsd_datapath}/nsddata_stimuli/stimuli/nsd/'
        stim_files = [f for f in os.listdir(stim_dir) if os.path.isfile(os.path.join(stim_dir, f))]

        with h5py.File(f'{stim_dir}{stim_files[0]}', 'r') as file:
            
            img_brick_dataset = file['imgBrick']
            
            if img_no == 'random':
                image_no = random.randint(0,img_brick_dataset.shape[0])
            else: image_no = img_no
            if crop: test_image = img_brick_dataset[image_no][163:263,163:263]
            else: test_image = img_brick_dataset[image_no]
        hor = ver = 10
        if small:
            hor = ver = 5        
        if hide is not True:
            plt.figure(figsize=(hor, ver))
            plt.imshow(test_image)
            plt.title(f'Image number {image_no}')
            plt.axis('off')
            plt.show()
            
        return test_image, image_no

    def mask_img(self, img_no:(str | int)="random", radius:int=1, small:bool=True):
        """
        Apply a circular mask to an image.

        Parameters:
        - img_no: str or int, optional (default: "random")
            The image number or label to apply the mask to. If "random", a random image will be selected.
        - radius: int, optional (default: 1)
            The radius of the circular mask.

        Returns:
        - masked_img: np.ndarray, the masked image
        """
        img = self.show_stim(img_no=img_no, small=False, hide=True, crop=False)[0]

        mask = self.nsp.utils.make_circle_mask(
            425, 213, 213, radius * (425 / 8.4), fill="y", margin_width=1
        ).reshape((425, 425))
        mask_3d = np.dstack([mask] * 3)

        masked_img = img * mask_3d

        smallfactor = 2 if small else 1
                    
        # Create a new figure with a larger size
        plt.figure(figsize=(10/smallfactor, 10/smallfactor))

        plt.imshow(masked_img)

        plt.axis('off')
        
        return masked_img
        
    def calc_rms_contrast_lab(self, rgb_image:np.ndarray, mask_w_in:(np.ndarray | None)=None, rf_mask_in:(np.ndarray | None)=None, 
                            normalise:bool=True, plot:bool=False, cmap:str='gist_gray', 
                            crop_post:bool=False, nsd_idx:(int | None)=None, lab_idx:int=0, 
                            cropped_input:bool=True) -> float:
        """"
        Function that calculates Root Mean Square (RMS) contrast after converting RGB to LAB, 
        which follows the CIELAB colour space. This aligns better with how visual input is
        processed in human visual cortex.

        Arguments:
            rgb_image (np.ndarray): Input RGB image
            mask_w_in (np.ndarray): Weighted mask
            rf_mask_in (np.ndarray): RF mask
            normalise (bool): If True, normalise the input array, default True
            plot (bool): If True, plot the square contrast and weighted square contrast, default False
            cmap (str): Matplotlib colourmap for the plot, default 'gist_gray'
            crop_post (bool): If True, crop the image after calculation (to enable comparison of
                RMS values to images cropped prior to calculation), default False
            nsd_idx (int | None): Optional selection of a specific natural scene image (from the NSD dataset), default None
            lab_idx (int): Optional selection of different LAB channels, default 0
            cropped_input (bool): If True, the input image is already cropped, default True

        Returns:
            float: Root Mean Square visual contrast of input img
        """
        x_min, x_max, y_min, y_max = self.nsp.utils.get_bounding_box(rf_mask_in)
        
        # Convert RGB image to LAB colour space
        lab_image = color.rgb2lab(rgb_image)
        
        # First channel [0] is Luminance, second [1] is green-red, third [2] is blue-yellow
        ar_in = lab_image[:, :, lab_idx] # Extract the L channel for luminance values, assign to input array

        if cropped_input is False:
            ar_in = ar_in[y_min:y_max, x_min:x_max]
            mask_w_in = mask_w_in[y_min:y_max, x_min:x_max]
            rf_mask_in = rf_mask_in[y_min:y_max, x_min:x_max]
        
        mask_w_in = np.ones((rgb_image.shape[:-1])) if mask_w_in is None else mask_w_in
        rf_mask_in = np.ones((rgb_image.shape[:-1])) if rf_mask_in is None else rf_mask_in
        
        if normalise:
            ar_in /= ar_in.max()
            
         # Difference between each pixel luminance and mean patch luminance
        square_contrast = np.square(ar_in - ar_in[rf_mask_in].mean())
        msquare_contrast = (mask_w_in * square_contrast).sum()
        
        
        # square_contrast = square_contrast[y_min:y_max, x_min:x_max]
        # mask_w_in = mask_w_in[y_min:y_max, x_min:x_max]

        if crop_post:     
            square_contrast = square_contrast[y_min:y_max, x_min:x_max]
            mask_w_in = mask_w_in[y_min:y_max, x_min:x_max]
        
        if nsd_idx is None:
            raw_img = rgb_image
            idx_str = ''
        else:
            raw_img = self.show_stim(img_no=nsd_idx, small=True, hide=True)[0]
            idx_str = f' {nsd_idx}'
        
        rms = np.sqrt(msquare_contrast)
        
        if plot:
            _, axs = plt.subplots(1, 4, figsize=(20, 5))
            plt.subplots_adjust(wspace=0.01)
            axs[0].imshow(raw_img)
            axs[0].axis('off')
            axs[0].set_title(f'Natural scene{idx_str}', fontsize=18)
            
            axs[1].imshow(rgb_image[y_min:y_max, x_min:x_max]) # Because rows,cols
            axs[1].axis('off')
            
            axs[2].imshow(square_contrast, cmap=cmap)
            axs[2].axis('off') 
            
            axs[3].imshow(mask_w_in * square_contrast, cmap=cmap)
            axs[3].set_title(f'RMS = {rms:.2f}', fontsize=18)
            axs[3].axis('off') 
            
        return rms
    
    # These two functions are coupled to run the feature computations in parallel.
    # This saves a lot of time. Should be combined with the feature_df function to assign
    # the values to the corresponding trials.
    def rms_single(self, args, ecc_max:int = 1, loc:str='center', plot_original:bool=False, plot_contrast:bool=False, 
                   crop_prior:bool = False, crop_post:bool = False, save_plot:bool = False, cmap:str='gist_gray', normalise:bool=True,lab_idx:int=0):
        
        i, start, n, plot_original, plot_contrast, loc, crop_prior, crop_post, save_plot = args
        dim = self.nsp.stimuli.show_stim(hide=True)[0].shape[0]
        radius = ecc_max * (dim / 8.4)
        if loc == 'center':
            x = y = (dim + 1)/2
        elif loc == 'irrelevant_patch':
            x = y = radius + 10
            
        mask_w_in = self.nsp.utils.css_gaussian_cut(dim, x, y, radius).reshape((425,425))
        rf_mask_in = self.nsp.utils.make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
        ar_in = self.nsp.stimuli.show_stim(img_no=i, hide=bool(np.abs(plot_original-1)), small=True)[0]  
        
        if i % 100 == 0:
            print(f"Processing image number: {i} out of {n + start}")
            
        if crop_prior:
            
            x_min, x_max, y_min, y_max = self.nsp.utils.get_bounding_box(rf_mask_in)
            
            ar_in = ar_in[x_min:x_max, y_min:y_max]
            mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
            rf_mask_in = rf_mask_in[x_min:x_max, y_min:y_max]
            
        return self.calc_rms_contrast_lab(ar_in,  mask_w_in, rf_mask_in, normalise=normalise, 
                                    plot=plot_contrast, cmap=cmap, crop_post=crop_post, nsd_idx=i, lab_idx=lab_idx)
        
    # This function is paired with rms_single to mass calculate the visual features using parallel computation.
    def rms_all(self, start, n, ecc_max = 1, plot_original:bool=False, plot_contrast:bool=True, loc = 'center', crop_prior:bool = False, crop_post:bool = True, save_plot:bool = False):
        img_vec = list(range(start, start + n))

        # Create a pool of worker processes
        with Pool() as p:
            rms_vec = p.map(self.rms_single, [(i, start, n, plot_original, plot_contrast, loc, crop_prior, crop_post, save_plot) for i in img_vec])

        rms_dict = pd.DataFrame({
            'rms': rms_vec
        })

        rms_dict = rms_dict.set_index(np.array(img_vec))
        return rms_dict


    def get_scce_contrast(
        self,
        rgb_image,
        plot="n",
        cmap="gist_gray",
        crop_prior: bool = False,
        crop_post: bool = False,
        save_plot: bool = False,
        return_imfovs: bool = True,
        imfov_overlay: bool = False,
        config_path:(str | None)=None,
        lgn_instance:(LGN | None)=None,
        patch_center:(tuple | None)=None,
        deg_per_pixel:(float | None)=None
    ):
        if config_path is None:
            config_path = '/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/default_config.yml'

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        lgn = LGN(config=config, default_config_path=config_path)

        threshold_lgn = loadmat(filepath='/home/rfpred/notebooks/alien_nbs/lgnpy/ThresholdLGN.mat')['ThresholdLGN']

        lgn_out = lgn_statistics(
            im=rgb_image,
            file_name="noname.tiff",
            config=config,
            force_recompute=True,
            cache=False,
            home_path="./notebooks/alien_nbs/",
            verbose=False,
            verbose_filename=False,
            threshold_lgn=threshold_lgn,
            compute_extra_statistics=False,
            crop_prior=True,
            plot_imfovs=False,
            return_imfovs=return_imfovs,
            lgn_instance=lgn_instance,
            patch_center=patch_center
        )

        ce = np.mean(lgn_out[0][:, :, 0])
        sc = np.mean(lgn_out[1][:, :, 0])
        beta = np.mean(lgn_out[2][:, :, 0])
        gamma = np.mean(lgn_out[3][:, :, 0])
        edge_dict = lgn_out[4]
        imfovbeta = lgn_out[5]
        imfovgamma = lgn_out[6]

        patch_x, patch_y = self.nsp.utils.get_circle_center(imfovbeta)

        if imfov_overlay:
            _, beta_rad = self.nsp.utils._get_circle_outline(imfovbeta, deg_per_pixel=deg_per_pixel, patch_center=(patch_x, patch_y))
            _, gamma_rad = self.nsp.utils._get_circle_outline(imfovgamma, deg_per_pixel=deg_per_pixel, patch_center=(patch_x, patch_y))
        
        if plot == "y":
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            plt.subplots_adjust(wspace=0.01, hspace=0.01)

            images = ["par1", "par2", "par3", "mag1", "mag2", "mag3"]
            for i, img_name in enumerate(images):
                img = lgn_out[4][img_name]

                axs[i // 3, i % 3].imshow(img, cmap=cmap)
                if imfov_overlay:
                    imfov_x = patch_x
                    imfov_y = patch_y
                    circle = (imfov_x, imfov_y, beta_rad if "par" in img_name else gamma_rad)
                    # Create a circle patch
                    circ = patches.Circle((circle[0], circle[1]), circle[2], edgecolor='r', facecolor='none', linewidth=2.4)
                    # Add the patch to the axes
                    axs[i // 3, i % 3].add_patch(circ)
                axs[i // 3, i % 3].axis("off")

            plt.tight_layout()

            if save_plot:
                fig.savefig(
                    f"rms_crop_prior_{str(crop_prior)}_crop_post_{str(crop_post)}.png"
                )

        return ce, sc, beta, gamma, imfovbeta, imfovgamma, edge_dict

    def scce_single(
        self,
        args,
        ecc_max=2.8,
        loc="center",
        plot="n",
        cmap: str = "gist_gray",
        return_imfovs: bool = False,
        imfov_overlay: bool = False,
        config_path:(str | None)=None,
        lgn_instance:(LGN | None)=None
    ):
        i, start, n, plot, loc, crop_prior, crop_post, save_plot = args
        dim = self.show_stim(hide=True)[0].shape[0]
        radius = ecc_max * (dim / 8.4)

        if loc == "center":
            x = y = (dim + 1) / 2
        
        elif loc == "irrelevant_patch":
            x = y = radius + 10

        # This shit is not used, I use the imfovbeta and imfovgamma
        mask_w_in = self.nsp.utils.css_gaussian_cut(dim, x, y, radius).reshape((425, 425))
        rf_mask_in = self.nsp.utils.make_circle_mask(dim, x, y, radius, fill="y", margin_width=0)
        full_ar_in = ar_in = self.show_stim(img_no=i, hide=True)[0]

        if i % 100 == 0:
            print(f"Processing image number: {i} out of {n + start}")

        # Crop the image first, then provide this as input to the visfeat function
        if crop_prior:

            x_min, x_max, y_min, y_max = self.nsp.utils.get_bounding_box(rf_mask_in)
            ar_in = ar_in[x_min:x_max, y_min:y_max]
            mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
            rf_mask_in = rf_mask_in[x_min:x_max, y_min:y_max]

        return self.get_scce_contrast(
            ar_in,
            plot=plot,
            cmap=cmap,
            crop_prior=crop_prior,
            crop_post=crop_post,
            save_plot=save_plot,
            return_imfovs=return_imfovs,
            imfov_overlay=imfov_overlay,
            config_path=config_path,
            lgn_instance=lgn_instance
        )


    # Function to get the visual contrast features and predictability estimates
    # IMPROVE: make sure that it also works for all subjects later on. Take subject arg, clean up paths.
    def features(self):
        feature_paths = [
            f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms.pkl', #dep, now get_rms
            f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms_crop_prior.pkl', #dep, now get_rms
            f'{self.nsp.own_datapath}/all_visfeats_scce.pkl',
            f'{self.nsp.own_datapath}/all_visfeats_scce_large.pkl',
            f'{self.nsp.own_datapath}/visfeats/scce/scce_stack.pkl',
            f'{self.nsp.own_datapath}/subj01/pred/all_predestims.h5', # old, .95 correlation with new
            f'{self.nsp.own_datapath}/visfeats/pred/all_predestims_vgg-b.csv', # also about .9-.95 correlation with alex
            f'{self.nsp.own_datapath}/visfeats/pred/all_predestims_alexnet_new.csv' 
            ]
        return {os.path.basename(file): self.nsp.datafetch.fetch_file(file) for file in feature_paths}
    
    def get_rms(self, subject:str, rel_or_irrel:str='rel', crop_prior:bool=True, outlier_bound:float=.3):
        """Function to get the Root Mean Square values for a given subject

        Args:
        - subject (str): Which subject.
        - rel_or_irrel (str, optional): Whether to get the RMS values for a central (relevant) patch, 
            or for a peripheral (irrelevant) patch. Defaults to 'rel'.
        - crop_prior (bool, optional): Whether to take the RMS values from computations in which the 
            image was cropped prior to computing the RMS, or from computations where images were cropped
            after computing the overall RMS of the image. Defaults to True.
        - outlier_bound (float, optional): What boundary to use for outlier filtering. Defaults to .3.

        Returns:
            np.ndarray: The resulting values.
        """        
        rms_loc_relevance = 'rms' if rel_or_irrel == 'rel' else 'rms_irrelevant'
        
        if crop_prior:
            Xraw = self.nsp.datafetch.fetch_file(f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms_crop_prior.pkl')[subject][rms_loc_relevance]['rms_z']
        else:
            Xraw = self.nsp.datafetch.fetch_file(f'{self.nsp.own_datapath}/visfeats/rms/all_visfeats_rms.pkl')[subject][rms_loc_relevance]['rms_z']

        Xnorm = zs(self.nsp.utils.replace_outliers(np.array(Xraw).reshape(-1,1), m=outlier_bound))
        indices = self.imgs_designmx()[subject] # Get the 73k-based indices for the specific subject

        return pd.DataFrame(Xnorm, index=indices, columns=['rms'])
        
    def get_scce(self, subject:str, sc_or_ce:str):
        """Function to get the Spatial Coherence or Contrast Energy values for a given subject

        Args:
        - subject (str): Which subject.
        - sc_or_ce (str): Whether to get the Spatial Coherence or Contrast Energy values.

        Returns:
            np.ndarray: The resulting values.
        """        
        indices = self.imgs_designmx()[subject] # Get the 73k-based indices for the specific subject
        scce = self.nsp.datafetch.fetch_file(f'{self.nsp.own_datapath}/visfeats/scce/scce_stack.pkl')
        X = scce[f'{sc_or_ce}_z'][indices]
        
        return X.to_frame(sc_or_ce)
        
        
    # DEPRECATED NOW. USE FUNCTIONS ABOVE
    def baseline_feats(self, subject:str, feat_type:str, outlier_bound:float=.3):
        """
        Input options:
        - 'rms' for Root Mean Square
        - 'ce' for Contrast Energy (ce_l) for larger pooling region (5 instead of 1 degree radius)
        - 'sc' for Spatial Coherence (sc_l) for larger pooling region (5 instead of 1 degree radius)
        """

        if feat_type == 'rms':
            file_name = 'all_visfeats_rms_crop_prior.pkl'
            category = feat_type
            key = 'rms_z'
        elif feat_type == 'sc':
            file_name = 'all_visfeats_scce.pkl'
            category = 'scce'
            key = 'sc_z'
        elif feat_type == 'ce':
            file_name = 'all_visfeats_scce.pkl'
            category = 'scce'
            key = 'ce_z'
        elif feat_type[-1:] == 'l':
            file_name = 'all_visfeats_scce_large.pkl'
            category = 'scce'
            key = 'ce_z' if 'ce' in feat_type else 'sc_z'
        # elif feat_type == 'ce_new':
        #     file_name = scc
        else:
            raise ValueError(f"Unknown feature type: {feat_type}")

        
        X = self.nsp.utils.replace_outliers(np.array(self.nsp.stimuli.features()[file_name]['subj01'][category][key]).reshape(-1,1), m=outlier_bound)
        return zs(X)
        
    def unpred_feats(self, cnn_type:str, content:bool, style:bool, ssim:bool, pixel_loss:bool, 
                     L1:bool, MSE:bool, verbose:bool, outlier_sd_bound:Optional[Union[str, float]]='auto', 
                     subject:Optional[str]=None):
        """
        Function to create an X matrix based on the exclusion criteria defined in the arguments.
        Input:
        - cnn_type: string, which type of cnn to get the unpredictability features from, 'vgg-b' or 'alexnet' 
            are currently available
        - content: boolean, whether to include content loss features
        - style: boolean, whether to include style loss features
        - ssim: boolean, whether to include structural similarity features
        - pixel_loss: boolean, whether to include pixel loss features
        - L1: boolean, whether to include L1 features
        - MSE: boolean, whether to include MSE or L2 features
        - verbose: boolean, whether to print intermediate info
        - outlier_sd_bound: float or 'auto', the number of standard deviations to use as a cutoff for outliers
        - subject: string, the subject to get the features for
        Output:
        - X: np.array, the X matrix based on the exclusion criteria
        """
        if outlier_sd_bound == 'auto':
            if cnn_type == 'vgg-b':
                cutoff_bound = 10
            elif cnn_type == 'alexnet' or 'alexnet_new':
                cutoff_bound = 5
        else: cutoff_bound = outlier_sd_bound
                        
        if cnn_type == 'alexnet':
            file_str = 'all_predestims.h5'
            predfeatnames = [name for name in list(self.features()[file_str].keys()) if name != 'img_ices']
        elif cnn_type == 'vgg-b':
            file_str = 'all_predestims_vgg-b.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']
        elif cnn_type == 'alexnet_new':
            file_str = 'all_predestims_alexnet_new.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name != 'img_ices']
        
        if subject is not None:    
            indices = self.imgs_designmx()[subject]
        else: indices = np.ones((30000,)).astype(bool)
            
        if not content:
            predfeatnames = [name for name in predfeatnames if 'content' not in name]
        if not style:
            predfeatnames = [name for name in predfeatnames if 'style' not in name]
        if not ssim:
            predfeatnames = [name for name in predfeatnames if 'ssim' not in name]
        if not pixel_loss:
            predfeatnames = [name for name in predfeatnames if 'pixel_loss' not in name]
        if not L1:
            predfeatnames = [name for name in predfeatnames if 'L1' not in name]
        if not MSE:
            predfeatnames = [name for name in predfeatnames if 'MSE' not in name]
        
        # data = {name: zs(self.nsp.utils.replace_outliers(self.nsp.stimuli.features()['all_predestims.h5'][name], m=outlier_bound)) for name in predfeatnames}
        
        data = {name: zs(self.nsp.utils.std_dev_cap(self.features()[file_str][name].fillna(.00001),num_std_dev=cutoff_bound))[indices] for name in predfeatnames}
        
        # Convert the dictionary values to a list of lists
        data_list = list(data.values())
        
        # Convert the list of lists to a 2D array
        X = np.array(data_list)

        # Transpose the array so that each row corresponds to a sample and each column corresponds to a feature
        X = X.T[:,:]
        
        if verbose:
            print(predfeatnames)
        
        return X

## THIS ONE WORKS, BUT DOESN'T ZSCORE IT YET -->> Also outdated now. 
    def unet_featmaps(self, list_layers:list, scale:str='cropped'):
        """
        Load in the UNet extracted feature maps
        Input:
        - list_layers: list with values between 1 and 4 to indicate which layers to include
        - scale: string to select either 'cropped' for cropped images, or 'full' for full images
        """
        # Initialize an empty list to store the loaded feature maps for each layer
        matrices = []

        # Load the feature maps for each layer and append them to the list
        for layer in list_layers:
            file_path = f'{self.nsp.own_datapath}/subj01/pred/featmaps/tests/{scale}_unet_gt_feats_{layer}.npy'
            feature_map = np.load(file_path)
            # Apply z-score normalization if needed
            # feature_map = self.nsp.utils.get_zscore(feature_map, print_ars='n')
            # Reshape the feature map to have shape (n_imgs, n_components, 256)
            reshaped_feature_map = feature_map.reshape(feature_map.shape[0], -1, 256)
            # Take the mean over the flattened dimensions (last axis)
            mean_feature_map = np.mean(reshaped_feature_map, axis=-1)
            matrices.append(mean_feature_map)

        # Horizontally stack the loaded and averaged feature maps
        Xcnn_stack = np.hstack(matrices)

        return Xcnn_stack
    
    # This is actually deprecated. Does cool stuff but I'm using different feature maps now. 
    def plot_unet_feats(self, layer:int, batch:int, cmap:str='bone', subject:str='subj01', scale:str='cropped'):
        """
        Function to plot a selection of feature maps extracted from the U-Net class.
        Input:
        - layer: integer to select layer
        - batch: integer to select batch
        - cmap: string to define the matplotlib colour map used to plot the feature maps
        - subject: string to select the subject
        - scale: string to select either 'cropped' for cropped images, or 'full' for full images
        """
        with open(f'{self.nsp.own_datapath}/{subject}/pred/featmaps/{scale}/feats_gt_np_{batch}.pkl', 'rb') as f:
            feats_gt_np = pickle.load(f)
            
        # Get the number of feature maps
        num_feature_maps = feats_gt_np[0].shape[1]

        # Calculate the number of rows and columns for the subplots
        num_cols = int(np.ceil(np.sqrt(num_feature_maps)))
        num_rows = int(np.ceil(num_feature_maps / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

        # Flatten the axes array to make it easier to loop over
        axes = axes.flatten()

        # Loop over the feature maps and plot each one
        for i in range(num_feature_maps):
            # first index is the CNN layer, then, the [image, featmap no, dimx, dimy]
            axes[i].imshow(feats_gt_np[layer][random.randint(0, 500), i, :, :], cmap=cmap)
            axes[i].axis('off')  # Hide axes

        # If there are more subplots than feature maps, hide the extra subplots
        for j in range(num_feature_maps, len(axes)):
            axes[j].axis('off')
        plt.show()
        
        
    # These are the correct featmaps (17-05-2024) DON'T THINK SO, IT'S THE NEXT ONE.
    def alex_featmaps_old(self, layers:list, pcs_per_layer:Union[int, str]='all', subject:str='subj01',
                      plot_corrmx:bool=True):
        """
        Load in the feature maps from the AlexNet model for a specific layer and subject
        
        Args:
        - layers: list of integers representing the layers of the AlexNet model to include in the X-matrix
        - pcs_per_layer: integer value indicating the top amount of principal components to which the feature map should be reduced to, or 
            'all' if all components should be included.
        - subject: string value representing the subject for which the feature maps should be loaded in
        - plot_corrmx: boolean value indicating whether a correlation matrix should be plotted for the top 500 principal components of the AlexNet model
        
        Out:
        - X_all: np.array containing the feature maps extracted at the specified layers of the AlexNet model
        """
        # Load in the feature maps extracted by the AlexNet model
        X_all = []
                    

        if isinstance(pcs_per_layer, int):
            cut_off = pcs_per_layer
        
        for n_layer, layer in enumerate(layers):
            this_X = np.load(f'{self.nsp.own_datapath}/subj01/center_strict/alex_lay{layer}.npy')
            if n_layer == 0:
                if pcs_per_layer == 'all':
                    cut_off = this_X.shape[0]
                X_all = this_X[:, :cut_off]
            else: X_all = np.hstack((X_all, this_X[:, :cut_off]))
            
        if plot_corrmx:
            # Correlation matrix for the 5 AlexNet layers
            # Split X_all into separate arrays for each layer
            X_split = np.hsplit(X_all, len(layers)) # Amazing function, splits the array into n arrays along the columns

            # Initialize an empty matrix for the correlations
            corr_matrix = np.empty((len(layers), len(layers)))

            # Calculate the correlation between each pair of layers
            for i in range(len(layers)):
                for j in range(len(layers)):
                    corr_matrix[i, j] = np.corrcoef(X_split[i].flatten(), X_split[j].flatten())[0, 1]

            print(corr_matrix)

            # Create a heatmap from the correlation matrix
            plt.imshow(corr_matrix, cmap='Greens_r', interpolation='nearest')
            plt.colorbar(label='Correlation coefficient')

            # Add annotations to each cell
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    plt.text(j, i, format(corr_matrix[i, j], '.2f'),
                            ha="center", va="center",
                            color="black")

            relu_nos = [no for no in range(1,6)]
            # Set the tick labels
            plt.xticks(np.arange(len(layers)), relu_nos)
            plt.yticks(np.arange(len(layers)), relu_nos)

            # Set the title and labels
            plt.title('Correlation matrix of\ntop 500 principal components of AlexNet')
            plt.xlabel('ReLU layer')
            plt.ylabel('ReLU layer')

            plt.show()
        
        return X_all    
    
    
    
    def alex_featmaps(self, layers:(list | int)=[1, 4, 7, 9, 11], subject:str='subj01',
                    plot_corrmx:bool=True, smallpatch:bool=False):
        """
        Load in the feature maps from the AlexNet model for a specific layer and subject
        
        Args:
        - layers: list of integers representing the layers of the AlexNet model to include in the X-matrix.
            Options are 1, 4, 7, 9, 11. These correspond to the ReLU layers of the AlexNet model.
        - subject: string value representing the subject for which the feature maps should be loaded in
        - plot_corrmx: boolean value indicating whether a correlation matrix should be plotted for the top 500 principal components of the AlexNet model
        
        Out:
        - X_all: np.array containing the feature maps extracted at the specified layers of the AlexNet model
        """
        smallpatch_str = '_smallpatch' if smallpatch else ''
        
        full_img_alex = []
        layers = [layers] if type(layers) is int else layers
        for n_layer, cnn_layer in enumerate(layers):
            if n_layer == 0:
                full_img_alex = np.load(f'{self.nsp.own_datapath}/{subject}/encoding/regprepped_featmaps{smallpatch_str}_layer{cnn_layer}.npy')
            else: full_img_alex = np.hstack((full_img_alex, np.load(f'{self.nsp.own_datapath}/{subject}/encoding/regprepped_featmaps{smallpatch_str}_layer{cnn_layer}.npy')))
 
        if len(layers) < 5:
            plot_corrmx = False
            
        if plot_corrmx:
            # Correlation matrix for the 5 AlexNet layers
            # Split X_all into separate arrays for each layer
            X_split = np.hsplit(full_img_alex, len(layers)) # Amazing function, splits the array into n arrays along the columns

            # Initialize an empty matrix for the correlations
            corr_matrix = np.empty((len(layers), len(layers)))

            # Calculate the correlation between each pair of layers
            for i in range(len(layers)):
                for j in range(len(layers)):
                    corr_matrix[i, j] = np.corrcoef(X_split[i].flatten(), X_split[j].flatten())[0, 1]

            print(corr_matrix)

            # Create a heatmap from the correlation matrix
            plt.imshow(corr_matrix, cmap='Greens_r', interpolation='nearest')
            plt.colorbar(label='Correlation coefficient')

            # Add annotations to each cell
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    plt.text(j, i, format(corr_matrix[i, j], '.2f'),
                            ha="center", va="center",
                            color="black")

            relu_nos = [no for no in range(1,6)]
            # Set the tick labels
            plt.xticks(np.arange(len(layers)), relu_nos)
            plt.yticks(np.arange(len(layers)), relu_nos)

            # Set the title and labels
            plt.title(f'Correlation matrix of\ntop {len(layers)} principal components of AlexNet')
            plt.xlabel('ReLU layer')
            plt.ylabel('ReLU layer')

            plt.show()
        
        return full_img_alex    
    
    # Create design matrix containing ordered indices of stimulus presentation per subject
    def imgs_designmx(self):
        
        subjects = os.listdir(f'{self.nsp.nsd_datapath}/nsddata/ppdata')
        exp_design = f'{self.nsp.nsd_datapath}/nsddata/experiments/nsd/nsd_expdesign.mat'
        
        # Load MATLAB file
        mat_data = loadmat(exp_design)

        # Order of the presented 30000 stimuli, first 1000 are shared between subjects, rest is randomized (1, 30000)
        # The values take on values betweeon 0 and 1000
        img_order = mat_data['masterordering']-1

        # The sequence of indices from the img_order list in which the images were presented to each subject (8, 10000)
        # The first 1000 are identical, the other 9000 are randomly selected from the 73k image set. 
        img_index_seq = (mat_data['subjectim'] - 1) # Change from matlab to python's 0-indexing
        
        # Create design matrix for the subject-specific stimulus presentation order
        stims_design_mx = {}
        stim_list = np.zeros((img_order.shape[1]))
        for n_sub, subject in enumerate(sorted(subjects)):
        
            for stim in range(0, img_order.shape[1]):
                
                idx = img_order[0,stim]
                stim_list[stim] = img_index_seq[n_sub, idx]
                
            stims_design_mx[subject] = stim_list.astype(int)
        
        return stims_design_mx
    
    # Get random design matrix to test other fuctions
    def random_designmx(self, idx_min = 0, idx_max = 40, n_img = 20):
        
        subjects = os.listdir(f'{self.nsp.nsd_datapath}/nsddata/ppdata')
        
        # Create design matrix for the subject-specific stimulus presentation order
        stims_design_mx = {}
        for subject in sorted(subjects):
            # Generate 20 random integer values between 0 and 40
            stim_list = np.random.randint(idx_min, idx_max, n_img)
            stims_design_mx[subject] = stim_list
        
        return stims_design_mx
    
    # Plot a correlation matrix for specific loss value estimations of unpredictability estimates
    def unpred_corrmatrix(self, subject='subj01', type:str='content', loss_calc:str='MSE', cmap:str='copper_r', cnn_type:str='alexnet'):
        """
        Plot a correlation matrix for specific loss value estimations of unpredictability estimates.

        Parameters:
        subject (str): The subject for which to plot the correlation matrix. Default is 'subj01'.
        type (str): The type of loss value estimations to include in the correlation matrix. Default is 'content'.
        loss_calc (str): The type of loss calculation to use. Default is 'MSE'.
        cmap (str): The colormap to use for the heatmap. Default is 'copper_r'.
        """
        
        # if cnn_type == 'alexnet':
        #     file_str = 'all_predestims.h5'
        #     predfeatnames = [name for name in list(self.features()[file_str].keys()) if name.endswith(loss_calc) and name.startswith(type)]
        # elif cnn_type == 'vgg-b':
        #     file_str = 'all_predestims_vgg-b.csv'
        #     predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith(type)]
        
        # # predfeatnames = [name for name in list(self.features()['all_predestims.h5'].keys()) if name.endswith(loss_calc) and name.startswith(type)]

        # # Build dataframe
        # data = {name: self.features()[file_str][name] for name in predfeatnames}
        
        # Get the subject specific-indices, only required as long as I haven't calculated all the features for all 73k
        indices = self.imgs_designmx()[subject]

        
        if cnn_type == 'alexnet':
            file_str = 'all_predestims.h5'
            predfeatnames = [name for name in list(self.features()[file_str].keys()) if name.endswith(loss_calc) and name.startswith('content')]
            indices = np.ones((30000,)).astype(bool)
        elif cnn_type == 'vgg-b':
            file_str = 'all_predestims_vgg-b.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith('content')]
        elif cnn_type == 'alexnet_new':
            file_str = 'all_predestims_alexnet_new.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith('content')]
        
        # Build dataframe
        data = {name: self.features()[file_str][name][indices] for name in predfeatnames}
        
        
        df = pd.DataFrame(data)

        # Compute correlation matrix
        corr_matrix = df.corr()
        ticks = [f'Layer {name.split("_")[2]}' for name in predfeatnames]
        # sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks, vmin=0, vmax=1)
        plt.title(f'U-Net unpredictability estimates\n{cnn_type} {type} loss {loss_calc} correlation matrix')
        plt.show()
        
    def plot_correlation_matrix(self, subject:str='subj01', include_rms:bool=True, include_ce:bool=True, include_ce_l:bool=True, include_sc:bool=True, 
                                include_sc_l:bool=True, include_ce_new:bool=True, include_sc_new:bool=True, cmap:str='copper_r', cnn_type:str='alexnet', loss_calc:str='MSE'): 
        """
        Plot a correlation matrix for the MSE content loss values per layer, and the baseline features.

        Parameters:
        include_rms (bool): If True, include the 'rms' column in the correlation matrix.
        include_ce (bool): If True, include the 'ce' column in the correlation matrix.
        include_ce_l (bool): If True, include the 'ce_l' column in the correlation matrix.
        include_sc (bool): If True, include the 'sc' column in the correlation matrix.
        include_sc_l (bool): If True, include the 'sc_l' column in the correlation matrix.
        """
        # predfeatnames = [name for name in list(self.features()['all_predestims.h5'].keys()) if name.endswith('MSE') and name.startswith('content')]
        # predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith(type)]
        
        # Get the subject specific-indices, only required as long as I haven't calculated all the features for all 73k
        indices = self.imgs_designmx()[subject]

        
        if cnn_type == 'alexnet':
            file_str = 'all_predestims.h5'
            predfeatnames = [name for name in list(self.features()[file_str].keys()) if name.endswith(loss_calc) and name.startswith('content')]
            indices = np.ones((30000,)).astype(bool)
        elif cnn_type == 'vgg-b':
            file_str = 'all_predestims_vgg-b.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith('content')]
        elif cnn_type == 'alexnet_new':
            file_str = 'all_predestims_alexnet_new.csv'
            predfeatnames = [name for name in self.features()[file_str].columns if name.endswith(loss_calc) and name.startswith('content')]
        
        # Build dataframe
        data = {name: self.features()[file_str][name][indices] for name in predfeatnames}
        if include_rms:
            # data['rms'] = self.baseline_feats('rms').flatten()
            data['rms'] = self.get_rms(subject).values.flatten()
        if include_ce:
            data['ce'] = self.baseline_feats(subject, feat_type = 'ce').flatten()
        if include_ce_l:
            data['ce_l'] = self.baseline_feats(subject, feat_type = 'ce_l').flatten()
        if include_sc:
            data['sc'] = self.baseline_feats(subject, feat_type = 'sc').flatten()
        if include_sc_l:
            data['sc_l'] = self.baseline_feats(subject, feat_type = 'sc_l').flatten()
        if include_ce_new:
            data['ce_new'] = self.get_scce(subject, 'ce').values.flatten()
        if include_sc_new:
            data['sc_new'] = self.get_scce(subject, 'sc').values.flatten()

        df = pd.DataFrame(data)

        # Compute correlation matrix
        corr_matrix = df.corr()
        ticks = [f'Pred {int(name.split("_")[2])+1}' for name in predfeatnames]
        if include_rms:
            ticks.append('RMS 1°')
        if include_ce:
            ticks.append('CE 1°')
        if include_ce_l:
            ticks.append('CE 5°')
        if include_sc:
            ticks.append('SC 1°')
        if include_sc_l:
            ticks.append('SC 5°')
        if include_ce_new:
            ticks.append('CE new')
        if include_sc_new:
            ticks.append('SC new')
            
        plt.figure(figsize=(9,7))
        # sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks, vmin=0, vmax=1)
        plt.title(f'Correlation matrix for the MSE content loss values per\n{cnn_type} layer, and the baseline features')
        plt.show()
            
        
    def extract_features(self, subject:str='subj01', layer:int=4, start_img:int=0, n_imgs:int=1,
                         batch_size:int=10, pca_components=10, verbose:bool=False, img_crop:bool=True):
        # Load the pretrained AlexNet model
        model = models.alexnet(pretrained=True)
        model.eval() # Set model to evaluation mode, as it's pretrained and we'll use it for feature extraction
        
        class ImageDataset(Dataset):
            def __init__(self, supclass, image_ids, transform=None):
                self.supclass = supclass
                self.image_ids = image_ids
                self.transform = transform

            def __len__(self):
                return len(self.image_ids)

            def __getitem__(self, idx):
                img_id = self.image_ids[idx]
                if img_crop: imgnp = (self.supclass.show_stim(img_no=img_id, hide=True, small=True)[0][163:263,163:263])
                else: imgnp = self.supclass.show_stim(img_no=img_id, hide=True, small=True)[0]
                imgPIL = Image.fromarray(imgnp) # Convert into PIL from np

                if self.transform:
                    imgPIL = self.transform(imgPIL)

                return imgPIL
                
        preprocess = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x24 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])
        
        layer_names = self.nsp.utils.get_layer_names(model) # Get the layer names
        train_nodes, _ = get_graph_node_names(model) # Get the node names
        if verbose:
            print(layer_names)
            print(train_nodes)
        this_layer = train_nodes[layer]
        this_layer_name = layer_names[layer]
        
        feature_extractor = create_feature_extractor(model, return_nodes=[this_layer])

        image_ids = self.imgs_designmx()[subject][start_img:start_img+n_imgs]
        dataset = ImageDataset(self, image_ids, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        def _fit_pca(feature_extractor, dataloader):
            # Define PCA parameters
            pca = IncrementalPCA(n_components=pca_components, batch_size=batch_size)

            while True:  # Keep trying until successful
                try:
                    # Fit PCA to batch
                    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                        # Extract features
                        ft = feature_extractor(d)
                        # Flatten the features
                        ft_flat = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
                        # Fit PCA to batch
                        pca.partial_fit(ft_flat.detach().cpu().numpy())
                    return pca, ft  # Return the PCA object
                except Exception as e:
                    print(f"Error occurred: {e}")
                    print("Restarting PCA fitting...")
                    
        pca, feature = _fit_pca(feature_extractor, dataloader)
        
        return image_ids, dataset, pca, feature, this_layer, this_layer_name
        
    def plot_features(self, which_img:int, features, layer:str, layer_type:str, img_ids:list, num_cols=10, random_cmap:bool=False):
            

        feature_maps = features[layer][which_img].detach().numpy()
        
        # Number of feature maps
        num_maps = feature_maps.shape[0]

        # Number of rows in the subplot grid
        num_rows = num_maps // num_cols
        if num_maps % num_cols:
            num_rows += 1
            
        cmaps = list(colormaps)
        this_cmap = cmaps[random.randint(0, len(cmaps))] if random_cmap else 'binary_r'
        if random_cmap:
            print (f'The Lord has decided for you to peek into feature space through the lens of {this_cmap}')
        
        # Create a figure for the subplots
        if layer_type == 'input':
            figsize = (10, 3)
        else: figsize = (num_cols, num_rows)
        plt.figure(figsize=figsize)

        # Plot each feature map
        for i in range(num_maps):
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(feature_maps[i], cmap=this_cmap)
            plt.axis('off')
        # Show the plot
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.tight_layout()
        plt.show()
    
    # Function to load in a list of images and masks given a list of indices. Used to provide the right input
    # to the U-Net model. Option to give the mask location, the eccentricity of the mask, and the output format.
    # The alternative mask_loc is 'irrelevant_patch', which places the mask at a fixed location in the image.
    # However, this is not yet working, because the final evaluation is done based on a 'eval_mask' object.
    # Perhaps also add this to the function.
    # Could also add the option to select a subject so it automatically gets a specified amount of their images.

    ##### Give this a better name, and change a bit so it works for different subjects. It is not really random, but 
    # it CAN be random, because it mainly just helps provide the lists
    def rand_img_list(self, n_imgs, asPIL:bool = True, add_masks:bool = True, mask_loc = 'center', ecc_max = 1, select_ices = None):
        imgs = []
        img_nos = []
        for i in range(n_imgs):
            img_no = random.randint(0, 27999)
            if select_ices is not None: img_no = select_ices[i]
            img = self.show_stim(img_no=img_no, hide=True)[0]

            if i == 0:
                dim = img.shape[0]
                radius = ecc_max * (dim / 8.4)
                
                if mask_loc == 'center': x = y = (dim + 1)/2
                elif mask_loc == 'irrelevant_patch': x = y = radius + 10

            if asPIL: img = Image.fromarray(img)

            imgs.append(img)
            img_nos.append(img_no)
        mask = (self.nsp.utils.make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0) == 0)

        if asPIL: mask = Image.fromarray(mask)
        masks = [mask] * n_imgs

        return imgs, masks, img_nos

    # Faster and more sparse computation of the unpred features, does require preparing
    # PIL lists of images and masks beforehand.        
    def comp_unpred(
        self,
        cnn_type: str = "alex",
        pretrain_version: str = "places20k",
        eval_mask_factor: float = 1.2,
        input_imgs: list | None = None,
        input_masks: list | None = None,
        plot_eval_mask: bool = False,
    ):
        """
        Computes the unpredictability of images using a U-Net model.

        Args:
            cnn_type (str, optional): The type of CNN model to use. Defaults to "alex".
            pretrain_version (str, optional): The version of pre-trained weights to use. 
                Defaults to "places20k".
            eval_mask_factor (float, optional): The factor to scale the evaluation mask size. 
                Defaults to 1.2.
            input_imgs (list, optional): The list of input images. Defaults to None.
            input_masks (list, optional): The list of input masks. Defaults to None.
            plot_eval_mask (bool, optional): Whether to plot the evaluation mask. Defaults to False.

        Returns:
            payload_crop: The result of the U-Net analysis.
        """
        
        # Load in the U-Net
        if pretrain_version == "places20k":
            pretrain = "pconv_circ-places20k.pth"
        elif pretrain_version == "places60k":
            pretrain = "pconv_circ-places60k-fine.pth"
        elif pretrain_version == "original":
            pretrain = "pretrained_pconv.pth"
        else:
            raise TypeError(
                "Please select a valid pretrain version: places20k, places60k or original"
            )

        unet = UNet(checkpoint_name=pretrain, feature_model=cnn_type)

        imgs = input_imgs
        masks = input_masks
        img_nos = list(range(0, (len(imgs) + 1)))

        # Get the evaluation mask based on the evaluation mask size factor argument.
        eval_mask = self.nsp.utils.scale_square_mask(
            ~np.array(masks[0]),
            min_size=((eval_mask_factor / 1.5) * 100),
            scale_fact=eval_mask_factor,
        )

        if plot_eval_mask:
            plt.imshow(eval_mask)
            plt.axis("off")

        # Run them through the U-Net
        payload_crop = unet.analyse_images(
            imgs, masks, return_recons=True, eval_mask=eval_mask
        )

        return payload_crop
        
    # Allround function to run the U-Net and create intuitive plots of the resulting predictability estimates.
    def predplot(self, subject:str = None, start_img:int = 0, n_imgs:int = 5, mask_loc:str = 'center', ecc_max:float = 1, select_ices = 'subject_based', 
                cnn_type:str = 'alex', pretrain_version:str = 'places20k', eval_mask_factor:float = 1.2, log_y_MSE:str = 'y', dark_theme:bool=False):
        
        # Load in the U-Net
        if pretrain_version == 'places20k':
            pretrain = 'pconv_circ-places20k.pth'
        elif pretrain_version == 'places60k':
            pretrain = 'pconv_circ-places60k-fine.pth'
        elif pretrain_version == 'original':
            pretrain = 'pretrained_pconv.pth'
        else:
            raise TypeError('Please select a valid pretrain version: places20k, places60k or original')
            
        unet=UNet(checkpoint_name = pretrain,feature_model = cnn_type)

        # What images will be processed:
        if select_ices == 'random': # A random set of images
            specific_imgs = [random.randint(0,72999) for _ in range(n_imgs)]
        # If it is a list, set specific_imgs to that list
        elif type(select_ices) == list:
            specific_imgs = select_ices
        elif select_ices == 'subject_based':
            dmx = self.imgs_designmx() # A range of images based on the subject-specific design matrix
            subj_imgs = list(dmx[subject])
            specific_imgs = subj_imgs[start_img:start_img + n_imgs]
        else: 
            raise TypeError('Please select a valid image selection method: random, subject_based or a list of specific image indices')
            
        # Get the images, masks and image numbers based on the specific image selection
        imgs, masks, img_nos = self.rand_img_list(n_imgs, asPIL = True, add_masks = True, mask_loc = mask_loc, ecc_max = ecc_max, select_ices = specific_imgs)
            
        # Get the evaluation mask based on the evaluation mask size factor argument.
        eval_mask = self.nsp.utils.scale_square_mask(~np.array(masks[0]), min_size=((eval_mask_factor/1.5)*100), scale_fact= eval_mask_factor)

        # Run the images through the U-Net and time how long it takes.
        start_time = time.time()
    
        # Run them through the U-Net
        payload_full = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = None)
        payload_crop = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = eval_mask)

        end_time = time.time()

        total_time = end_time - start_time
        average_time_per_image = (total_time / n_imgs) / 2

        print(f"Average time per image: {average_time_per_image} seconds")
        
        if dark_theme:
            plt.style.use('dark_background')  # Apply dark background theme

        for img_idx in range(len(imgs)):
            scene_no = img_nos[img_idx]
            titles = ['Ground Truth', 'Input Masked', 'Output Composite', '', 'Content loss values', '', 'Style loss values']
            # fig, axes = plt.subplots(2, 5, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1, 1, 2, 2]})  # Create 4 subplots
            fig, axes = plt.subplots(2, 7, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 2, 2, .4, 2.5, 1.3, 2.5]})  # Create 4 subplots
            for ax in axes[:,3]:
                ax.axis('off')
            for ax in axes[:,5]:
                ax.axis('off')

            for eval_size in range(2):
                this_payload = payload_full if eval_size == 0 else payload_crop

                for loss_type in ['content', 'style']:
                    ntype = 6 if loss_type == 'style' else 4
                    yrange = [0, 5] if loss_type == 'content' else [0, .05]
                    ylogrange = [0.1, 100] if loss_type == 'content' else [0.00001, .1]
                    n_layers = 5
                    losses = {}
                    MSE = []
                    L1 = []

                    for i in range(n_layers):
                        MSE.append(round(this_payload[f"{loss_type}_loss_{i}_MSE"][img_idx], 3))  # Get the loss for each layer
                        L1.append(round(this_payload[f"{loss_type}_loss_{i}_L1"][img_idx], 3))  # Get the loss for each layer
                        losses['MSE'] = MSE
                        losses['L1'] = L1
                    
                    # Plot the loss values
                    axes[eval_size, ntype].plot(range(1, n_layers + 1), L1, marker='o', color='crimson', linewidth=3)  # L1 loss
                    if eval_size == 0:
                        axes[eval_size, ntype].set_title(titles[ntype])
                    if eval_size == 1:
                        axes[eval_size, ntype].set_xlabel('Feature space (Alexnet layer)')
                                
                    if loss_type == 'content':
                        # Create a secondary y-axis for MSE
                        ax_mse = axes[eval_size, ntype].twinx()
                    else: 
                        ax_mse = axes[eval_size, ntype]
                    ax_mse.plot(range(1, n_layers + 1), MSE, marker='o', color='cornflowerblue', linewidth=3)  # MSE loss
                    if loss_type == 'content':
                        ax_mse.tick_params(axis='y', labelcolor='cornflowerblue', labelsize = 12)
                    
                    
                    if log_y_MSE == 'y' and loss_type == 'content':
                        ax_mse.set_yscale('log')  # Set y-axis to logarithmic scale for MSE
                        ax_mse.set_ylabel('MSE Loss (log)', color='cornflowerblue', fontsize = 14)
                        ax_mse.set_ylim(ylogrange[0], ylogrange[1])
                        ax_mse.grid(False)
                        axes[eval_size, ntype].set_ylabel('L1 Loss (linear)', color='crimson', fontsize = 14)
                        axes[eval_size, ntype].set_ylim([yrange[0], yrange[1]])  # Set the range of the y-axis for L1
                    else:
                        axes[eval_size, ntype].set_ylabel('Loss value', color='white', fontsize = 14)
                        axes[eval_size, ntype].set_ylim([yrange[0], yrange[1]])
                        
                    axes[eval_size, ntype].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
                    if loss_type == 'content':
                        axes[eval_size, ntype].tick_params(axis='y', labelcolor='crimson', labelsize = 12)
                    axes[eval_size, ntype].grid(False)
                    
                    if loss_type == 'style':
                        ax_mse.legend(['L1 (MAE)', 'MSE'], loc='upper right')
                    
                fig.suptitle(f"Image Number: {scene_no}\n\n"
                            f"Cropped eval dissimilarity stats:\n"
                            f"Structural Similarity: {round(this_payload['ssim'][img_idx],4)}\n"
                            f"Pixel loss L1 (MAE): {round((this_payload['pixel_loss_L1'][img_idx]).astype('float'), 4)}\n"
                            f"Pixel loss MSE: {round((this_payload['pixel_loss_MSE'][img_idx]).astype('float'), 4)}", fontsize=14)

                # Loop through each key in the recon_dict to plot
                for i, key in enumerate(['input_gt', 'input_masked', 'out_composite']):
                    img = this_payload['recon_dict'][key][img_idx].permute(1, 2, 0)
                    axes[eval_size, i].imshow(img)
                    if eval_size == 0:
                        axes[eval_size, i].set_title(titles[i], fontsize = 12)
                    axes[eval_size, i].axis('off')  # To keep images square and hide the axes
            
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05)  # Adjust the spacing between subplots
            plt.show()
            
        plt.style.use('default')  # Apply dark background theme

        return imgs, masks, img_nos, payload_full, payload_crop
        
        
    def get_img_extremes(
        self,
        X: np.ndarray,
        n: int,
        start_img:int=0,
        top: str = "unpred",
        layer: bool = 5,
        add_circle: bool = True,
        plot:bool=True,
        verbose: bool = True,
    ):
        """
        Returns the indices of the top or bottom n values in each column of the input array X.

        Parameters:
        X (ndarray): Input array of shape (m, n).
        n (int): Number of indices to return.
        top (bool, optional): If True, returns the indices of the top n values. If False, returns the indices of the bottom n values. Default is True.

        Returns:
        ndarray: Array of shape (n, n_columns), where n_columns is the number of columns in X. Each column contains the indices of the top or bottom n values.
        """

        # Argsort returns the indices that would sort the array
        sorted_indices = np.argsort(X, axis=0)

        final_img = start_img + n

        # We want the top or bottom n, so we take the last or first n rows
        if top == "unpred":
            n_indices = sorted_indices[-final_img:-start_img if start_img != 0 else None]
            # The rows are in ascending order, so we reverse them to get descending order
            n_indices = n_indices[::-1]
            topbottom_str = "top unpredictable"
        else:
            n_indices = sorted_indices[start_img:final_img]
            topbottom_str = "top predictable"
            
        if plot:
            # Plot the value distribution
            plt.figure()
            if top == "unpred":
                plt.plot(sorted(X[:, layer])[-final_img:])
            else:
                plt.plot(sorted(X[:, layer])[:final_img])
            plt.title(f"Value distribution - {topbottom_str} Layer {layer}")
            plt.show()

        if add_circle:
            mask = self.nsp.utils.make_circle_mask(
                425, 213, 213, (425 / 8.4), fill="n", margin_width=5
            ).reshape((425, 425))
            mask_3d = np.dstack([np.abs(mask)] * 3).astype(bool)
        else:
            mask_3d = 1
                
        if plot:
            for i in range(0, n):
                this_img = n_indices[i, layer]
                img = self.show_stim(
                    img_no=this_img, small=True, hide=True, crop=False
                )[0]
                
                # Apply the mask
                img_masked = img.copy()  # Create a copy to avoid modifying the original image
                img_masked[mask_3d] = img.max()

                plt.figure()
                plt.imshow(img_masked)
                plt.axis('off')
                plt.title(f"Image {this_img} - {topbottom_str} Layer {layer}")

        if verbose:
            print(f"Returning {n} {topbottom_str} images for layer {layer}...")

        return n_indices
        
        
class Analysis():
    
    def __init__(self, NSPobj):
        self.nsp = NSPobj
        pass
    
    def load_y(self, subject:str, roi:str, voxelsieve=VoxelSieve, n_trials:Union[int,str]=30000, include_xyz:bool=False) -> np.ndarray:
        """
        Loads the y values for a given subject and region of interest (ROI).

        Args:
        - subject (str): The subject.
        - roi (str): The region of interest.
        - voxelsieve (VoxelSieve class): VoxelSieve instance used to select voxels.
        - n_trials (int or str, optional): The number of trials to load. If 'all', loads 
            all trials (up to 30000). Default is 30000.
        - include_xyz (bool, optional): Whether to include the x, y, z coordinates in 
            the output. If False, these columns are skipped. Default is False.

        Returns: 
        - (np.ndarray) The loaded y-matrix consisting of the HRF signal betas from the NSD.

        Raises:
        - ValueError If n_trials is greater than 30000.
        """
        if isinstance(n_trials, int) and n_trials > 30000:
            raise ValueError("n_trials cannot be greater than 30000.")

        start_column = 0 if include_xyz else 3
        n_trials = 30000 if n_trials == 'all' else n_trials 
        
        # return (np.load(f'{self.nsp.own_datapath}/{subject}/betas/{roi}/all_betas.npy')[voxelsieve.vox_pick, start_column:][voxelsieve.vox_pick])[:, :n_trials]
        return (np.load(f'{self.nsp.own_datapath}/{subject}/betas/{roi}/all_betas.npy')[voxelsieve.vox_pick, start_column:])[:, :n_trials]
                                
    def run_ridge_regression(self, X:np.array, y:np.array, alpha:float=1.0, fit_icept:bool=False):
        """Function to run a ridge regression model on the data.

        Args:
        - X (np.array): The independent variables with shape (n_trials, n_features)
        - y (np.array): The dependent variable with shape (n_trials, n_outputs)
        - alpha (float, optional): Regularisation parameter of Ridge regression, larger values penalise stronger. Defaults to 1.0.

        Returns:
        - sk.linear_model._ridge.Ridge: The model object
        """        
        model = Ridge(alpha=alpha, fit_intercept=fit_icept)
        model.fit(X, y)
        return model

    # Not really necessary
    def _get_coefs(self, model:sk.linear_model._ridge.Ridge):
        return model.coef_

    # def _get_r(self, y:np.ndarray, y_hat:np.ndarray):
    #     """Function to get the correlation between the predicted and actual HRF signal betas.

    #     Args:
    #     - y (np.ndarray): The original HRF signal betas from the NSD
    #     - y_hat (np.ndarray): The predicted HRF signal betas

    #     Returns:
    #     - float: The correlation between the two sets of betas as a measure of fit
    #     """        
    #     return np.mean(y * self.nsp.utils.get_zscore(y_hat, print_ars='n'), axis=0)
    
    
    def _get_r(self, y_true:np.ndarray, y_pred:np.ndarray):
        """correlation coefficient between the **columns** of a matrix as goodness of fit metric
        
        in:
        y_true: ndarray, shape(n_samlpes,n_responses)
            true target/response vector
        y_pred: ndarray, shape(n_samples,n_responses)
            predicted target or response vector
        out:
        rscores: ndarray, shape(n_responses)
            correlation coefficient for every response 
        """
        zs = lambda v: (v-v.mean(0))/v.std(0) # z-score 
        return((zs(y_pred)*zs(y_true)).mean(0))
    
    def get_r_numpy(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Correlation coefficient between the **columns** of a matrix as goodness of fit metric
        
        Args:
            y_true: ndarray, shape(n_samples, n_responses)
                True target/response vector
            y_pred: ndarray, shape(n_samples, n_responses)
                Predicted target or response vector

        Returns:
            rscores: ndarray, shape(n_responses)
                Correlation coefficient for every response 
        """
        # Transpose the input matrices and compute the correlation coefficient
        r = np.corrcoef(zs(y_true).T, zs(y_pred).T)

        # np.corrcoef returns a 2D array, where the diagonal elements represent the correlation coefficients of each column with itself
        # and the off-diagonal elements represent the correlation coefficients between different columns.
        # Since we're only interested in the correlation between corresponding columns of y_true and y_pred, we only need the diagonal elements.
        # Since the input to np.corrcoef was [y_true.T, y_pred.T], the correlation between y_true and y_pred is on the off-diagonal.
        # Therefore, we need to take one off-diagonal from the 2x2 correlation matrix for each response.
        # This can be achieved by taking the elements with indices (i, n_responses + i) for all i in range(n_responses).
        n_responses = y_true.shape[1]
        return np.array([r[i, n_responses + i] for i in range(n_responses)])
    

    def score_model(self, X:np.ndarray, y:np.ndarray, model:sk.linear_model._ridge.Ridge, cv:int=5):
        """This function evaluates the performance of the model using cross-validation.

        Args:
        - X (np.ndarray): X-matrix, independent variables with shape (n_trials, n_features)
        - y (np.ndarray): y-matrix, dependent variable with shape (n_trials, n_outputs)
        - model (sk.linear_model._ridge.Ridge): The ridge model to score
        - cv (int, optional): The number of cross validation folds. Defaults to 5.

        Returns:
        - tuple: A tuple containing:
                - y_hat (np.ndarray): The predicted values for y, with shape (n_trials, n_outputs)
                - scores (np.ndarray): The R^2 scores for each output, with shape (n_outputs,)
        """        
        # Initialize the KFold object
        kf = KFold(n_splits=cv)
        
        # Initialize lists to store the predicted values and scores for each fold
        y_hat = []
        cor_scores = []
        
        # For each fold...
        for train_index, test_index in kf.split(X):
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Clone the model to ensure it's fresh for each fold
            model_clone = clone(model)
        
            # Fit the model on the training data
            model_clone.fit(X_train, y_train)
            
            # Fit the model on the training data
            # model.fit(X_train, y_train)
            
            # Predict the values for the testing data
            y_hat_fold = model_clone.predict(X_test)
            
            # Calculate the R^2 score for each column, no multi output
            # scores_fold = [r2_score(y_test[:, i], y_hat_fold[:, i]) for i in range(y_test.shape[1])]
            r_fold = self._get_r(y_test, y_hat_fold)
            
            # Append the predicted values and scores for this fold to the lists
            y_hat.append(y_hat_fold)
            cor_scores.append(r_fold)
        
        # Concatenate the predicted values from each fold into a single array
        y_hat = np.concatenate(y_hat)
        
        # Calculate the average R^2 score for each column
        # scores = np.mean(cor_scores, axis=0)
        
        return y_hat, cor_scores
    
    def plot_brain(self, prf_dict:dict, roi_masks:dict, subject:str, brain_numpy:np.ndarray, cmap, glass_brain:bool=False, save_img:bool=False, img_path:str='brain_image.png', lay_assign_plot:bool=False):
        """Function to plot a 3D np.ndarray with voxel-specific values on an anatomical brain template of that subject.

        Args:
        - prf_dict (dict): The pRF dictionary
        - roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
        - subject (str): The subject ID
        - brain_numpy (np.ndarray): The 3D np.ndarray with voxel-specific values
        - glass_brain (bool, optional): Optional argument to plot a glass brain instead of a static map. Defaults to False.
        - save_img (bool, optional): Optional argument to save the image to a file. Defaults to False.
        - img_path (str, optional): The path where the image will be saved. Defaults to 'brain_image.png'.
        """        
        brain_nii = nib.Nifti1Image(brain_numpy, self.nsp.cortex.anat_templates(prf_dict)[subject].affine)
        if glass_brain:
            display = plotting.plot_glass_brain(brain_nii, display_mode='ortho', colorbar=True, cmap=cmap, symmetric_cbar=False)
        else:
            display = plotting.plot_stat_map(brain_nii, bg_img=self.nsp.cortex.anat_templates(prf_dict)[subject], display_mode='ortho', colorbar=True, cmap=cmap, symmetric_cbar=False)
        
        if lay_assign_plot:        
            # New code to format colorbar ticks
            def format_tick(x, pos):
                return f'{x:.0f}'

            formatter = FuncFormatter(format_tick)

            if display._cbar:
                display._cbar.update_ticks()
                display._cbar.ax.yaxis.set_major_formatter(formatter)
                display._cbar.ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 6)))  # set ticks manually

        plt.show()
        
        if save_img:
            display.savefig(img_path)  # save figure to file
        
    def stat_on_brain(self, prf_dict:dict, roi_masks:dict, subject:str, stat:np.ndarray, xyzs:np.ndarray, glass_brain:bool, cmap, save_img:bool=False, img_path:Optional[str]='/home/rfpred/data/custom_files'):
        """Function to create a brain plot based on a specific statistic and the corresponding voxel coordinates.

        Args:
        - prf_dict (dict): The pRF dictionary
        - roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
        - subject (str): The subject ID
        - stat (np.ndarray): The statistic to plot on the brain
        - xyzs (np.ndarray): The voxel coordinates
        - glass_brain (bool, optional): Optional argument to plot a glass brain instead of a static map. Defaults to False.
        """        
        n_voxels = len(xyzs)
        statmap = np.zeros((n_voxels, 4))
        for vox in range(n_voxels):
            # statmap[vox, :3] = (xyzs[vox][0][0], xyzs[vox][0][1], xyzs[vox][0][2]) # this is for the old xyzs
            statmap[vox, :3] = xyzs[vox]
            statmap[vox, 3] = stat[vox]

        brainp = self.nsp.utils.coords2numpy(statmap, roi_masks[subject]['V1_mask'].shape, keep_vals=True)
        
        self.plot_brain(prf_dict, roi_masks, subject, brainp, cmap, glass_brain, save_img, img_path)
      
    def plot_learning_curve(self, X, y, model=None, alpha=1.0, cv=5):
        if model is None:
            # Create and fit the model
            model = self.run_ridge_regression(X, y, alpha)

        # Initialize the KFold object
        kf = KFold(n_splits=cv)

        # Initialize a list to store the scores for each fold
        scores = []

        # For each fold...
        for train_index, test_index in kf.split(X):
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Predict the values for the testing data
            y_hat = model.predict(X_test)

            # Calculate the correlation for each column
            scores_fold = [self._get_r(y_test[:, i], y_hat[:, i]) for i in range(y_test.shape[1])]

            # Append the average correlation for this fold to the list
            scores.append(scores_fold)

        # Plot the scores
        for i, scores_fold in enumerate(scores, start=1):
            # Scatter plot of individual scores
            plt.scatter([i]*len(scores_fold), scores_fold, color='blue', alpha=0.5)

            # Line plot of mean score
            plt.plot(i, np.mean(scores_fold), color='red', marker='o')

        plt.xlabel('Fold')
        plt.ylabel('Correlation Score')
        plt.title('Learning Curve')

        # Set x-axis to only show integer values
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()
                
                
    def plot_residuals(self, X, y, model=None, alpha=1.0):
        """Plot the residuals of the model, which is the difference between the actual y and the predicted y (y_hat)

        Args:
        - X (_type_): _description_
        - y (_type_): _description_
        - model (_type_, optional): _description_. Defaults to None.
        - alpha (float, optional): _description_. Defaults to 1.0.
        """        
        if model is None:
            # Create and fit the model
            model = self.run_ridge_regression(X, y, alpha)

        # Get the predicted values
        y_hat = model.predict(X)

        # Calculate the residuals
        residuals = y - y_hat

        # Create a scatter plot of the predicted values and residuals
        plt.scatter(y_hat, residuals, alpha=0.3)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()  
        
    def analysis_chain_slim(
            self,
            subject: str,
            ydict: Dict[str, np.ndarray],
            voxeldict: Dict[str, VoxelSieve],
            X: np.ndarray,
            alpha: float,
            cv: int,
            rois: list,
            X_alt: np.ndarray,
            fit_icept: bool = False,
            save_outs: bool = False,
            regname: (str | None) = "", # remove
            plot_hist: bool = True, 
            alt_model_type: str = "alternative model", # remove
            save_folder: str | None = None, # remove
            X_str: str | None = None, # remove
        ) -> pd.DataFrame:
        """Function to run a chain of analyses on the input data for each of the four regions of interest (ROIs).
            Includes comparisons with an uninformative dependent variable X matrix (such as a shuffled 
            version of the original X matrix), to assess the quality of the model in a relative way.
            Returns an array of which the first 3 columns contain the voxel coordinates (xyz) and the 
            fourth contains the across cross validation fold mean correlation R scores between the actual
            and predicted dependent variables (y vs. y_hat).

        Args:
        - ydict (np.ndarray): The dictionary containing the dependent variables y-matrices for each ROI.
        - X (np.ndarray): The independent variables X-matrix.
        - alpha (float): The regularisation parameter of the Ridge regression model.
        - cv (int): The number of cross-validation folds.
        - rois (list): The list of regions of interest (ROIs) to analyse.
        - X_alt (np.ndarray): The alternative model's X-matrix to compare the model against.
        - fit_icept (bool, optional): Whether or not to fit an intercept. If both X and y matrices are z-scored
                It is highly recommended to set it to False, otherwise detecting effects becomes difficult. Defaults to False.
        - save_outs (bool, optional): Whether or not to save the outputs. Defaults to False.

        Returns:
        - np.ndarray: Object containing the voxel coordinates and the mean R scores for each ROI. This can be
                efficiently turned into a numpy array using NSP.utils.coords2numpy, which in turn can be converted 
                into a nifti file using nib.Nifti1Image(np.array, affine), in which the affine can be extracted
                from a readily available nifti file from the specific subject (using your_nifti.affine).
        """
        r_values = {}
        r_uninformative = {}
        regcor_dict = {}  # Dictionary to store cor_scores
        regcor_dict['X'] = {}
        
        if save_outs:
            save_path = f'{self.nsp.own_datapath}/{subject}/results/{save_folder}'
            os.makedirs(save_path, exist_ok=True)
        
        # Calculate scores for the given X matrix
        for roi in rois:
            y = ydict[roi]
            model_og = self.run_ridge_regression(X, y, alpha=alpha, fit_icept=False)
            _, cor_scores = self.score_model(X, y, model_og, cv=cv)
            r_values[roi] = np.mean(cor_scores, axis=0)
            regcor_dict['X'][roi] = cor_scores  # Save cor_scores to dictionary

            xyz = voxeldict[roi].xyz
            # Get a vector with the roi name repeated for each voxel
            roi_vec = np.array([roi] * len(xyz)).reshape(-1,1)
            
            this_coords = np.hstack((xyz, roi_vec, np.array(r_values[roi]).reshape(-1,1)))
            this_coefs = np.mean(model_og.coef_, axis=1).reshape(-1,1)
            
            if roi == 'V1':
                coords = this_coords
                beta_coefs = this_coefs
            else:
                coords = np.vstack((coords, this_coords))
                beta_coefs = np.vstack((beta_coefs, this_coefs))

        regcor_dict['X_shuffled'] = {}
        
        # Calculate scores for the uninformative/baseline X matrix
        for roi in rois:
            y = ydict[roi]
            model_comp = self.run_ridge_regression(X_alt, y, alpha=alpha, fit_icept=fit_icept)
            _, cor_scores = self.score_model(X_alt, y, model_comp, cv=cv)
            r_uninformative[roi] = np.mean(cor_scores, axis=0)
            regcor_dict['X_shuffled'][roi] = cor_scores  # Save cor_scores to dictionary
            if roi == 'V1':
                uninf_scores = r_uninformative[roi].reshape(-1,1)
            else:
                uninf_scores = np.vstack((uninf_scores, r_uninformative[roi].reshape(-1,1)))

        delta_r_df = pd.DataFrame()
        
        for i, roi in enumerate(rois[:4]):
            # Calculate and store the delta-R values 
            if roi == 'V1':
                all_vox_delta_r = (r_values[roi] - r_uninformative[roi]).reshape(-1,1)
            else:
                all_vox_delta_r = np.vstack((all_vox_delta_r, (r_values[roi] - r_uninformative[roi]).reshape(-1, 1)))

            this_delta_r = round(np.mean(r_values[roi]) - np.mean(r_uninformative[roi]), 5) # TODO: is this the same as first delta_r /vox and then mean?
            
            delta_r_df[roi] = [this_delta_r]
            
            if plot_hist:
                if roi == 'V1': # Create a figure with 4 subplots
                    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                    # Flatten the axs array for easy iteration
                    axs = axs.flatten()
                
                # Underlay with the histogram of r_uninformative[roi] values
                axs[i].hist(r_uninformative[roi], bins=25, edgecolor=None, alpha=1, label=alt_model_type, color='burlywood')
                # Plot the histogram of r_values[roi] values in the i-th subplot
                axs[i].hist(r_values[roi], bins=25, edgecolor='black', alpha=0.5, label=X_str, color='dodgerblue')
                axs[i].set_title(f'{roi} delta-R: {this_delta_r}')
                axs[i].legend() if roi == 'V1' else None

                if roi == 'V4': # Add title and display the figure
                    plt.suptitle(f'{regname}', fontsize=16)
                    plt.tight_layout()
                    plt.show()
                
                plt.savefig(f'{save_path}/{regname}_plot.png') if save_outs else None
                    
        coords = np.hstack((coords, uninf_scores, all_vox_delta_r.reshape(-1,1), beta_coefs)) # also added beta coefficients as last column. very rough but works
                
        coords_df = pd.DataFrame(coords, columns=['x', 'y', 'z', 'roi', 'R', 'R_alt_model', 'delta_r', 'betas'])
                    
        coords_df.to_csv(f'{save_path}/{regname}_regdf.csv', index=False) if save_outs else None

        return coords_df    
    
    def analysis_chain(self, subject:str, ydict:Dict[str, np.ndarray], voxeldict:Dict[str, VoxelSieve], 
                       X:np.ndarray, alpha:float, cv:int, rois:list, X_uninformative:np.ndarray, 
                       fit_icept:bool=False, save_outs:bool=False, regname:Optional[str]='', plot_hist:bool=True,
                       shuf_or_baseline:str='s', save_folder:(str | None)=None, X_str:(str | None)=None) -> (np.ndarray, pd.DataFrame):
        """Function to run a chain of analyses on the input data for each of the four regions of interest (ROIs).
            Includes comparisons with an uninformative dependent variable X matrix (such as a shuffled 
            version of the original X matrix), to assess the quality of the model in a relative way.
            Returns an array of which the first 3 columns contain the voxel coordinates (xyz) and the 
            fourth contains the across cross validation fold mean correlation R scores between the actual
            and predicted dependent variables (y vs. y_hat).

        Args:
        - ydict (np.ndarray): The dictionary containing the dependent variables y-matrices for each ROI.
        - X (np.ndarray): The independent variables X-matrix.
        - alpha (float): The regularisation parameter of the Ridge regression model.
        - cv (int): The number of cross-validation folds.
        - rois (list): The list of regions of interest (ROIs) to analyse.
        - X_uninformative (np.ndarray): The uninformative X-matrix to compare the model against.
        - fit_icept (bool, optional): Whether or not to fit an intercept. If both X and y matrices are z-scored
                It is highly recommended to set it to False, otherwise detecting effects becomes difficult. Defaults to False.
        - save_outs (bool, optional): Whether or not to save the outputs. Defaults to False.

        Returns:
        - np.ndarray: Object containing the voxel coordinates and the mean R scores for each ROI. This can be
                efficiently turned into a numpy array using NSP.utils.coords2numpy, which in turn can be converted 
                into a nifti file using nib.Nifti1Image(np.array, affine), in which the affine can be extracted
                from a readily available nifti file from the specific subject (using your_nifti.affine).
        """
        comp_X_str = 'Shuffled model' if shuf_or_baseline == 's' else 'Baseline model'
        X_str = 'model' if X_str is None else X_str
        r_values = {}
        r_uninformative = {}
        regcor_dict = {}  # Dictionary to store cor_scores
        regcor_dict['X'] = {}
        # Calculate scores for the given X matrix
        for roi in rois:
            y = ydict[roi]
            model_og = self.run_ridge_regression(X, y, alpha=alpha, fit_icept=False)
            _, cor_scores = self.score_model(X, y, model_og, cv=cv)
            r_values[roi] = np.mean(cor_scores, axis=0)
            regcor_dict['X'][roi] = cor_scores  # Save cor_scores to dictionary

            xyz = voxeldict[roi].xyz
            this_coords = np.hstack((xyz, np.array(r_values[roi]).reshape(-1,1)))
            this_coefs = np.mean(model_og.coef_, axis=1).reshape(-1,1)
            if roi == 'V1':
                coords = this_coords
                beta_coefs = this_coefs
            else:
                coords = np.vstack((coords, this_coords))
                beta_coefs = np.vstack((beta_coefs, this_coefs))


        regcor_dict['X_shuffled'] = {}
        # Calculate scores for the uninformative/baseline X matrix
        for roi in rois:
            y = ydict[roi]
            model_comp = self.run_ridge_regression(X_uninformative, y, alpha=alpha, fit_icept=fit_icept)
            _, cor_scores = self.score_model(X_uninformative, y, model_comp, cv=cv)
            r_uninformative[roi] = np.mean(cor_scores, axis=0)
            regcor_dict['X_shuffled'][roi] = cor_scores  # Save cor_scores to dictionary
            if roi == 'V1':
                uninf_scores = r_uninformative[roi].reshape(-1,1)
            else:
                uninf_scores = np.vstack((uninf_scores, r_uninformative[roi].reshape(-1,1)))

        coords = np.hstack((coords, uninf_scores, beta_coefs)) # also added beta coefficients as last column. very rough but works
        delta_r_df = pd.DataFrame()
        
        for i, roi in enumerate(rois[:4]):
            # Calculate and store the delta-R values 
            this_delta_r = round(np.mean(r_values[roi]) - np.mean(r_uninformative[roi]), 5)
            delta_r_df[roi] = [this_delta_r]
            
            if plot_hist:
                if roi == 'V1': # Create a figure with 4 subplots
                    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                    # Flatten the axs array for easy iteration
                    axs = axs.flatten()
                
                # Underlay with the histogram of r_uninformative[roi] values
                axs[i].hist(r_uninformative[roi], bins=25, edgecolor=None, alpha=1, label=comp_X_str, color='burlywood')
                # Plot the histogram of r_values[roi] values in the i-th subplot
                axs[i].hist(r_values[roi], bins=25, edgecolor='black', alpha=0.5, label=X_str, color='dodgerblue')
                axs[i].set_title(f'{roi} delta-R: {this_delta_r}')
                axs[i].legend() if roi == 'V1' else None

                if roi == 'V4': # Add title and display the figure
                    plt.suptitle(f'{regname}', fontsize=16)
                    plt.tight_layout()
                    plt.show()
                
        if save_outs:
            if save_folder is None:
                if 'unpred' in regname:
                    save_folder = 'unpred'
                elif 'baseline' in regname:
                    save_folder = 'baseline'
                elif 'encoding' in regname:
                    save_folder = 'encoding'
                else: 
                    save_folder = 'various'
                
            save_path = f'{self.nsp.own_datapath}/{subject}/brainstats/{save_folder}'
            os.makedirs(save_path, exist_ok=True)
            
            # Save the delta_r_df to a file
            delta_r_df.to_pickle(f'{save_path}/{regname}_delta_r.pkl')

            plt.savefig(f'{save_path}/{regname}_plot.png')  # Save the plot to a file

            # Save cor_scores to a file
            np.save(f'{save_path}/{regname}_regcor_scores.npy', coords)  # Save the coords to a file

            print(f'Succesfully saved the outputs to {regname}_plot.png and {regname}_regcor_scores.npy')

            # Save the regcor_dict to a file
            with open(f'{save_path}/{regname}_regcor_dict.pkl', 'wb') as f:
                pickle.dump(regcor_dict, f)
            
        return coords, delta_r_df

    def load_regresults(self, subject:str, prf_dict:dict, roi_masks:dict, feattype:str, cnn_layer:Optional[int]=None, 
                    plot_on_viscortex:bool=True, plot_result:Optional[str]='r', lowcap:float=0, upcap:float=None,
                    file_tag:str='', verbose:bool=True, reg_folder:str=''):
        """Function to load in the results from the regressions.

        Args:
        - subject (str): The subject for which the regression results are to be loaded
        - prf_dict (dict): The pRF dictionary
        - roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
        - feattype (str): options: 'rms', 'ce', 'sc', 'unpred', 'alexunet', 'alexown'
        - cnn_layer (int): Optional, only necessary for the 'unpred, 'alexunet', and 'alexown' feature types
        - plot_on_viscortex (bool): Whether to plot the results on the V1 mask
        - plot_result (str): The result to plot, options: 'r', 'r_shuf', 'r_rel', 'betas'
        - lowcap (float): The lower cap for the colourmap when plotting the brain
        - upcap (float): The upper cap, idem.
        - file_tag (str): Optional tag to append to the file name
        - verbose (bool): Whether to print the coords dataframe
        Out:
        - cor_scores_dict (Dict): The dictionary that contains for both the actual X matrix and the shuffled X matrix the
            r correlation scores for every separate cv fold.
        - coords (pd.DataFrame): This dataframe contains the mean r and beta scores over all of the cross-validation folds
        """    
        
        reg_str = f'{feattype}'
        if feattype in ['unpred', 'alexunet', 'alexown'] or cnn_layer is not None:
            if cnn_layer is None:
                raise ValueError('Please provide a cnn_layer number for the feature type you have chosen')
            reg_str = f'{feattype}_lay{cnn_layer}{file_tag}'
        
        # This is the dictionary that contains for both the actual X matrix and the shuffled X matrix the
        # r correlation scores for every separate cv fold.
        with open (f'{self.nsp.own_datapath}/{subject}/brainstats/{reg_folder}/{reg_str}_regcor_dict.pkl', 'rb') as f:
            # Structure: cor_scores_dict['X' or 'X_uninformative'][roi][cross-validation fold]
            cor_scores_dict = pickle.load(f)
            
        # This dataframe contains the mean scores over all of the cross-validation folds
        # coords = pd.DataFrame(np.load(f'{self.nsp.own_datapath}/subj01/brainstats/{reg_str}_regcor_scores.npy'), 
        coords = pd.DataFrame(np.load(f'{self.nsp.own_datapath}/{subject}/brainstats/{reg_folder}/{reg_str}_regcor_scores.npy'), 
                            columns=['x', 'y', 'z', 'r', 'r_shuf', 'beta'])
        
        
        if plot_result == 'r':
            plot_val = 3
        elif plot_result == 'r_shuf':
            plot_val = 4
        elif plot_result == 'r_rel':
            plot_val = 6
            coords['r_rel'] = coords['r'] - coords['r_shuf']
        elif plot_result == 'betas':
            plot_val = 5
            
        if verbose:
            print(coords)

        if plot_on_viscortex:
            brain_np = self.nsp.utils.coords2numpy(np.hstack((np.array(coords)[:,:3],np.array(coords)[:,plot_val].reshape(-1,1))), roi_masks[subject]['V1_mask'].shape, keep_vals=True)

            self.plot_brain(prf_dict, roi_masks, subject, self.nsp.utils.cap_values(np.copy(brain_np), lowcap, upcap), False, save_img=False, img_path='/home/rfpred/imgs/rel_scores_np.png')


            self.nsp.cortex.viscortex_plot(prf_dict=prf_dict, 
                                    vismask_dict=roi_masks, 
                                    plot_param=None, 
                                    subject=subject, 
                                    upcap=upcap, 
                                    lowcap=lowcap, 
                                    inv_colour=False, 
                                    cmap='RdGy_r',
                                    regresult= brain_np)
        
        return cor_scores_dict, coords


    def plot_delta_r(self, subject:str, rois:list, cnn_type:str='alex',
                    file_tag:str='', save_imgs:bool=False, basis_param:str='betas',
                    which_reg:str='unpred'):
        """Function to plot the delta r values that have resulted from regressions across different cnn layers.
        Works for the predictability estimates and for the encoding models. 

        Args:
        - subject (str): The subject.
        - rois (list): List of ROIs to consider.
        - cnn_type (str, optional): The type of CNN model to use. Defaults to 'alex'.
        - file_tag (str, optional): Optional tag to append to the file name. Defaults to ''.
        - save_imgs (bool, optional): Whether to save the images. Defaults to False.
        - basis_param (str, optional): The basis for assigning layers. Defaults to 'betas', alternative is 'r' or 'delta_r'.
        - which_reg (str, optional): The type of regression to use. Defaults to 'unpred'.
        """    
        first_lay = 0
        last_lay = 5 if cnn_type == 'alex' or cnn_type == 'alexnet' else 6
        
        if which_reg == 'unpred':
            feattype = f'{cnn_type}_unpred'
        elif which_reg == 'encoding':
            feattype = 'allvox_alexunet'
            first_lay = 1
            last_lay = 5
            
        n_layers = last_lay - first_lay
        
        if basis_param == 'betas':
            param_col = 5
        elif basis_param == 'r':
            param_col = 3
        elif basis_param == 'delta_r':
            param_col = 6
        
        for layer in range(first_lay,last_lay): # Loop over the layers of the alexnet
            cnn_layer = f'er{str(layer)}' if which_reg == 'encoding' else f'{str(layer)}'
                
            delta_r_layer = pd.read_pickle(f'{self.nsp.own_datapath}/{subject}/brainstats/{feattype}_lay{cnn_layer}{file_tag}_delta_r.pkl').values[0].flatten()
            if layer == first_lay:
                all_delta_r = delta_r_layer
            else:
                all_delta_r = np.vstack((all_delta_r, delta_r_layer))
                
        df = pd.DataFrame(all_delta_r, columns = rois)
        print(df)

        df.reset_index(inplace=True)

        # Melt the DataFrame to long-form or tidy format
        df_melted = df.melt('index', var_name='ROI', value_name='b')
        
        fig, ax = plt.subplots()

        # Create the line plot
        sns.lineplot(x='index', y='b', hue='ROI', data=df_melted, marker='o', ax=ax)

        ax.set_xticks(range(n_layers))  # Set x-axis ticks to be integers from 0 to 4
        ax.set_xlabel(f'{cnn_type} Layer')
        ax.set_ylabel('Delta R Value')
        ax.set_title(f'Delta R Value per {cnn_type} Layer')

        if save_imgs:
            # Save the plot
            fig.savefig(f'{self.nsp.own_datapath}/{subject}/brainstats/{feattype}_lay{cnn_layer}{file_tag}_delta_r_plot.png')

        plt.show()

    def assign_layers_old(self, subject:str, 
                      prf_dict:dict, 
                      roi_masks:dict, 
                      rois:list, 
                      cmap, 
                      cnn_type:str='alex', 
                      plot_on_brain:bool=True, 
                      file_tag:str='', 
                      save_imgs:bool=False, 
                      basis_param:str='betas',
                      which_reg:str='unpred', 
                      man_title:(str | None)=None, 
                      return_nifti:bool=False,
                      first_lay:(int | None)=None,
                      last_lay:(int | None)=None,
                      direct_folder:(str | None)=None):
        """
        Assigns layers to voxels based on the maximum beta value across layers for each voxel.

        Args:
            subject (str): The subject.
            prf_dict (dict): Dictionary containing pRF model results.
            roi_masks (dict): Dictionary containing ROI masks.
            rois (list): List of ROIs to consider.
            cnn_type (str, optional): The type of CNN model to use. Defaults to 'alex'.
            plot_on_brain (bool, optional): Whether to plot the results on the brain. Defaults to True.
            file_tag (str, optional): Optional tag to append to the file name. Defaults to ''.
            save_imgs (bool, optional): Whether to save the images. Defaults to False.
            basis_param (str, optional): The basis for assigning layers. Defaults to 'betas', alternative is 'r' or 'delta_r'.
            which_reg (str, optional): The type of regression to use. Defaults to 'unpred'.
        """    
        first_lay = 0
        last_lay = 5 if cnn_type == 'alex' or cnn_type == 'alexnet' else 6
        n_layers = last_lay - first_lay

        if basis_param == 'betas':
            param_col = 5
        elif basis_param == 'r':
            param_col = 3
        elif basis_param == 'delta_r':
            param_col = 6
            
        if direct_folder is None:
            if which_reg == 'unpred':
                feattype = f'{cnn_type}_unpred'
            elif which_reg == 'encoding':
                feattype = 'allvox_alexunet'
            elif which_reg == 'encoding_smallpatch':
                feattype = 'smallpatch_allvox_alexunet'
                
        
            for layer in range(first_lay,last_lay): # Loop over the layers of the alexnet
                cnn_layer = f'er{str(layer)}' if which_reg == 'encoding' or which_reg == 'encoding_smallpatch' else f'{str(layer)}'
                cordict, coords = self.load_regresults(subject, prf_dict, roi_masks, feattype, cnn_layer, plot_on_viscortex=False, plot_result='r', file_tag=file_tag, verbose=False, reg_folder=which_reg)
                
                coords['delta_r'] = coords['r'] - coords['r_shuf'] # Compute the delta_r values
                    
                if layer == first_lay:
                    all_betas = np.hstack((np.array(coords)[:,:3], np.array(coords)[:,param_col].reshape(-1,1)))
                else:
                    all_betas = np.hstack((all_betas, np.array(coords)[:,param_col].reshape(-1,1)))
        else:
            layer = first_lay if first_lay is not None else 0

            # Get the list of filenames
            filenames = os.listdir(f'{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}')

            # Sort the filenames based on the layer number
            sorted_filenames = sorted(filenames, key=self.nsp.utils._extract_layno)
            print(sorted_filenames)
            
            for layerfile in sorted_filenames:
            # for layerfile in os.listdir(f'{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}'):
                print(layerfile)
                if fnmatch.fnmatch(layerfile, '*regcor_dict.pkl'):
                    with open(f'{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}/{layerfile}', 'rb') as f:
                        cordict = pickle.load(f)
                if fnmatch.fnmatch(layerfile, '*regcor_scores.npy'):
                    coords_np = np.load(f'{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}/{layerfile}')
                    coords = pd.DataFrame(coords_np, columns=['x', 'y', 'z', 'r', 'r_shuf', 'beta'])
                    del coords_np
                    print(f'Now looking at layer numero {layer}')
                    
                    coords['delta_r'] = coords['r'] - coords['r_shuf'] # Compute the delta_r values
             
                    if layer == first_lay:
                        all_betas = np.hstack((np.array(coords)[:,:3], np.array(coords)[:,param_col].reshape(-1,1)))
                    else:
                        all_betas = np.hstack((all_betas, np.array(coords)[:,param_col].reshape(-1,1)))
                    layer += 1
                    
                
        for n_roi, roi in enumerate(rois):
            n_roivoxels = len(cordict['X'][roi][0])
            
            if roi == 'V1':
                vox_of_roi = np.ones((n_roivoxels, 1))
            else:
                vox_of_roi = (np.vstack((vox_of_roi, (np.ones((n_roivoxels, 1))* (n_roi + 1))))).astype(int)

        all_betas_voxroi = np.hstack((all_betas, vox_of_roi))[:,3:]
        all_betas_voxroi[:,:n_layers] = stats.zscore(all_betas_voxroi[:,:n_layers], axis=0)
        print(all_betas_voxroi)

        # Get the index of the maximum value in each row, excluding the last column
        max_indices = np.argmax(all_betas_voxroi[:, :-1], axis=1) + 1 # Add 1 to the max_indices to get the layer number
        print(max_indices)
        barcmap = LinearSegmentedColormap.from_list('NavyBlueVeryLightGreyDarkRed', ['#000080', '#CCCCCC', '#FFA500', '#FF0000'], N=n_layers)
        
        # Create a new colourmap for the glass brain plot, as it has difficulty adapting the colourmap to non-symmetrical/positive values
        colors = np.concatenate([barcmap(np.linspace(0, 1, 128)), barcmap(np.linspace(0, 1, 128))])
        glass_cmap = ListedColormap(colors, name='double_lay_assign')

        # Create a DataFrame from the array
        df = pd.DataFrame(all_betas_voxroi, columns=[f'col_{i}' for i in range(all_betas_voxroi.shape[1])])

        # Rename the last column to 'ROI'
        df.rename(columns={df.columns[-1]: 'ROI'}, inplace=True)

        # Add the max_indices as a new column
        df['AlexNet layer'] = max_indices

        # Convert the 'ROI' column to int for plotting
        df['ROI'] = df['ROI'].astype(int)

        # Calculate the proportions of max_indices within each ROI
        df_prop = (df.groupby('ROI')['AlexNet layer']
                    .value_counts(normalize=True)
                    .unstack(fill_value=0))

        # Create a mapping from old labels to new labels
        roi_mapping = {1: 'V1', 2: 'V2', 3: 'V3', 4: 'V4'}

        # Change the labels on the x-axis
        df_prop.rename(index=roi_mapping, inplace=True)

        # Plot the proportions using a stacked bar plot
        ax = df_prop.plot(kind='bar', stacked=True, colormap=barcmap)

        # Add a y-axis label
        ax.set_ylabel('Layer assignment (%)')

        # Get current handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Reverse handles and labels
        handles, labels = handles[::-1], labels[::-1]
        
        # Create legend
        legend = plt.legend(handles, labels, title='CNN\nLayer', loc='center right', bbox_to_anchor=(1.15, 0.5),
                ncol=1, fancybox=False, shadow=False, fontsize=10)
        
        if man_title is None:
            plt.title(f'Layer assignment {which_reg} {cnn_type} {basis_param}-based')
        else:
            plt.title(man_title)
            
        if save_imgs:
            save_path = f'{self.nsp.own_datapath}/{subject}/brainstats/{cnn_type}_unpred_layassign{file_tag}'
            # Save the plot
            plt.savefig(f'{save_path}.png')
        else: save_path = ''

        plt.show()
        if plot_on_brain:    
            self.stat_on_brain(prf_dict, roi_masks, 'subj01', 
                               max_indices, 
                               all_betas[:,:3].astype(int), 
                               glass_brain=True, 
                               cmap=glass_cmap, 
                               save_img=save_imgs, 
                               img_path=f'{save_path}_glassbrain.png')
            
        if return_nifti:
            brain_coords = np.hstack((all_betas[:,:3].astype(int), (max_indices.reshape(-1,1)))) # Earlier I had +1 here
            brain_np = self.nsp.utils.coords2numpy(brain_coords, roi_masks['subj01']['V1_mask'].shape, keep_vals=True)
            brain_nii = nib.Nifti1Image(brain_np, affine=self.nsp.cortex.anat_templates(prf_dict)[subject].affine)
            if save_imgs:
                nib.save(brain_nii, f'{save_path}.nii')
            return brain_nii
            
            
    
    def explained_var_plot(self, relu_layer:int, n_pcs:int, smallpatch:bool=False):
        """Method to plot the explained variance ratio of the PCA instance created.
        
        Args:
        - relu_layer (int): Which ReLU layer of the AlexNet used for the encoding analyses
                to plot the explained variance ratio of. Options are 1, 2, 3, 4, and 5.
                These correspond to overall layers 1, 4, 7, 9, 11 in the CNN. 
        - n_pcs (int): The number of principle components. Options are 1000 or 600. 
        """        
        
        print('koekjes')
        
        
        smallpatch_str = 'smallpatch_' if smallpatch else ''
            
        # pca_instance = joblib.load(f'{self.nsp.own_datapath}/visfeats/cnn_featmaps/pca/pca_{smallpatch_str}{relu_layer}_{n_pcs}pcs.joblib')
        pca_instance = joblib.load(f'{self.nsp.own_datapath}/visfeats/cnn_featmaps/pca_{smallpatch_str}{relu_layer}_{n_pcs}pcs.joblib')
        # Create a figure and a set of subplots
        fig, ax = plt.subplots()

        # Number of components
        n_components = np.arange(1, len(pca_instance.explained_variance_ratio_) + 1)
        cumulative_explained_variance_ratio = np.cumsum(pca_instance.explained_variance_ratio_)
        # Plot the explained variance ratio
        ax.bar(n_components, pca_instance.explained_variance_ratio_, alpha=0.5,
        align='center', label='individual explained variance')

        # Plot the cumulative explained variance ratio
        ax.step(n_components, cumulative_explained_variance_ratio, where='mid',
                label='cumulative explained variance')

        # Add a horizontal line at y=0.95
        ax.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')

        # Add labels and title
        ax.set_xlabel('Principal components')
        ax.set_ylabel('Explained variance ratio')
        ax.set_title(f'Explained variance ratio across all {n_pcs} PCs of layer {relu_layer}')

        # Add a legend
        ax.legend(loc='best')

        # Show the plot
        plt.show()
