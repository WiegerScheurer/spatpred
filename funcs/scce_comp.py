import sys
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imp
import yaml
import cv2
import lgnpy.CEandSC.lgn_statistics
from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN
import time
import pandas as pd
import os
import h5py
import cv2
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy
import yaml
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression, Ridge
from matplotlib.backends.backend_pdf import PdfPages
import mat73
from scipy import io
from scipy.ndimage import binary_dilation
from multiprocessing import Pool
import random

# Change working directory
os.chdir('/home/rfpred/notebooks/alien_nbs')

# Function to show a randomly selected image of the nsd dataset
def show_stim(hide = 'n', img_no = 'random', small = 'n'):
    # Example code to show how to access the image files, these are all 73000 of them, as np.arrays
    # I keep it like this as it might be useful to also store the reconstructed images with the autoencoder
    # using a .hdf5 folder structure, but I can change this later on.

    stim_dir = '/home/rfpred/data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/'
    stim_files = os.listdir(stim_dir)

    with h5py.File(f'{stim_dir}{stim_files[0]}', 'r') as file:
        img_brick_dataset = file['imgBrick']
        
        if img_no == 'random':
            image_no = random.randint(0,img_brick_dataset.shape[0])
        else: image_no = img_no
        
        test_image = img_brick_dataset[image_no]
    hor = ver = 10
    if small == 'y':
        hor = ver = 5        
    if hide == 'n':
        plt.figure(figsize=(hor, ver))
        plt.imshow(test_image)
        plt.title(f'Image number {image_no}')
        plt.axis('off')
        plt.show()
        
    return test_image, image_no

# Function to create a circle mask
def make_circle_mask(size, center_row, center_col, radius, fill='y', margin_width=1):
    rows = np.arange(size)
    cols = np.arange(size)
    rows, cols = np.meshgrid(rows, cols)

    # Calculate the distance from each point to the center
    distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)

    # Create a binary mask where values within the radius are set to 1, others to 0
    circle_mask = np.where(distances <= radius, 1, 0)

    # Create a dilated version of the binary mask to add a margin
    dilated_circle_mask = binary_dilation(circle_mask, iterations=margin_width)

    # Subtract the dilated version to create the outline
    outline_circle_mask = circle_mask - dilated_circle_mask

    if fill == 'y':
        return circle_mask
    elif fill == 'n':
        return -outline_circle_mask
    
def css_gaussian_cut(size, center_row, center_col, sigma):
    rows = np.arange(size)
    cols = np.arange(size)
    rows, cols = np.meshgrid(rows, cols)

    distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
    mask = np.where(distances <= sigma, 1, 0)

    exponent = -((rows - center_row)**2 / (2 * sigma**2) + (cols - center_col)**2 / (2 * sigma**2))
    gaussian = np.exp(exponent)
    gaussian *= mask
    return gaussian

def get_bounding_box(mask):
    # Get the indices where the mask is True
    y_indices, x_indices = np.where(mask)

    # Get the minimum and maximum indices along each axis
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    return x_min, x_max, y_min, y_max

from multiprocessing import Pool

# These two functions are coupled to run the feature computations in parallel.
# This saves a lot of time. Should be combined with the feature_df function to assign
# the values to the corresponding trials.
def scce_single(args, ecc_max = 1, loc = 'center', plot = 'n', cmap = 'gist_gray'):
# def scce_single(args, ecc_max = 1, loc = 'center', plot = 'n', crop_prior:bool = False, crop_post:bool = False, save_plot:bool = False, cmap = 'gist_gray'):
    i, start, n, plot, loc, crop_prior, crop_post, save_plot  = args
    dim = show_stim(hide = 'y')[0].shape[0]
    radius = ecc_max * (dim / 8.4)
    
    if loc == 'center':
        x = y = (dim + 1)/2
    elif loc == 'irrelevant_patch':
        x = y = radius + 10
        
    mask_w_in = css_gaussian_cut(dim, x, y, radius)
    rf_mask_in = make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
    full_ar_in = ar_in = show_stim(img_no = i, hide = 'y')[0]  
    
    if i % 100 == 0:
        print(f"Processing image number: {i} out of {n + start}")
        
        
    # Crop the image first, then provide this as input to the visfeat function
    if crop_prior:
        
        x_min, x_max, y_min, y_max = get_bounding_box(rf_mask_in)
        ar_in = ar_in[x_min:x_max, y_min:y_max]
        mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
        rf_mask_in = rf_mask_in[x_min:x_max, y_min:y_max]
        
    # return get_rms_contrast_lab(ar_in, mask_w_in, rf_mask_in, full_ar_in, normalise = normalise, 
    #                             plot = plot, cmap = cmap, crop_prior = crop_prior, crop_post = crop_post, 
    #                             save_plot = save_plot)
    
    return get_scce_contrast(ar_in, mask_w_in, rf_mask_in, full_ar_in, plot = plot, cmap = cmap, 
                             crop_prior = crop_prior, crop_post = crop_post, save_plot = save_plot)

def scce_all(start, n, ecc_max = 1, plot = 'n', loc = 'center', crop_prior:bool = False, crop_post:bool = True, save_plot:bool = False):
    img_vec = list(range(start, start + n))

    # Create a pool of worker processes
    with Pool() as p:
        scce_vec = p.map(scce_single, [(i, start, n, plot, loc, crop_prior, crop_post, save_plot) for i in img_vec])

    # Unpack scce_vec into separate lists
    ce, sc, beta, gamma = zip(*scce_vec)

    scce_dict = pd.DataFrame({
        'ce': ce,
        'sc': sc,
        'beta': beta,
        'gamma': gamma
    })

    scce_dict = scce_dict.set_index(np.array(img_vec))
    return scce_dict

# Function that calculates rms but based on a RGB to LAB conversion, which follows the CIELAB colour space
# This aligns best with the way humans perceive visual input. 
def get_scce_contrast(rgb_image, mask_w_in, rf_mask_in, full_array, plot = 'n', cmap = 'gist_gray', 
                      crop_prior:bool = False, crop_post:bool = False, save_plot:bool = False):



    lgn_out = lgn_statistics(im=rgb_image, file_name='noname.tiff',
                                        config=config, force_recompute=True, cache=False,
                                        home_path='./', verbose = False, verbose_filename=False,
                                        threshold_lgn=threshold_lgn, compute_extra_statistics=False,
                                        crop_prior = True)
    
    ce = np.mean(lgn_out[0][:, :, 0])
    sc = np.mean(lgn_out[1][:, :, 0])
    beta = np.mean(lgn_out[2][:, :, 0])
    gamma = np.mean(lgn_out[3][:, :, 0])
    
    if plot == 'y':
        fig,axs = plt.subplots(2,3, figsize=(15,10))
        axs[0,0].imshow(lgn_out[4]['par1'], cmap=cmap)
        axs[0,0].axis('off')
        axs[0,1].imshow(lgn_out[4]['par2'], cmap=cmap)
        axs[0,1].axis('off')
        axs[0,2].imshow(lgn_out[4]['par3'], cmap=cmap)
        axs[0,2].axis('off')
        axs[1,0].imshow(lgn_out[4]['mag1'], cmap=cmap)
        axs[1,0].axis('off')
        axs[1,1].imshow(lgn_out[4]['mag2'], cmap=cmap)
        axs[1,1].axis('off')
        axs[1,2].imshow(lgn_out[4]['mag3'], cmap=cmap)
        axs[1,2].axis('off')

        if save_plot:
            fig.savefig(f'rms_crop_prior_{str(crop_prior)}_crop_post_{str(crop_post)}.png')
            
    return ce, sc, beta, gamma


config_path = 'lgnpy/lgnpy/CEandSC/default_config.yml'

with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

lgn = LGN(config=config, default_config_path=f'./lgnpy/lgnpy/CEandSC/default_config.yml')

threshold_lgn = loadmat(filepath='./lgnpy/ThresholdLGN.mat')['ThresholdLGN']


# Define the steps
steps = [20000, 20000, 20000, 13000]
start = [0, 20000, 40000, 60000]

# steps = [10, 10, 10, 10]
# start = [0, 10, 20, 30]

# Print the steps
for i in range(4):
    scce_dict_center = scce_all(start[i], steps[i], plot = 'n' , loc = 'center', crop_prior = True, crop_post = False)
    scce_dict_center.to_pickle(f'scce_dict_center_{str(start[i])[:2]}k_{str(steps[i] + start[i])[:2]}k.pkl')
    
    print(f'last center save was{str(start[i])[:2]}k_{str(steps[i] + start[i])[:2]}k.pkl')
    scce_dict_irrelpatch = scce_all(start[i], steps[i], plot = 'n' , loc = 'irrelevant_patch', crop_prior = True, crop_post = False)
    scce_dict_irrelpatch.to_pickle(f'scce_dict_irrelpatch_{str(start[i])}k_{str(steps[i] + start[i])[:2]}k.pkl')
    print(f'last irrelevant patch save was{str(start[i])[:2]}k_{str(steps[i] + start[i])[:2]}k.pkl')
    
print('succeeded')

# scce_dict = {}
# ces = []
# scs = []

# (ce, sc, beta, gamma, edge_dict)
# for current_img in range(10):

#     picca = show_stim(img_no = current_img, hide='y')[0]

#     # This is the circle that corresponds to the middle 2 degs
#     zirkel = make_circle_mask(425, 213, 213, 1 * 425/8.4, fill='y', margin_width = 1)

#     box_1deg = get_bounding_box(zirkel)

#     picca_crop = picca[box_1deg[0]:box_1deg[1], box_1deg[2]:box_1deg[3]]

#     lgn_out = lgn_statistics(im=picca_crop, file_name='noname.tiff',
#                                             config=config, force_recompute=True, cache=False,
#                                             home_path='./', verbose = False, verbose_filename=False,
#                                             threshold_lgn=threshold_lgn, compute_extra_statistics=False,
#                                             crop_prior = True)

#     ces = np.append(ces, round(np.mean(lgn_out[0][:, :, 0]), 10))
#     scs = np.append(scs, round(np.mean(lgn_out[1][:, :, 0]), 10))

#     scce_dict['ce'] = ces
#     scce_dict['sc'] = scs

# print(scce_dict)