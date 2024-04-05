import os
import sys
os.chdir('/home/rfpred')
sys.path.append('/home/rfpred')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

import torchvision
import skimage
import os
import zipfile
import requests
import shutil
import torch
import random
import time

from funcs.rf_tools import (get_dat, calculate_sigma, calculate_pRF_location, prf_plots_new, prf_plots, make_visrois_dict, 
                            make_gaussian_2d, make_circle_mask, css_gaussian_cut, roi_filter, write_prf_dict, compare_radius, 
                            get_mask, compare_masks, prf_heatmap, nsd_R2_dict, rsq_to_size, rsquare_selection)
from funcs.utility import print_dict_structure, print_large
from funcs.imgproc import show_stim, get_img_prf, get_contrast_df, get_rms_contrast, get_imgs_designmx

# /Users/wiegerscheurer/miniconda3/envs/wieg_env_nsd

#@title import model class and utility functions
from unet_recon.inpainting import UNet
from PIL import Image
import glob,copy
import matplotlib.pyplot as plt
import numpy as np

# imgs=glob.glob('unet_recon/examples/img*.jpg')

def draw_circmask(dims, maskrad, offset=(0,0), invert=True):
    import numpy as np
    y, x = np.ogrid[:dims[0], :dims[1]]
    center_x, center_y = dims[1] // 2 + offset[0], dims[0] // 2 + offset[1]
    mask = (x - center_x)**2 + (y - center_y)**2 <= maskrad**2
    return ~mask if invert else mask
def square_circle(mask_in,mask_val=1):
    """
    from a circular mask, get the square mask outlining the circle

    in:
    - mask_in: ndarray, (2d or 3d)
        boolean-type mask image
    - maskv_val: float/int/bool (default:1)
        the value to look for as the definition of in the circle of the mask.

    out:
    -square_mask: ndarray
        like the circular input mask, but now with a square outine around the mask
    """
    def _square_mask(_mask_in,maskval=1):
        """local func doing the work"""
        mask_out=copy.deepcopy(_mask_in)
        nz_rows,nz_cols=np.nonzero(_mask_in==maskval)
        nz_r,nz_c=np.unique(nz_rows),np.unique(nz_cols)
        mask_out[np.min(nz_r):np.max(nz_r)+1,np.min(nz_c):np.max(nz_c)+1]=maskval
        return(mask_out)

    # switch dealing with RGB vs grayscale images
    if mask_in.ndim==3:
        mask_square=_square_mask(mask_in[:,:,0],maskval=mask_val)
        return(_make_img_3d(mask_square))
    elif mask_in.ndim==2:
        return(_square_mask(mask_in,maskval=mask_val))
    else:
        raise ValueError('can only understand 3d (RGB) or 2d array images!')

def slice_array_with_mask(arr_in, mask_in):
    """
    Slices a 2D array using a 2D boolean mask with a contiguous square of True values.

    :param arr_in: 2D numpy array.
    :param mask_in: 2D boolean numpy array of the same shape as arr_in.
    :return: Sliced section of arr_in corresponding to the True values in mask_in.
    """
    # Find the indices of the mask where the value is True
    rows, cols = np.where(mask_in)
    top_left = (min(rows), min(cols))
    bottom_right = (max(rows), max(cols))

    # Slice the array
    return arr_in[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

def scale_square_mask(mask_in:np.ndarray, scale_fact=np.sqrt(1.5), mask_val=1, min_size=50):
    """given a square mask, scale width and height with a given factor

    in:
    - mask_in: ndarray, (2d or 3d)
        boolean-type mask image
    - mask_val: float/int/bool (default:1)
        the value to look for as the definition of in the circle of the mask.
    - min_size: int
        minimum size of the square mask.

    out:
    -scaled_mask: ndarray
        like the square input mask, but now with a square outline around the mask
    """
    def _do_scaling(_mask_in:np.ndarray, scale_fact=np.sqrt(2), mask_val=1, min_size=50):
        """inner function doing the actual scaling"""
        mask_out=copy.deepcopy(_mask_in)
        nz_rows,nz_cols=np.nonzero(_mask_in==mask_val)
        nz_r,nz_c=np.unique(nz_rows),np.unique(nz_cols)
        # determine square masks that spans the circle
        width, height = nz_r[-1]-nz_r[0], nz_c[-1]-nz_c[0]

        # make actual spanning mask a bit larger (delta determined by scale_fact or min_size)
        ideal_delta_w = max(np.round(((width*scale_fact) - width)*.5), (min_size - width) // 2)
        ideal_delta_h = max(np.round(((height*scale_fact) - height)*.5), (min_size - height) // 2)

        # Adjust deltas based on mask's proximity to image borders
        delta_w_left = min(ideal_delta_w, nz_c[0])
        delta_w_right = min(ideal_delta_w, mask_out.shape[1] - nz_c[-1] - 1)
        delta_h_top = min(ideal_delta_h, nz_r[0])
        delta_h_bottom = min(ideal_delta_h, mask_out.shape[0] - nz_r[-1] - 1)

        # If mask is near the border, expand on the other side
        if delta_w_left < ideal_delta_w:
            delta_w_right = max(ideal_delta_w * 2 - delta_w_left, delta_w_right)
        if delta_w_right < ideal_delta_w:
            delta_w_left = max(ideal_delta_w * 2 - delta_w_right, delta_w_left)
        if delta_h_top < ideal_delta_h:
            delta_h_bottom = max(ideal_delta_h * 2 - delta_h_top, delta_h_bottom)
        if delta_h_bottom < ideal_delta_h:
            delta_h_top = max(ideal_delta_h * 2 - delta_h_bottom, delta_h_top)

        mask_out[int(nz_r[0]-delta_h_top):int(nz_r[-1]+delta_h_bottom),
                 int(nz_c[0]-delta_w_left):int(nz_c[-1]+delta_w_right)] = mask_val
        # set values to 1, square mask
        return(mask_out)

    # switch dealing with RGB [colmns,rows,colours] vs grayscale images [columns,rows]
    if mask_in.ndim==3:
        mask_scaled=_do_scaling(mask_in[:,:,0],scale_fact=scale_fact, mask_val=mask_val, min_size=min_size)
        return(_make_img_3d(mask_scaled))
    elif mask_in.ndim==2:
        return(_do_scaling(mask_in, scale_fact=scale_fact, mask_val=mask_val, min_size=min_size))
    else:
        raise ValueError('can only understand 3d (RGB) or 2d array images!')

def _make_img_3d(mask_in,):
    """for 2d array, copy to make 3-dimensional"""
    return(np.repeat(mask_in[:,:,np.newaxis],3,axis=2))



welke_plaat = 1354
plaatje = show_stim(img_no = welke_plaat, small = 'y')
ecc_max = 1
dim = plaatje[0].shape[0]
radius = ecc_max * (dim / 8.4)
loc = 'center'

if loc == 'center':
    x = y = (dim + 1)/2
elif loc == 'irrelevant_patch':
    x = y = radius + 10

# The in stands for inverse in this case. For the inpainting we need the np.bool_ non-inverse images
mask_w_in = css_gaussian_cut(dim, x, y, radius)
rf_mask_in = make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
full_ar_in = ar_in = show_stim(img_no = welke_plaat, hide = 'y')[0] 
# Get the boolean version of the non-inverse mask
rf_mask_nsd = rf_mask_in == 0

# Load in the U-Net model
unet=UNet(checkpoint_name='pconv_circ-places20k.pth',feature_model='alex')



def rand_img_list(n_imgs, asPIL:bool = True, add_masks:bool = True, mask_loc = 'center', ecc_max = 1):
    imgs = []
    img_nos = []
    for i in range(n_imgs):
        img_no = random.randint(0, 27999)
        img = show_stim(img_no = img_no, hide = 'y')[0]

        if i == 0:
            dim = img.shape[0]
            radius = ecc_max * (dim / 8.4)

            if mask_loc == 'center':
                x = y = (dim + 1)/2
            elif mask_loc == 'irrelevant_patch':
                x = y = radius + 10

        if asPIL:
            img = Image.fromarray(img)

        imgs.append(img)
        # img_nos.append(Image.fromarray(img_no))
        img_nos.append(img_no)
    mask = (make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0) == 0)

    if asPIL:
        mask = Image.fromarray(mask)

    masks = [mask] * n_imgs

    return imgs, masks, img_nos


# Retrieve a number of random images, masks
n_imgs = 5
imgs, masks, img_nos = rand_img_list(n_imgs, asPIL = True, add_masks = True, mask_loc = 'center', ecc_max = 1)

start_time = time.time()

# Run them through the U-Net
payload_nsd=unet.analyse_images(imgs, masks, return_recons=True)

end_time = time.time()

total_time = end_time - start_time
average_time_per_image = total_time / n_imgs

print(f"Average time per image: {average_time_per_image} seconds")