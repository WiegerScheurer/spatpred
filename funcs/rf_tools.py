print('diarreeklont')
import os
import sys
from tkinter import Y
import nipype
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import nsdcode
from nsdcode.nsd_mapdata import NSDmapdata
from nsdcode.nsd_datalocation import nsd_datalocation
from nsdcode.nsd_output import nsd_write_fs
from nsdcode.utils import makeimagestack
from scipy.ndimage import binary_dilation
import io
import json
# from scipy.ndimage import gaussian_filter
# import cv2
import random
import time
import os
import seaborn as sns
import pprint as pprint


# Function to load in nifti (.nii.gz) data and create some useful variables 
def get_dat(path):

    full_dat = nib.load(path)
    dat_array = full_dat.get_fdata()
    # Calculate the range of values
    flat_arr = dat_array[~np.isnan(dat_array)]

    dat_dim = dat_array.shape

    return full_dat, dat_array, dat_dim, {'min': round(np.nanmin(flat_arr),7), 'max': np.nanmax(flat_arr), 'mean': round(np.nanmean(flat_arr),5)}
  
# Function to binarize a list of non binary masks, by providing both the directory
# in which theses masks are located, and the mask names in the form of a list. 
# Need to make sure the file loading works the same as it did in colab.


def calculate_sigma(eccentricity, angle, visual_stimulus_size=8.4):
    # Convert polar coordinates to Cartesian coordinates
    x = eccentricity * np.cos(angle)
    y = eccentricity * np.sin(angle)

    # Calculate the scaling factor based on visual stimulus size and eccentricity range
    eccentricity_range = 1000  
    scaling_factor = visual_stimulus_size / eccentricity_range

    # Calculate sigma
    sigma = np.sqrt(x**2 + y**2) * scaling_factor

    return sigma, x, y

def calculate_pRF_location(prf_size, prf_ecc, prf_angle, image_size=(200, 200), visual_angle_extent=8.4):
    # Calculate sigma parameter in degrees visual angle
    sigma = prf_size * np.sqrt(2)
    
    # Calculate sigma parameter in pixel units
    sigma_px = sigma * (image_size[0] / visual_angle_extent)
    
    # Calculate pRF y-position in row pixel units
    r_index = (image_size[0] + 1) / 2 - (prf_ecc * np.sin(np.radians(prf_angle)) * (image_size[0] / visual_angle_extent))
    
    # Calculate pRF x-position in column pixel units
    c_index = (image_size[1] + 1) / 2 + (prf_ecc * np.cos(np.radians(prf_angle)) * (image_size[0] / visual_angle_extent))
    
    return sigma_px, r_index, c_index

def prf_plots_new(subj_no, bottom_percent=95):
    prf_info = ['angle', 'eccentricity', 'size']
    prf_plot_dict = {}
    
    # Load data for angle, eccentricity, and size
    for idx in prf_info:
        _, prf_plot_dict[idx], _, prf_range = get_dat(f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/subj0{subj_no}/func1mm/prf_{idx}.nii.gz')
    
    # Calculate the common boolean mask based on exclusion criteria
    common_mask = (~np.isnan(prf_plot_dict['eccentricity'])) & (prf_plot_dict['eccentricity'] < 1000) & (prf_dict['size']<1000)  # Adjust conditions as needed

    # Apply the common boolean mask to all relevant arrays
    for key in prf_info:
        prf_dict[key] = prf_dict[key][common_mask]

    # Calculate sigma, x, and y
    sigma_array, x, y = calculate_sigma(prf_dict['eccentricity'], prf_dict['angle'])

    # Add sigma_array, x, and y to prf_dict
    prf_dict.update({'sigma': sigma_array, 'x': x, 'y': y})

    # Calculate pRF location for each voxel
    prf_dict['sigma_px'], prf_dict['r_index'], prf_dict['c_index'] = calculate_pRF_location(
        prf_dict['size'], prf_dict['eccentricity'], prf_dict['angle']
    )

    # Plot histograms for all dictionary elements excluding NaN and large values
    for key, value in prf_dict.items():
        if key in ['x', 'y']:
            # Skip histograms for 'x' and 'y'
            continue
        
        plt.figure()
        
        # Determine adaptive binning based on the range of valid values
        num_bins = min(50, int(np.sqrt(len(value))))  # Adjust as needed
        
        # Apply logarithmic scale to the data for better visibility
        plt.hist(np.log1p(value.flatten()), bins=num_bins, color='red', alpha=0.7)  # Using np.log1p to handle zero values
        plt.title(f'Histogram for {key} (excluding NaN and values > 950)')
        plt.xlabel(f'Log({key} + 1)')  # Adding 1 to avoid log(0)
        
        plt.ylabel('Frequency')
        plt.show()

    # Scatter plot for 'x' and 'y' as coordinates
    plt.figure()
    plt.scatter(prf_dict['x'], prf_dict['y'], c=prf_dict['size'], cmap='cividis', s=prf_dict['size'], alpha=0.7)
    plt.colorbar(label='Size (degrees of visual angle)')
    plt.title('pRF locations based on Eccentricity-Angle, coloured for pRF size')
    plt.xlabel('Eccentricity (degrees)')
    plt.ylabel('Angle (degrees)')
    plt.show()

    # Scatter plot for 'r_index', 'c_index' and 'sigma_px' as coordinates
    # plt.figure()
    # plt.scatter(prf_dict['c_index'], prf_dict['r_index'], c=prf_dict['sigma'], cmap='cividis', s=20)
    # plt.colorbar(label='Sigma')
    # plt.title('pRF locations based on Row-Column, coloured for sigma value')
    # plt.xlabel('Column Index (pixel units)')
    # plt.ylabel('Row Index (pixel units)')
    # plt.show()
    
    return prf_plot_dict

# This one works
def prf_plots(subj_no, bottom_percent=95):
    prf_info = ['angle', 'eccentricity', 'size']
    prf_dict = {}
    
    # Load data for angle, eccentricity, and size
    for idx in prf_info:
        _, prf_dict[idx], _, prf_range = get_dat(f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/subj0{subj_no}/func1mm/prf_{idx}.nii.gz')
    
    # Calculate the common boolean mask based on exclusion criteria
    common_mask = (~np.isnan(prf_dict['eccentricity'])) & (prf_dict['eccentricity'] <= 950)  # Adjust conditions as needed

    # Apply the common boolean mask to all relevant arrays
    for key in prf_info:
        prf_dict[key] = prf_dict[key][common_mask]

    # Calculate sigma, x, and y
    sigma_array, x, y = calculate_sigma(prf_dict['eccentricity'], prf_dict['angle'])

    # Add sigma_array, x, and y to prf_dict
    prf_dict.update({'sigma': sigma_array, 'x': x, 'y': y})

    # Plot histograms for all dictionary elements excluding NaN and large values
    for key, value in prf_dict.items():
        if key in ['x', 'y']:
            # Skip histograms for 'x' and 'y'
            continue
        
        plt.figure()
        
        # Determine adaptive binning based on the range of valid values
        num_bins = min(50, int(np.sqrt(len(value))))  # Adjust as needed
        
        plt.hist(value.flatten(), bins=num_bins, color='red', alpha=0.7)
        plt.title(f'Histogram for {key} (excluding NaN and values > 950)')
        plt.xlabel(key)
        
        plt.ylabel('Frequency')
        plt.show()

    # Scatter plot for 'x' and 'y' as coordinates
    plt.figure()
    plt.scatter(prf_dict['x'], prf_dict['y'], c=prf_dict['sigma'], cmap='cividis', s=20)
    plt.colorbar(label='Sigma')
    plt.title('pRF locations based on Eccentricity-Angle, coloured for sigma value')
    plt.xlabel('Eccentricity (degrees)')
    plt.ylabel('Angle (degrees)')
    plt.show()
    
    return prf_dict

def make_visrois_dict(vox_count = 'n', bin_check = 'n', n_subjects = None):
    binary_masks = {}

    for subj_no in range(1, n_subjects + 1):
        print(f'Subject {subj_no}')
        mask_dir = f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/subj0{subj_no}/func1mm/roi'

        # read in and sort all the filenames in the mapped masks folder for each subject
        
        non_binary_masks = sorted([file for file in os.listdir(mask_dir) if '_mask.nii' in file])
        subj_binary_masks = {}

        for idx, mask_file in enumerate(non_binary_masks):
            # Load the mask file
            subj_binary_masks[non_binary_masks[idx][:-7]] = (nib.load(os.path.join(mask_dir, mask_file)).get_fdata()).astype(int)

        if vox_count == 'y':
            for key, subj_binary_mask in subj_binary_masks.items():
                print(key)
                print(f"Non-zero voxels in {key}: {np.sum(subj_binary_mask)}")

        # Print the maximum value of each binary mask for verification
        if bin_check == 'y':
            for key, subj_binary_mask in subj_binary_masks.items():
                print(f"{key}: {np.max(subj_binary_mask)}")

        binary_masks[f'subj0{subj_no}'] = subj_binary_masks


    return binary_masks

# Function to create a dictionary containing all the R2 explained variance data of the NSD experiment, could also be turned into a general dict-making func
def nsd_R2_dict(binary_masks = None):
    n_subjects = len(os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata'))
    subject_list = [f'subj{i:02d}' for i in range(1, n_subjects + 1)] 

    nsd_R2_dict = {}

    # Make a loop to go over all the subjects
    for subject in subject_list:
        nsd_R2_dict[subject] = {}
        nsd_R2_dict[subject]['full_R2'] = {}
        nsd_R2_dict[subject]['R2_roi'] = {}
        
        # Create list for all visual rois
        roi_list = list(binary_masks[subject].keys())

        nsd_R2_path = f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/R2.nii.gz'
        nsd_R2_dat, nsd_R2_ar, nsd_R2_dim, nsd_R2_range = get_dat(nsd_R2_path)
        nsd_R2_dict[subject]['full_R2'] = {
                'R2_dat': nsd_R2_dat,
                'R2_ar': nsd_R2_ar,
                'R2_dim': nsd_R2_dim,
                'R2_range': nsd_R2_range
            }
        
        for roi in roi_list:
            nsd_R2_dict[subject]['R2_roi'][roi] = roi_filter(binary_masks[subject][roi], nsd_R2_dict[subject]['full_R2']['R2_ar'])

    return nsd_R2_dict

# Create a dictionary for the top n R2 prf/nsd values, the amount of explained variance
# it does so for every visual roi and subject separately. dataset can be 'nsd' or 'prf'
# and input_dict should be given accordingly.
def rsquare_selection(input_dict = None, top_n = 1000, n_subjects = None, dataset = 'nsd'):
    rsq_dict = {}
    
    if dataset == 'prf':
        rois = input_dict['subj01']['proc'].keys()
    elif dataset == 'nsd':
        rois = input_dict['subj01']['R2_roi'].keys()

    for subj_no in range (1, n_subjects + 1):
        subj_rsq = {}
        for roi in rois:
            if dataset == 'prf':
                rsq_ar = input_dict[f'subj0{subj_no}']['proc'][roi]['R2']
            elif dataset == 'nsd':
                rsq_ar = input_dict[f'subj0{subj_no}']['R2_roi'][roi]
            rsq_sort = np.argsort(rsq_ar[:, -1])
            rsq_top = rsq_ar[rsq_sort[-top_n:]]
            subj_rsq[roi] = rsq_top
        rsq_dict[f'subj0{subj_no}'] = subj_rsq
    return rsq_dict
        

# Function to create the Gaussian image
def make_gaussian_2d(size, center_row, center_col, sigma):
    rows = np.arange(size)
    cols = np.arange(size)
    rows, cols = np.meshgrid(rows, cols)
    exponent = -((rows - center_row)**2 / (2 * sigma**2) + (cols - center_col)**2 / (2 * sigma**2))
    gaussian = np.exp(exponent)
    return gaussian

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

# Function to create a list solely containing roi-based voxels
def roi_filter(roi_mask, input_array):
    roi_ices = np.argwhere(roi_mask != 0)

    # Create list that only contains the voxels of the specific roi
    roi_ar = np.column_stack((roi_ices, input_array[roi_ices[:, 0], roi_ices[:, 1], roi_ices[:, 2]]))

    # Filter away the nan values
    output_roi = roi_ar[~np.isnan(roi_ar).any(axis=1)]

    rounded_output_roi = np.round(roi_ar, 5)
    
    # Set print options to control precision and suppress scientific notation
    np.set_printoptions(precision=5, suppress=True)
    
    return rounded_output_roi

def write_prf_dict(binary_masks):
    n_subjects = len(os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata'))
    subject_list = [f'subj{i:02d}' for i in range(1, n_subjects + 1)] 

    prf_dict = {}

    # Make a loop to go over all the subjects
    for subject in subject_list:
        prf_dict[subject] = {}
        prf_dict[subject]['nsd_dat'] = {}
        # Initialize dictionaries if they don't exist
        prf_dict[subject]['proc'] = {}

        # Create list for all visual rois
        roi_list = list(binary_masks[subject].keys())

        # Get the overall prf results, save them in a dict
        prf_types = ['angle', 'eccentricity', 'exponent', 'gain', 'meanvol', 'R2', 'size']

        for prf_type in prf_types:
            prf_path = f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/prf_{prf_type}.nii.gz'
            prf_dat, prf_ar, prf_dim, prf_range = get_dat(prf_path)
            prf_dict[subject]['nsd_dat'][prf_type] = {
                'prf_dat': prf_dat,
                'prf_ar': prf_ar,
                'prf_dim': prf_dim,
                'prf_range': prf_range
            }
        
        for roi in roi_list:
            prf_dict[subject]['proc'][roi] = {
                prf_type : None for prf_type in prf_types
            } 
            for prf_type in prf_types:
                prf_dict[subject]['proc'][roi][prf_type] = roi_filter(binary_masks[subject][roi], prf_dict[subject]['nsd_dat'][prf_type]['prf_ar'])


           # Calculate the linear pRF sigma values, these tend to be smaller and don't take
            # into account the nonlinear relationship between input and neural respons
            lin_sigmas = prf_dict[subject]['proc'][roi]['size'][:,3] * np.sqrt(prf_dict[subject]['proc'][roi]['exponent'][:,3])

            prf_dict[subject]['proc'][roi]['lin_sigma'] = np.column_stack([prf_dict[subject]['proc'][roi]['size'][:,0:3], lin_sigmas])


    return prf_dict


# Create a plot in which the different calculations (CSS,nonlinear vs. linear) for pRF radius are compared
def compare_radius(prf_dictionary, size_key='size', sigma_key='lin_sigma', x_lim=(0, 20), y_lim=(0, 8), ci = 95):
    """
    Plot pRF data for each ROI side by side.

    Parameters:
    - prf_plots_dict: Dictionary containing pRF data.
    - size_key: Key for accessing pRF size data in the dictionary.
    - sigma_key: Key for accessing linear sigma data in the dictionary.
    - x_lim: Tuple specifying the x-axis limits.
    - y_lim: Tuple specifying the y-axis limits.
    """
    # Set seaborn style to remove background grid
    sns.set_style("white")

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Titles for subplots
    subplot_titles = ['pRF size as radius', 'Linear sigma as radius']

    # Loop over subplots
    for k, ax in enumerate(axes):
        # Set subplot title
        ax.set_title(subplot_titles[k])

        # Define a list of bright colors for each ROI
        bright_palette = sns.color_palette('bright', n_colors=len(prf_dictionary['subj01']['proc']))

        legend_handles = []
        all_subs = {'y': {'V1': np.empty(0), 'V2': np.empty(0), 'V3': np.empty(0), 'V4': np.empty(0)},
                    'x': {'V1': np.empty(0), 'V2': np.empty(0), 'V3': np.empty(0), 'V4': np.empty(0)}}

        for subject, dict_shelve in prf_dictionary.items():
            for j, (roi, data) in enumerate(dict_shelve['proc'].items()):
                ecc_values = data.get('eccentricity', [])

                if k == 0:
                    size_values = data.get(size_key, [])
                    y_data = size_values[:, 3]  # Assuming 4th column of prf_size
                    label_prefix = 'pRF size'
                else:
                    sigma_values = data.get(sigma_key, [])
                    y_data = sigma_values[:, 3]  # Assuming 4th column of linear sigma
                    label_prefix = 'Linear sigma'

                # Filter values within the desired range
                valid_indices = (
                    (0 <= ecc_values[:, 3]) & (ecc_values[:, 3] <= 20) &
                    (0 <= y_data) & (y_data <= 20)
                )

                ecc_values = ecc_values[valid_indices]
                y_data = y_data[valid_indices]

                x_data = ecc_values[:, 3]  # Assuming 4th column of prf_ecc

                all_subs['y'][f'{roi[:2]}'] = np.concatenate([all_subs['y'][f'{roi[:2]}'], y_data])
                all_subs['x'][f'{roi[:2]}'] = np.concatenate([all_subs['x'][f'{roi[:2]}'], x_data])

        for n_col, roi in enumerate(['V1', 'V2', 'V3', 'V4']):
            # Collect legend handles for each ROI
            legend_handles.append(Line2D([0], [0], color=bright_palette[n_col], lw=2, label=f'{roi}'))

            # Plot all subjects' data for the current ROI
            sns.regplot(x=all_subs['x'][f'{roi}'], y=all_subs['y'][f'{roi}'],
                        scatter_kws={'alpha': 0, 's': 1, 'color': bright_palette[n_col]},
                        line_kws={'alpha': 1, 'color': bright_palette[n_col]},
                        truncate=False, ci=ci, ax=ax)

        sns.despine()

        # Set the limits of the x and y axes
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xticks(np.arange(0, x_lim[1]+1, 5))
        ax.set_yticks(range(y_lim[1]+1))

        # Add legend outside the subplot
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel('pRF eccentricity (deg)')
        ax.set_ylabel(f'{label_prefix} (deg)')

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    plt.show()
    return all_subs

# # This class is to make sure that the heatmap can still be plotted if all pRF
# # options have been considered.
class AllPRFConsidered(Exception):
    pass

def get_mask(dim = 200, subject = 'subj01', binary_masks = None, 
             prf_proc_dict = None, type = 'full_gaussian', roi = 'V2', 
             plot = 'y', heatmap = 'n', prf_vec = None, iter = None, excl_reason = 'n', 
             sigma_min = 0, sigma_max = 4.2, ecc_max = 4.2, rand_seed = None, filter_dict = None, 
             ecc_strict = None, grid = 'n'):

    if rand_seed == None:
        random.seed(random.randint(1, 1000000))
    else:
        random.seed(rand_seed)
    
    # Construct the variable name for binary mask using roi argument CHECK IF I USE THIS
    roi_flt = binary_masks[subject][f'{roi}_mask']
    
    # Create objects for all the required pRF data
    roi_mask_data = prf_proc_dict[subject]['proc'][f'{roi}_mask']
    angle_roi, ecc_roi, expt_roi, size_roi, rsq_roi= roi_mask_data['angle'], roi_mask_data['eccentricity'], roi_mask_data['exponent'], roi_mask_data['size'], roi_mask_data['R2']

    # Define a mask to filter away data rows based on the filter_dict, which is supposed to be
    # a dictionary that includes a subset of filtered values for every subject, roi, based on
    # another parameter, such as explained mean variance, R2.
    if filter_dict != None:
        smaller_xyz = filter_dict[subject][f'{roi}_mask'][:, :3]
        mask = np.any(np.all(angle_roi[:, None, :3] == smaller_xyz, axis=-1), axis=1)
    else:
        mask = range(0, angle_roi.shape[0])
    
    # Condition for when the function is used to plot a heatmap, set to 'y', or any other value to do so
    if heatmap == 'n':
        prf_vec = random.sample(range(angle_roi[mask].shape[0]), angle_roi[mask].shape[0])
        iter = 0

    max_prf_vec = max(prf_vec)  # Maximum value of prf_vec

    while True:
        if iter >= max_prf_vec:
            raise AllPRFConsidered("All potential pRFs have been considered")

        n = prf_vec[iter]
        iter += 1

        prf_angle, prf_ecc, prf_expt, prf_size, prf_rsq = angle_roi[mask][n][3], ecc_roi[mask][n][3], expt_roi[mask][n][3], size_roi[mask][n][3], rsq_roi[mask][n][3]
        x_vox, y_vox, z_vox = int(angle_roi[mask][n][0]), int(angle_roi[mask][n][1]), int(angle_roi[mask][n][2])

        sigma = prf_size * np.sqrt(prf_expt)
        sigma_pure = sigma * (dim / 8.4)
        outer_bound = prf_ecc
        
        # Condition to regulate the strictness of maximum eccentricity values
        if ecc_strict == 'y':
            outer_bound = prf_ecc + prf_size
        
        # Sinus is used to calculate height, cosinus width
        # so c_index is the y coordinate and r_index is the x coordinate. 
        # the * (dim / 8.4) is the factor to translate it into raw pixel values
        
        y = ((1 + dim) / 2) - (prf_ecc * np.sin(np.radians(-prf_angle)) * (dim / 8.4)) #y in pix (c_index)
        x = ((1 + dim) / 2) + (prf_ecc * np.cos(np.radians(prf_angle)) * (dim / 8.4)) #x in pix (r_index)

        degrees_per_pixel = 8.4 / dim

        if type == 'circle' or type == 'gaussian':
            deg_radius = sigma
            pix_radius = sigma_pure
        elif type == 'cut_gaussian' or type == 'full_gaussian' or type == 'outline':
            deg_radius = prf_size
            pix_radius = prf_size * (dim / 8.4)
            
        valid_conditions = (
            0 < x < dim,
            0 < y < dim,
            sigma_min < deg_radius,
            deg_radius < sigma_max,
            outer_bound < ecc_max,
            # prf_expt > 0
        )

        if all(valid_conditions):
            break

        # Check for argument option to print reason for excluding voxels
        elif excl_reason == 'y':
            print(f"Discarding pRF mask for voxel [{x_vox}, {y_vox}, {z_vox}] due to:")
            if not valid_conditions[0]:
                print("   - x out of bounds")
            if not valid_conditions[1]:
                print("   - y out of bounds")
            if not valid_conditions[2]:
                print("   - sigma_pure too small")
            if not valid_conditions[3]:
                print("   - sigma_pure too large")
            if not valid_conditions[4]:
                print(f"   -  pRF outside of center {2 * ecc_max}° visual degrees")
            # if not valid_conditions[4]:
            #     print("   - expt_ar value too small")

    # Note: all the masks are made using pixel values for x, y, and sigma
    # Check whether the same is done later on, in the heatmaps and get_img_prf.
    if type == 'gaussian':
        prf_mask = make_gaussian_2d(dim, x, y, sigma_pure)
    elif type == 'circle':
        prf_mask = make_circle_mask(dim, x, y, sigma_pure)
    elif type == 'full_gaussian':
        prf_mask = make_gaussian_2d(dim, x, y, prf_size * (dim / 8.4))
    elif type == 'cut_gaussian':
        prf_mask = css_gaussian_cut(dim, x, y, prf_size * (dim / 8.4))
    elif type == 'outline':
        prf_mask = (make_circle_mask(dim, x, y, prf_size * (dim / 8.4), fill = 'n'))
    else:
        raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian', and 'outline'.")
    
    # Convert pixel indices to degrees of visual angle
    x_deg = (x - (dim / 2)) * degrees_per_pixel
    y_deg = ((dim / 2) - y) * degrees_per_pixel
    
    if plot == 'y':
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(prf_mask, cmap='bone', origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
        ax.set_title(f'Region Of Interest: {roi}\n'
                    f'Voxel: [{x_vox}, {y_vox}, {z_vox}]\n'
                    f'pRF x,y,σ: {round(x_deg, 1), round(y_deg, 1), round(deg_radius, 1)}\n'
                    f'Angle: {round(prf_angle, 2)}°\nEccentricity: {round(prf_ecc, 2)}°\n'
                    f'Exponent: {round(prf_expt, 2)}\nSize: {round(prf_size, 2)}°\n'
                    f'Explained pRF variance (R2): {round(prf_rsq, 2)}%')
        ax.set_xlabel('Horizontal Degrees of Visual Angle')
        ax.set_ylabel('Vertical Degrees of Visual Angle')

        # Set ticks at every 0.1 step
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

            # Create a dictionary to store the values
    prf_output_dict = {
        'mask': prf_mask,
        'x': x,
        'y': y,
        'pix_radius': pix_radius,
        'deg_radius': deg_radius, 
        'iterations': iter,
        'x_vox': x_vox,
        'y_vox': y_vox,
        'z_vox': z_vox,
        'x_deg': x_deg,
        'y_deg': y_deg,
        'angle': prf_angle,
        'eccentricity': prf_ecc,
        'exponent': prf_expt,
        'size': prf_size,
        'R2': prf_rsq
    }

        # Return the dictionary
    return prf_output_dict

# Function to compare the different ways of reaching a pRF filter. Nonlinear (CSS) and linear
def compare_masks(mask_dict = None, prf_dict = None, subject='subj01', roi='V1', sigma_min=0.1, 
                  sigma_max=4.2, cmap = 'afmhot'):
  
    def plot_mask(ax, mask, title):
        ax.imshow(mask, cmap = cmap)
        ax.set_title(title)
        ax.axis('off')

    # Assuming get_mask returns the mask you want to plot
    dobbel = random.randint(1, 1000)

    circle_dict = get_mask(dim=425, subject=subject, binary_masks = mask_dict, 
                                       prf_proc_dict=prf_dict, type='circle', roi=roi,
                                       plot='n', excl_reason='n', sigma_min=sigma_min, sigma_max=sigma_max, rand_seed=dobbel)

    gaus = make_gaussian_2d(425, circle_dict['x'], circle_dict['y'], circle_dict['pix_radius'])
    full_gaus = make_gaussian_2d(425, circle_dict['x'], circle_dict['y'], (circle_dict['size'] * (425 / 8.4)))
    cut_gaus = css_gaussian_cut(425, circle_dict['x'], circle_dict['y'], (circle_dict['size'] * (425 / 8.4)))
    
    # Create a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Plot each mask on a separate axis
    plot_mask(axs[0], circle_dict['mask'], 'Boolean mask')
    plot_mask(axs[1], gaus, 'Simple Gaussian')
    plot_mask(axs[2], full_gaus, 'Full CSS input responsive Gaussian \n with prf_size as sigma')
    plot_mask(axs[3], cut_gaus, '1 SD cut CSS Gaussian')

    # Add a main title for the entire figure
    fig.suptitle(f'{roi} of {subject}', fontsize=14)

    plt.show()


# Function to compare the heatmaps on different bases, need to add different comparison types, than just roi. For now suffices
def compare_heatmaps(n_prfs, binary_masks=None, prf_proc_dict=None, filter_dict=None, basis='roi',
                     mask_type='cut_gaussian', cmap='CMRmap', roi='V1', excl_reason='n', sigma_min=0,
                     sigma_max=4.2, ecc_max=2, print_prog='n', outline_degs=None, fill_outline='n', ecc_strict=None, grid = 'n'):
    if basis == 'roi':
        rois = sorted(prf_proc_dict['subj01']['proc'].keys())

    def plot_mask(ax, mask, title, last=None):
        img = ax.imshow(mask, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        ax.set_xlabel('Horizontal Degrees of Visual Angle')
        ax.set_ylabel('Vertical Degrees of Visual Angle')
        # if last == 'y':
        #     cbar = plt.colorbar(img, ax=ax, pad=0.01)  # Ensure colorbar placement
        #     cbar.set_label('pRF density')

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for n, roi in enumerate(rois):
        heatmap, iter, end_premat, roi, prf_sizes, rel_surf, total_prfs_found = prf_heatmap(
                                    n_prfs, binary_masks=binary_masks, prf_proc_dict=prf_proc_dict,
                                    mask_type=mask_type, cmap=cmap, roi=roi[:2], excl_reason=excl_reason,
                                    sigma_min=sigma_min, sigma_max=sigma_max, ecc_max=ecc_max,
                                    print_prog=print_prog, subjects='all', outline_degs=outline_degs,
                                    filter_dict=filter_dict, fill_outline=fill_outline, plot_heat='n',
                                    ecc_strict=ecc_strict, grid = grid)

        last_plot = 'y' if n == (len(rois) - 1) else 'n'
        plot_mask(axs[n], heatmap, f'{roi}\n Average pRF radius: {round(np.mean(prf_sizes), 2)}°, {rel_surf}% of outline surface\n'
                  f'Total amount of pRFs found: {total_prfs_found}', last=last_plot)

    plt.tight_layout()
    plt.show()

# class AllPRFConsidered(Exception):
#     pass

def prf_heatmap(n_prfs, binary_masks, prf_proc_dict, dim=425, mask_type='gaussian', cmap='gist_heat', 
                roi='V2', sigma_min=1, sigma_max=25, ecc_max = 4.2, print_prog='n', excl_reason = 'n', subjects='all',
                outline_degs = None, filter_dict = None, fill_outline = 'n', plot_heat = 'y', ecc_strict = None, grid = 'n'):
    
    outline_surface = np.pi * outline_degs**2
    prf_sumstack = []
    prf_sizes = []
    total_prfs_found = 0
    if subjects == 'all':
        subjects = list(binary_masks)
    else:
        subjects = [subjects]
        
    # To make sure that the maximum amount of pRFs that is searched through is adapted to the individual
    for subject in subjects:
        # This is to make sure that the random sampling is done correctly, for different restrictions on the amount of
        # pRFs to sample from. This can be restricted through exclusion criteria, or for example the filter_dict.
        if filter_dict != None:
            smaller_xyz = filter_dict[subject][f'{roi}_mask'][:, :3]
            # filter = np.any(np.all(binary_masks[subject][f'{roi}_mask'][:, None, :3] == smaller_xyz, axis=-1), axis=1)
            filter = np.any(np.all(prf_proc_dict[subject]['proc'][f'{roi}_mask']['angle'][:, None, :3] == smaller_xyz, axis=-1), axis=1)
            roi_flt = filter_dict[subject][f'{roi}_mask'].shape[0] # Amount of voxels in top rsq dict for subj, roi
            prf_vec = random.sample(range(roi_flt), roi_flt) # Create random vector to shuffle order voxels to consider
            
        else:
            filter = range(0, prf_proc_dict[subject]['proc'][f'{roi}_mask']['angle'].shape[0])
            roi_flt = binary_masks[subject][f'{roi}_mask'] # This is the total number of voxels for subj, roi
            prf_vec = random.sample(range(np.sum(roi_flt)), np.sum(roi_flt)) # Idem dito as in the 'if' part
            
        # FIX THIS STILL!!!
        if n_prfs == 'all':
            n_prfs_subject = np.sum(binary_masks[subject][f'{roi}_mask']) # This does not work
            # n_prfs_subject = random.randint(10,20)
        else:
            n_prfs_subject = n_prfs

        # Create an empty array to fill with the masks
        prf_single = np.zeros([dim, dim, n_prfs_subject])

        iter = 0
        end_premat = False
        for prf in range(n_prfs_subject):
            try:
                # prf_single[:, :, prf], _, _, _, new_iter = get_mask(dim=dim,
                prf_dict = get_mask(dim=dim,
                                    subject=subject,
                                    binary_masks=binary_masks,
                                    prf_proc_dict=prf_proc_dict,
                                    type=mask_type,
                                    roi=roi,
                                    plot='n',
                                    heatmap='y',
                                    prf_vec=prf_vec,
                                    iter=iter,
                                    sigma_min=sigma_min,
                                    sigma_max=sigma_max,
                                    ecc_max = ecc_max,
                                    excl_reason=excl_reason,
                                    filter_dict = filter_dict,
                                    ecc_strict = ecc_strict,
                                    grid = grid)
                prf_single[:, :, prf] = prf_dict['mask']
                iter = prf_dict['iterations']
                prf_size = prf_dict['size']
                prf_sizes.append(prf_size)
                if print_prog == 'y':
                    print(f"Subject: {subject}, Voxel {prf+1} out of {n_prfs_subject} found")
                    if (prf+1) == n_prfs_subject:
                        print('\n')
            except AllPRFConsidered:
                if prf >= n_prfs_subject:
                    print(f'All potential pRFs have been considered at least once.\n'
                        f'Total amount of pRFs found: {len(prf_sizes)}')
                    end_premat = True
                    
                break  # Exit the loop immediately
        
        prf_sumstack.append(np.mean(prf_single, axis=2))
        total_prfs_found += len(prf_sizes)
         
    avg_prf_surface = np.pi * np.mean(prf_sizes)**2
    relative_surface = round(((avg_prf_surface / outline_surface) * 100), 2)
    # Combine heatmaps of all subjects
    prf_sum_all_subjects = np.mean(np.array(prf_sumstack), axis=0)
    outline = make_circle_mask(425, 213, 213, outline_degs * 425/8.4, fill=fill_outline)
    # Create a circle outline if an array is provide in the outline argument (should be same dimensions, binary)
    prf_sum_all_subjects += (np.max(prf_sum_all_subjects) * outline) if outline_degs is not None else 1

    # Display the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(prf_sum_all_subjects, cmap=cmap, origin='lower', extent=[-4.2, 4.2, -4.2, 4.2])
    ax.set_title(f'Region Of Interest: {roi}\n'
                 f'Spatial restriction of central {2 * ecc_max}° visual angle\n'
                 f'Average pRF radius: {round(np.mean(prf_sizes), 2)}°, {relative_surface}% of outline surface\n'
                 f'Total amount of pRFs found: {total_prfs_found}')
    ax.set_xlabel('Horizontal Degrees of Visual Angle')
    ax.set_ylabel('Vertical Degrees of Visual Angle')
    cbar = plt.colorbar(im, ax=ax, shrink = .6)
    cbar.set_label('pRF density')  
    
    # Set ticks at every 0.1 step
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    if plot_heat == 'n':
        plt.close()
    else: 
        plt.show()

    return prf_sum_all_subjects, iter, end_premat, roi, prf_sizes, relative_surface, total_prfs_found


# This function applies a gaussian filter to the loaded image
def get_img_prf(image, x = None, y = None, sigma = None, type = 'gaussian', heatmask = None, 
                binary_masks = None, prf_proc_dict = None, roi = 'V1', sigma_min=1, sigma_max=25, 
                rand_seed = None, invert = 'n', central = 'n', filter_dict = None, grid = 'n'):
    # arguments can be specified manually, or generated randomly if none are given
    # when entered manually there is no specification of parameters (yet)
    # I have to think about whether this is actually relevant, don't think so.
    # It's nothing more than a check whether it works, which it does.

    masked_arr = np.zeros(image.shape) # Create empty array for masked image
    if type == 'heatmask':
        # prf_mask = np.mean(heatmask, axis=2)
        prf_mask = heatmask
    else:
        if x is None and y is None and sigma is None:
            prf_info = get_mask(dim = image.shape[0], subject = 'subj01', plot = 'n', 
                                binary_masks = binary_masks, prf_proc_dict = prf_proc_dict, 
                                type = type, sigma_min=sigma_min, sigma_max=sigma_max, 
                                rand_seed = rand_seed, filter_dict = filter_dict, grid = grid)

        
        x, y, sigma = prf_info['x'], prf_info['y'], prf_info['pix_radius']
        masked_arr = np.zeros(image.shape) # Create empty array for masked image

        pix_radius = sigma #* (image.shape[0]/8.4)
        
        if central == 'y':
            x = y = ((image.shape[0] + 1) / 2)
            pix_radius = (2 * (image.shape[0] / 8.4))

            
        if type == 'gaussian':
            prf_mask = make_gaussian_2d(image.shape[0], x, y, pix_radius)
        elif type == 'circle':
            prf_mask = make_circle_mask(image.shape[0], x, y, pix_radius)
        elif type == 'full_gaussian':
            prf_mask = make_gaussian_2d(image.shape[0], x, y, pix_radius)
        elif type == 'cut_gaussian':
            prf_mask = css_gaussian_cut(image.shape[0], x, y, pix_radius)
        elif type == 'outline':
            prf_mask = (make_circle_mask(image.shape[0], x, y, pix_radius, fill = 'n'))
        else:
            raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian', and 'outline'.")

    # Apply the mask per layer of the input image using matrix multiplication
    for colour in range(image.shape[2]):
        if invert == 'n':
            # masked_arr[:,:,colour] = image[:,:,colour] * np.flipud(prf_mask)
            masked_arr[:,:,colour] = image[:,:,colour] * np.flipud(prf_mask)
        elif invert == 'y':
            masked_arr[:,:,colour] = image[:,:,colour] * np.flipud(1 - prf_mask)
            

    # Normalize the masked image according to the RGB range of 0-255
    masked_img = masked_arr / 255

    # ax.imshow(masked_img, origin='upper', extent=[0,image.shape[0],0,image.shape[0]])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(masked_img, origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
    ax.set_title(f'Region Of Interest: {roi}\n')
    ax.set_xlabel('Horizontal Degrees of Visual Angle')
    ax.set_ylabel('Vertical Degrees of Visual Angle')




    # Set ticks at every 0.1 step
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    
    return prf_info

