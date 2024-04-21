#!/usr/bin/env python3

import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from matplotlib.ticker import MultipleLocator
import nibabel as nib
import pickle
from importlib import reload
import h5py
from nilearn import plotting
import nibabel as nib
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

print(sys.path)

from funcs.imgproc import (get_imgs_designmx, rms_all, feature_df, show_stim, get_rms_contrast, 
                           get_rms_contrast_lab, get_contrast_df, get_img_prf, get_visfeature_dict,
                           get_betas)
from funcs.rf_tools import (get_mask, css_gaussian_cut, make_circle_mask, make_visrois_dict, 
                            write_prf_dict, nsd_R2_dict, rsquare_selection, make_gaussian_2d, 
                            find_top_vox, plot_top_vox, find_roi, rsq_to_size, get_good_voxel)
from funcs.utility import (print_dict_structure, print_large, get_zscore, mean_center, hypotheses_plot, 
                           multiple_regression)
from funcs.analyses import (multivariate_regression, regression_dict_multivariate, plot_roi_beta_distribution, 
                            get_hrf_dict, plot_beta_to_icept, reg_plots, univariate_regression, multivariate_regression)
from funcs.utility import(numpy2coords, coords2numpy, filter_array_by_size, find_common_rows, 
                          cap_values, _sort_by_column, _get_voxname_for_xyz)

from notebooks.alien_nbs.lgnpy.lgnpy.CEandSC import lgn_statistics


n_subjects = len(os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata'))
vismask_dict = make_visrois_dict(vox_count = 'y', bin_check = 'n', n_subjects=n_subjects)
prf_dict = write_prf_dict(binary_masks = vismask_dict)

# Load in the original RMS dict (where RMS is calculated before cropping)
with open('./data/custom_files/all_visfeats_rms.pkl', 'rb') as fp:
   visfeats_rms_full_img = pickle.load(fp)
   
# Load in the new RMS dict (where RMS is calculated after cropping, thus cropping prior to RMS)
with open('./data/custom_files/all_visfeats_rms_crop_prior.pkl', 'rb') as fp:
   visfeats_rms_crop_prior = pickle.load(fp)
   
# Load in the Spatial Coherence and Contrast Energy dictionary (this is still where for both the pooling range was 1 degree visual angle)
with open('/home/rfpred/data/custom_files/all_visfeats_scce.pkl', 'rb') as fp:
    visfeats_scce = pickle.load(fp)
    
# Load in the Spatial Coherence and Contrast Energy dictionary (here the pooling range was 2.8 degrees of visual angle, so only look at SC)
with open('/home/rfpred/data/custom_files/all_visfeats_scce_large.pkl', 'rb') as fp:
    visfeats_scce_large = pickle.load(fp)
    
# Load in the .h5 file containing the predictability estimates for subject 01
with h5py.File('/home/rfpred/data/custom_files/subj01/pred/all_predestims.h5', 'r') as hf:
    data = hf.keys()
    predfeats = {key: np.array(hf[key]).flatten() for key in data}
    
# Load in the saved masks for selected voxels that have their prf inside the inner patch.
with open('./data/custom_files/subj01/prf_mask_center_strict.pkl', 'rb') as fp:
    prf_mask_center_strict = pickle.load(fp)
    
# Load in the saved masks for voxels with pRF inside central 1.25 degrees patch
with open('./data/custom_files/subj01/prf_mask_central_strict_l.pkl', 'rb') as fp:
   prf_mask_center_strict_l = pickle.load(fp)    

# Load in the saved masks for voxels with pRF loosely (at least 50%) inside central 1 degrees patch
with open('./data/custom_files/subj01/prf_mask_central_halfloose.pkl', 'rb') as fp:
   prf_mask_central_halfloose = pickle.load(fp)    

# Load in the saved masks for voxels with pRFs that have their center inside the central 1 degree patch
with open('./data/custom_files/subj01/prf_mask_central_loose.pkl', 'rb') as fp:
    prf_mask_central_loose = pickle.load(fp) 

# Load in the saved masks that have their pRFs exclusively outside the central 1 degree patch
with open('./data/custom_files/subj01/prf_mask_periphery_strict.pkl', 'rb') as fp:
   prf_mask_periphery_strict = pickle.load(fp)
  
# # Get subject-specific T1 anatomical maps to use as base for later overlays
# anat_maps = {}
# for subject in prf_dict.keys():
#     anat_maps[subject] = nib.load(f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/T1_to_func1mm.nii.gz')
    
# R2_dict_hrf = nsd_R2_dict(vismask_dict, glm_type = 'hrf')
# R2_dict_onoff = nsd_R2_dict(vismask_dict, glm_type = 'onoff')    
    
# prf_rsq_dict = rsquare_selection(prf_dict, 1000, n_subjects = n_subjects, dataset = 'prf')
# prf_rsq_dict_mapped = coords2numpy(prf_rsq_dict['subj01']['V1_mask'], shape = vismask_dict['subj01']['V1_mask'].shape, keep_vals = True)
# nsd_rsq_dict_hrf = rsquare_selection(R2_dict_hrf, 1000, n_subjects = n_subjects, dataset = 'nsd')
# nsd_rsq_dict_onoff = rsquare_selection(R2_dict_onoff, 1000, n_subjects = n_subjects, dataset = 'nsd')

    
    
# Get the betas
hrf_dict_tight, xyz_to_name  = get_hrf_dict('subj01', voxels = prf_mask_center_strict, prf_region = 'center_strict', 
                                             min_size = .2, max_size = 1, prf_proc_dict = prf_dict, max_voxels = 100 ,plot_sizes = 'y',
                                             vismask_dict = vismask_dict, minimumR2 =50)


# Define the X matrices

# Get some visual feat values for these specific trials
Xrms = visfeats_rms_crop_prior['subj01']['rms']['rms_z']
Xce = visfeats_scce['subj01']['scce']['ce_z']
Xce_l = visfeats_scce_large['subj01']['scce']['ce_z']
Xsc = visfeats_scce['subj01']['scce']['sc_z']
Xsc_l = visfeats_scce_large['subj01']['scce']['sc_z']
Xvisfeats = np.vstack((Xrms, Xce, Xsc, Xsc_l))
# Define the feature names
feature_names = [
    'content_loss_0_MSE', 'content_loss_1_MSE', 'content_loss_2_MSE', 'content_loss_3_MSE', 'content_loss_4_MSE',
    'content_loss_0_L1', 'content_loss_1_L1', 'content_loss_2_L1', 'content_loss_3_L1', 'content_loss_4_L1',
    'pixel_loss_MSE', 'pixel_loss_L1', 'ssim'
]

# Initialize a dictionary to store the z scores
Xvisfeats_dict = {}

# Calculate the z score for each feature
for feature in feature_names:
    Xvisfeats_dict[feature] = get_zscore(predfeats[feature], print_ars = 'n')

# Convert the dictionary to a numpy array
Xpred = np.array(list(Xvisfeats_dict.values()))

# for i in range(Xpred.shape[0]):
#     plt.plot(get_zscore(predfeats[f'content_loss_{i}_{loss}'][:n_trials], print_ars = 'n'), label=f'content_loss_{i}_MSE')
# 
# plt.legend()  # Display the legend
# plt.show()  # Display the plot    
# Initialize a dictionary to store the R^2 values
r2_values = {}
for this_roi in hrf_dict_tight['subj01'].keys():
    r2_values[this_roi] = {}
    print(f'Now running regressions for {this_roi[:2]}')
    for this_voxel in range(hrf_dict_tight['subj01'][this_roi]['R2_vals'].shape[0]):
        print(f'Now running regressions for {this_voxel}')

        vox_xyz, voxname = get_good_voxel(subject='subj01', roi=this_roi[:2], hrf_dict=hrf_dict_tight, xyz_to_voxname=xyz_to_name, 
                                        pick_manually=this_voxel, plot=True, prf_dict=prf_dict, vismask_dict=vismask_dict,selection_basis='R2')
        # good ones for V1: 2,3,6 (0.14)

        y = hrf_dict_tight['subj01'][f'{this_roi[:2]}_mask'][voxname]['hrf_betas_z']


        # Assuming Xpred and y are already defined in your workspace

        # Reshape X and y to 2D arrays because sklearn expects 2D inputs
        # X = Xpred.T
        X = Xvisfeats.T

        # Create a PLSRegression and LinearRegression objects
        PLSmodel = PLSRegression(n_components=X.shape[1])
        OLSmodel = LinearRegression()

        # Fit the models to the data
        PLSmodel.fit(X, y)
        OLSmodel.fit(X, y)

        # Use the models to predict y values
        y_hat_PLS = PLSmodel.predict(X)
        y_hat_OLS = OLSmodel.predict(X)

        # Perform 5-fold cross-validation and get predicted y values
        kf = KFold(n_splits=2)
        y_hat_cv_PLS = cross_val_predict(PLSmodel, X, y, cv=kf)
        y_hat_cv_OLS = cross_val_predict(OLSmodel, X, y, cv=kf)

        # Calculate and print the R-squared values for PLS and OLS models
        r2_PLS = r2_score(y, y_hat_PLS)
        r2_OLS = r2_score(y, y_hat_OLS)
        print(f"PLS R-squared: {r2_PLS:.3f}")
        print(f"OLS R-squared: {r2_OLS:.3f}")

        # Calculate the R-squared values for cross-validated PLS and OLS models
        r2_cv_PLS = r2_score(y, y_hat_cv_PLS)
        r2_cv_OLS = r2_score(y, y_hat_cv_OLS)
        print(f"Cross-validated PLS R-squared: {r2_cv_PLS:.3f}")
        print(f"Cross-validated OLS R-squared: {r2_cv_OLS:.3f}\n")
        
        # Store the R^2 values in the dictionary
        r2_values[this_roi][this_voxel] = {
            'voxelname': voxname,
            'voxel_xyz': vox_xyz,
            'PLS': r2_PLS,
            'OLS': r2_OLS,
            'Cross-validated PLS': r2_cv_PLS,
            'Cross-validated OLS': r2_cv_OLS
        }

# Save the R^2 values to a file
with open('/home/rfpred/data/custom_files/subj01/center_strict/r2_values_visfeats.pkl', 'wb') as f:
    pickle.dump(r2_values, f)
    
print(f'Regressions completed and R^2 values saved to /home/rfpred/data/custom_files/subj01/center_strict/r2_values_visfeats.pkl')





