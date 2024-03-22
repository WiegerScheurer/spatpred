import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import os
import pickle
from sklearn.linear_model import LinearRegression
from funcs.utility import numpy2coords, coords2numpy, filter_array_by_size, find_common_rows, get_zscore

# Function to create a dictionary containing all the relevant HRF signal info for the relevant voxels.
def get_hrf_dict(subjects, voxels, prf_region = 'center_strict', min_size = .1, max_size = 1, prf_proc_dict = None, vox_n_cutoff = None, plot_sizes = 'n'):
    
    hrf_dict = {}
    voxdict_select = {}
    
    for subject in [subjects]:
        hrf_dict[subject] = {}
        voxdict_select[subject] = {}

        # Get a list of files in the directory
        files = os.listdir(f'/home/rfpred/data/custom_files/{subject}/{prf_region}/')

        # Filter files that start with "beta_dict" and end with ".pkl"
        filtered_files = [file for file in files if file.startswith("beta_dict") and file.endswith(".pkl")]

        # Sort files based on the first number after 'beta_dict'
        sorted_files = sorted(filtered_files, key=lambda x: int(''.join(filter(str.isdigit, x.split('beta_dict')[1]))))

        # Print the sorted file names
        for n_file, file_name in enumerate(sorted_files):
            print(file_name)
                
            # Load in the boolean mask for inner circle voxel selection per roi.
            with open(f'/home/rfpred/data/custom_files/{subject}/{prf_region}/{file_name}', 'rb') as fp:
                beta_session = pickle.load(fp)
            
            rois = list(beta_session[subject].keys())

            if n_file == 0:
                hrf_dict[subject] = copy.deepcopy(beta_session[subject])
            for i, roi in enumerate(rois):

                
                voxel_mask = voxels[subject][roi] # These is the boolean mask for the specific subject, roi
                if vox_n_cutoff == None:
                    vox_n_cutoff = numpy2coords(voxel_mask).shape[0]
                if min_size != None and max_size != None:
                    preselect_voxels = numpy2coords(voxel_mask)[:vox_n_cutoff,:] # This cutoff is to allow for checking whether the amount of voxels per category matters (peripher/central)
                    size_selected_voxels = filter_array_by_size(prf_proc_dict[subject]['proc'][roi]['size'], min_size, max_size)

                    joint_voxels = find_common_rows(preselect_voxels, size_selected_voxels)
                    
                    voxel_mask = coords2numpy(joint_voxels, voxels['subj01']['V1_mask'].shape) * 1
                    
                    # Acquire the specific RF sizes for inspection, plots.
                    vox_slct = joint_voxels.reshape(-1, 1, joint_voxels.shape[1])
                    sizes_reshape = size_selected_voxels[:, :3].reshape(1, -1, size_selected_voxels.shape[1]-1)
                    equal_rows = np.all(vox_slct == sizes_reshape, axis = 2)
                    matching_rows = np.any(equal_rows, axis=0)
                    size_slct = size_selected_voxels[matching_rows]
                    
                voxdict_select[subject][roi] = voxel_mask
                n_voxels = numpy2coords(voxel_mask).shape[0]
                print(f'\tAmount of voxels: {n_voxels}')

                vox_indices = np.zeros([n_voxels, 3], dtype = int) # Initiate an empty array to store vox indices
                hrf_dict[subject][roi]['roi_sizes'] = size_slct
                for coordinate in range(vox_indices.shape[1]): # Fill the array with the voxel coordinates as indices
                    vox_indices[:, coordinate] = np.where(voxel_mask == 1)[coordinate]
                    
                # for voxel in range(len(beta_session[subject][roi])):
                for voxel in range(n_voxels):
                    hrf_betas_ses = copy.deepcopy(beta_session[subject][roi][f'voxel{voxel + 1}'])
                    
                    if n_file == 0:
                        total_betas = hrf_betas_ses
                        hrf_dict[subject][roi][f'voxel{voxel+1}'] = {
                            'xyz': list(vox_indices[voxel]),
                            'size': size_slct[voxel][3],
                            'hrf_betas': total_betas,
                            'hrf_betas_z': 0,
                            'hrf_rsquared': 0,
                            'hrf_rsquared_z': 0
                        }
                             
                    else: 
                        old_betas = hrf_dict[subject][roi][f'voxel{voxel + 1}']['hrf_betas']
                        hrf_dict[subject][roi][f'voxel{voxel + 1}']['hrf_betas']
                        total_betas = np.append(old_betas, hrf_betas_ses)   
                             
                    hrf_dict[subject][roi][f'voxel{voxel+1}'] = {
                        'xyz': list(vox_indices[voxel]),
                        'size': size_slct[voxel][3],
                        'hrf_betas': total_betas,
                        'hrf_betas_z': 0,
                        'hrf_rsquared': 0,
                        'hrf_rsquared_z': 0
                    }
            n_betas = len(hrf_dict[subject][roi][f'voxel{voxel+1}']['hrf_betas'])
            print(f'\tProcessed images: {n_betas}')

    if plot_sizes == 'y':
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Create a figure with 2x2 subplots
        axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
        cmap = plt.get_cmap('ocean')  # Get the 'viridis' color map
        for i, roi in enumerate(rois):
            sizes = hrf_dict[subject][roi]['roi_sizes'][:, 3]
            color = cmap(i / len(rois))  # Get a color from the color map
            sns.histplot(sizes, kde=True, ax=axs[i], color=color)  # Plot on the i-th subplot
            axs[i].set_title(f'RF sizes for {roi[:2]} (n={sizes.shape[0]})')  # Include the number of voxels in the title
            axs[i].set_xlim([min_size-.1, max_size+.1])  # Set the x-axis limit from 0 to 2
        fig.suptitle(f'{prf_region}', fontsize=18)
        plt.tight_layout()
        plt.show()
                
    with open(f'./data/custom_files/{subjects}hrf_dict.pkl', 'wb') as fp:
        pickle.dump(hrf_dict, fp)
    
            
    return hrf_dict, voxdict_select, joint_voxels, size_selected_voxels



def multivariate_regression(X, y_matrix, z_scorey:bool = False):
    # Reshape X to (n_imgs, 1) if it's not already
    if X.ndim == 1:
        X = X.reshape(-1, 1)


    if z_scorey:
        y_matrix = get_zscore(y_matrix, print_ars = 'n')

    # Fit the multivariate regression model
    model = LinearRegression().fit(X, y_matrix)

    # Extract beta coefficients and intercepts
    beta_values = model.coef_
    intercept_values = model.intercept_

    # Calculate R-squared values
    rsquared_values = model.score(X, y_matrix)

    return beta_values, intercept_values, rsquared_values, model

def regression_dict_multivariate(subject, feat_type, voxels, hrfs, feat_vals, n_imgs='all', z_scorey:bool = False):
    reg_dict = {}
    
    # Set the amount of images to regress over in case all images are available.
    if n_imgs == 'all':
        n_imgs = len(feat_vals)
    
    X = np.array(feat_vals[feat_type][:n_imgs]).reshape(n_imgs, 1)  # Set the input matrix for the regression analysis
  
    # This function will run the multiple regression analysis for each voxel, roi, image, for a subject.
    rois = list(voxels[subject].keys())

    for roi in rois:
        reg_dict[roi] = {}
        voxel_mask = voxels[subject][roi]  # These are the boolean mask for the specific subject, roi
        n_voxels = np.sum(voxel_mask).astype('int')  # This is the number of voxels in this roi
        vox_indices = np.column_stack(np.where(voxel_mask == 1))  # Get voxel indices for the current ROI
        
        # Extract y_matrix for all voxels within the ROI
            
        if z_scorey:
            y_matrix = get_zscore(y_matrix, print_ars = 'n')
            
        y_matrix = np.array([hrfs[subject][roi][f'voxel{voxel + 1}']['hrf_betas'] for voxel, xyz in enumerate(vox_indices)]).T #/ 300

        # Perform multivariate regression
        beta_values, intercept_values, rsquared_value, reg_model = multivariate_regression(X, y_matrix, z_scorey = z_scorey)
        
        reg_dict[roi]['voxels'] = {}
        
        for voxel, vox_idx in enumerate(vox_indices):
            reg_dict[roi]['voxels'][f'vox{voxel}'] = {
                'xyz': list(vox_idx),
                'beta': beta_values[voxel],
                'icept': intercept_values[voxel]
            }
            
        reg_dict[roi]['y_matrix'] = y_matrix
        reg_dict[roi]['all_reg_betas'] = beta_values
        reg_dict[roi]['all_intercepts'] = intercept_values
        reg_dict[roi]['rsquared'] = rsquared_value
        


    return reg_dict, X

import seaborn as sns
def plot_roi_beta_distribution(reg_dict, z_score = None, icept_correct = None, feat_type = ''):
    
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    colors = sns.color_palette('ocean', n_colors=len(reg_dict))
    num_bins = 30  # Specify the number of bins
    icept_values = []
    for i, (roi, voxels) in enumerate(reg_dict.items()):
        
        beta_values = np.concatenate([voxel_data['beta'] for voxel_data in voxels['voxels'].values()])
        plot_vals = beta_values
        icept_values = np.concatenate([np.array([voxel_data['icept']]) for voxel_data in voxels['voxels'].values()])        

        if icept_correct == 'y':
            plot_vals = beta_values / get_zscore(icept_values, print_ars = 'n')

        sns.histplot(plot_vals, kde=True, ax=axes[i], color=colors[i], label=f'{roi} ROI', bins=num_bins)  # Specify the bins

        axes[i].set_title(f'Distribution of Beta Values for {roi[:2]}\n'
                        f'n_voxels={len(beta_values)}')
        axes[i].set_ylabel('Occurrence freq', weight = 'normal', fontsize = 12)
        axes[i].set_xlim(-.05, .15)  # Set the same x range for all subplots

        # axes[i].set_xticks(np.arange(-1, 15, .5))  # Set the ticks to be more frequent
        axes[i].set_xticks(np.arange(-.05, .15, .05))  # Set the ticks to be more frequent

    axes[-1].set_xlabel('Beta values', weight = 'normal', fontsize = 12)
    fig.suptitle(f'Multivariate regression approach (Subject 1, all images, HRF beta) {feat_type}', fontsize=16, y=1)

    plt.tight_layout()
    plt.show()
    
