import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import os
import pickle
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from funcs.utility import numpy2coords, coords2numpy, filter_array_by_size, find_common_rows, get_zscore, mean_center

# Function to create a dictionary containing all the relevant HRF signal info for the relevant voxels.
def get_hrf_dict(subjects, voxels, prf_region = 'center_strict', min_size = .1, max_size = 1, 
                 prf_proc_dict = None, vox_n_cutoff = None, plot_sizes = 'n'):
    
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
                    preselect_voxels = numpy2coords(voxel_mask)
                    size_selected_voxels = filter_array_by_size(prf_proc_dict[subject]['proc'][roi]['size'], min_size, max_size)

                    joint_voxels = find_common_rows(preselect_voxels, size_selected_voxels)[:vox_n_cutoff,:] # This cutoff is to allow for checking whether the amount of voxels per category matters (peripher/central)
                    
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
            
    plt.style.use('default')

    if plot_sizes == 'y':
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Create a figure with 2x2 subplots
        axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
        cmap = plt.get_cmap('ocean')  # Get the 'viridis' color map
        for i, roi in enumerate(rois):
            sizes = hrf_dict[subject][roi]['roi_sizes'][:, 3]
            color = cmap(i / len(rois))  # Get a color from the color map
            sns.histplot(sizes, kde=True, ax=axs[i], color=color, bins = 100)  # Plot on the i-th subplot
            axs[i].set_title(f'RF sizes for {roi[:2]} (n={sizes.shape[0]})')  # Include the number of voxels in the title
            axs[i].set_xlim([min_size-.1, max_size+.1])  # Set the x-axis limit from 0 to 2
        fig.suptitle(f'{prf_region}', fontsize=18)
        plt.tight_layout()
        plt.show()
                
    with open(f'./data/custom_files/{subjects}hrf_dict.pkl', 'wb') as fp:
        pickle.dump(hrf_dict, fp)
    
            
    return hrf_dict, voxdict_select, joint_voxels, size_selected_voxels

def univariate_regression(X, y, z_scorey:bool = False, meancentery:bool = False):
    # Reshape X to (n_imgs, 1) if it's not already
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Reshape y to (n_imgs, 1) if it's not already
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if z_scorey:
        y = get_zscore(y, print_ars = 'n')
        
    if meancentery:
        y = mean_center(y, print_ars = 'n')

    # Fit the univariate regression model
    model = LinearRegression().fit(X, y)

    # Extract beta coefficient and intercept
    beta_value = model.coef_
    intercept_value = model.intercept_

    # Calculate R-squared value
    rsquared_value = model.score(X, y)

    return beta_value, intercept_value, rsquared_value, model, X, y

def multivariate_regression(X, y_matrix, z_scorey:bool = False, meancentery:bool = False, fit_intercept:bool = False):
    # Reshape X to (n_imgs, 1) if it's not already
    if X.ndim == 1:
        X = X.reshape(-1, 1)


    if z_scorey:
        y_matrix = get_zscore(y_matrix, print_ars = 'n')
        
    if meancentery:
        y_matrix = mean_center(y_matrix, print_ars = 'n')

    # Fit the multivariate regression model
    model = LinearRegression(fit_intercept=fit_intercept).fit(X, y_matrix)

    # Extract beta coefficients and intercepts
    beta_values = model.coef_
    intercept_values = model.intercept_

    # Calculate R-squared values
    rsquared_values = model.score(X, y_matrix)

    return beta_values, intercept_values, rsquared_values, model, X, y_matrix


def regression_dict_multivariate(subject, feat_type, voxels, hrfs, feat_vals, n_imgs='all', 
                                 z_scorey:bool = False, z_scorex:bool = False, meancentery:bool = False,
                                 fit_intercept:bool = False, non_patch:bool = False,
                                 multi_feats:bool = False, rms:bool = False, sc:bool = False, ce:bool = False):
    
        # Load in the new RMS dict (where RMS is calculated after cropping, thus cropping prior to RMS)
    with open('./data/custom_files/all_visfeats_rms_crop_prior.pkl', 'rb') as fp:
        visfeats_rms_crop_prior = pickle.load(fp)
    
    with open('/home/rfpred/data/custom_files/all_visfeats_scce.pkl', 'rb') as fp:
        visfeats_scce = pickle.load(fp)
    
    reg_dict = {}
    
    # Set the amount of images to regress over in case all images are available.
    if n_imgs == 'all':
        n_imgs = len(feat_vals)
    
    if non_patch:
        irrel = '_irrelevant'
    else:
        irrel = ''
    
    if multi_feats:
        # Set empty array
        feats = []

        # Conditionally add arrays based on the values of rms, sc, and ce
        if rms:
            rms_X = visfeats_rms_crop_prior[subject][f'rms{irrel}']['rms_z']
            feats.append(rms_X)
        if sc:
            sc_X = visfeats_scce[subject][f'scce{irrel}']['sc_z']
            feats.append(sc_X)
        if ce:
            ce_X = visfeats_scce[subject][f'scce{irrel}']['ce_z']
            feats.append(ce_X)

        # Stack the arrays vertically and then transpose
        X = np.vstack(feats).T
        
    else:
    # Set the input matrix for the regression analysis
        X = np.array(feat_vals[feat_type][:n_imgs])  # Now X can have multiple columns

    # This function will run the multiple regression analysis for each voxel, roi, image, for a subject.
    rois = list(voxels[subject].keys())

    for roi in rois:
        reg_dict[roi] = {}
        voxel_mask = voxels[subject][roi]  # These are the boolean mask for the specific subject, roi
        n_voxels = np.sum(voxel_mask).astype('int')  # This is the number of voxels in this roi
        vox_indices = np.column_stack(np.where(voxel_mask == 1))  # Get voxel indices for the current ROI
        
        # Extract y_matrix for all voxels within the ROI
        y_matrix = np.array([hrfs[subject][roi][f'voxel{voxel + 1}']['hrf_betas'] for voxel, xyz in enumerate(vox_indices)]).T
        
        if z_scorey:
            # Reshape y_matrix into 40 batches of 750 values
            y_matrix_reshaped = y_matrix.reshape(-1, 750, y_matrix.shape[1])

            # Initialize an empty array to store the z-scores
            z_scores = np.empty_like(y_matrix_reshaped)

            # Calculate the z-scores for each batch
            for i in range(y_matrix_reshaped.shape[0]):
                z_scores[i] = get_zscore(y_matrix_reshaped[i], print_ars='n')

            # Flatten z_scores back into original shape
            y_matrix = z_scores.reshape(y_matrix.shape)
                        
        if meancentery:
            # Idem dito
            y_matrix_reshaped = y_matrix.reshape(-1, 750, y_matrix.shape[1])
            mc_scores = np.empty_like(y_matrix_reshaped)
            for i in range(y_matrix_reshaped.shape[0]):
                mc_scores[i] = mean_center(y_matrix_reshaped[i], print_ars='n')
            y_matrix = mc_scores.reshape(y_matrix.shape)
            
        # Perform multivariate regression
        beta_values, intercept_values, rsquared_value, reg_model, X_used, y_used = multivariate_regression(X, y_matrix, z_scorey = z_scorey, meancentery = meancentery, fit_intercept = fit_intercept)
        
        # If no intercept values are fit, set the output to all zeros.
        # N.B. in reality these values are not exactly 0, so don't make the mistake of interpreting them as such.
        # We just don't fit the intercept because we z-score both X and y, thus we theoretically shouldn't need an icept.
        if fit_intercept == False:
            intercept_values = np.zeros_like(beta_values)
            
        
        reg_dict[roi]['voxels'] = {}
        
        for voxel, vox_idx in enumerate(vox_indices):
            reg_dict[roi]['voxels'][f'vox{voxel}'] = {
                'xyz': list(vox_idx),
                'beta': beta_values[voxel],
                'icept': intercept_values[voxel]
            }
            
        reg_dict[roi]['y_matrix'] = y_matrix
        reg_dict[roi]['X_matrix'] = X_used
        reg_dict[roi]['all_reg_betas'] = beta_values
        reg_dict[roi]['all_intercepts'] = intercept_values
        reg_dict[roi]['rsquared'] = rsquared_value
    
    return reg_dict, X



def plot_roi_beta_distribution(reg_dict, dictdescrip1 = '', icept_correct = None, feat_type = '', comparison_reg_dict = None, dictdescrip2 = '', comptype = ''):
    plt.style.use('dark_background')  # Apply dark background theme
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
    
    # COol custom color palettes
    colors = LinearSegmentedColormap.from_list("my_palette", ["darkblue", "blue"], N=len(reg_dict))
    comparison_colors = LinearSegmentedColormap.from_list("my_palette", ["darkred", "red"], N=len(comparison_reg_dict)) if comparison_reg_dict else None

    num_bins = 40  # Specify the number of bins
    icept_values = []
    min_x = np.inf
    max_x = -np.inf
    for i, (roi, voxels) in enumerate(reg_dict.items()):
        
        beta_values = np.concatenate([voxel_data['beta'] for voxel_data in voxels['voxels'].values()])
        plot_vals = beta_values
        icept_values = np.concatenate([np.array([voxel_data['icept']]) for voxel_data in voxels['voxels'].values()])        

        if icept_correct == 'y':
            plot_vals = beta_values / icept_values

        sns.histplot(plot_vals, kde=True, ax=axes[i], color=colors(i/len(reg_dict)), label=f'{dictdescrip1} ({len(beta_values)} voxels)', bins=num_bins, edgecolor='black', alpha = .8)  # Specify the bins

        min_x = min(min_x, np.min(plot_vals))
        max_x = max(max_x, np.max(plot_vals))

        if comparison_reg_dict is not None and roi in comparison_reg_dict:
            comparison_voxels = comparison_reg_dict[roi]
            comparison_beta_values = np.concatenate([voxel_data['beta'] for voxel_data in comparison_voxels['voxels'].values()])
            comparison_plot_vals = comparison_beta_values
            comparison_icept_values = np.concatenate([np.array([voxel_data['icept']]) for voxel_data in comparison_voxels['voxels'].values()])        

            if icept_correct == 'y':
                comparison_plot_vals = comparison_beta_values / comparison_icept_values

            sns.histplot(comparison_plot_vals, kde=True, ax=axes[i], color=comparison_colors(i/len(comparison_reg_dict)), label=f'{dictdescrip2} ({len(comparison_beta_values)} voxels)', bins=num_bins, edgecolor='black', alpha = .9)

            min_x = min(min_x, np.min(comparison_plot_vals))
            max_x = max(max_x, np.max(comparison_plot_vals))

        axes[i].set_title(f'Distribution of Beta Values for {roi[:2]}', color='white')

        axes[i].set_ylabel('Occurrence freq', weight = 'normal', fontsize = 12, color='white')
        axes[i].tick_params(colors='white')
        axes[i].legend(fontsize = 'large')

    for ax in axes:
        ax.set_xlim(min_x, max_x)  # Set the x range to include all data
        # ax.set_xticks(np.around(np.arange(min_x, max_x, .05), 2))  # Set the ticks to be more frequent and round them
        
        # Calculate the maximum absolute value of min_x and max_x
        max_abs = max(abs(min_x), abs(max_x))

        # Generate ticks for negative values, 0 and positive values
        negative_ticks = np.around(np.arange(min_x, 0, 0.1), 2)
        positive_ticks = np.around(np.arange(0, max_abs + 0.1, 0.1), 2)  # +0.05 to include max_abs in the range

        # Combine negative and positive ticks
        ticks = np.concatenate((negative_ticks, positive_ticks))

        ax.set_xticks(ticks)
                
                
        
    if len(comptype) != 0: comptype = f'\n{comptype}' 
    
    fig.suptitle(f'Baseline visual feature regression of subject 1, {feat_type}{comptype}', fontsize=16, y=1, color='white')

    plt.tight_layout()
    plt.show()
    
    
def plot_beta_to_icept(reg_dict, dictdescrip1 = '', comparison_reg_dict = None, feat_type = '', dictdescrip2 = '', comptype = ''):
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    rois = ['V1_mask', 'V2_mask', 'V3_mask', 'V4_mask']

    for i, roi in enumerate(rois):
        betas = [beta for vox in reg_dict[roi] for beta in reg_dict[roi]['all_reg_betas']]
        icepts = [icept for vox in reg_dict[roi] for icept in reg_dict[roi]['all_intercepts']]

        row = i // 2
        col = i % 2

        axs[row, col].scatter(betas, icepts, c='blue', label=dictdescrip1)
        
        if comparison_reg_dict and roi in comparison_reg_dict:
            comparison_betas = [beta for vox in comparison_reg_dict[roi] for beta in comparison_reg_dict[roi]['all_reg_betas']]
            comparison_icepts = [icept for vox in comparison_reg_dict[roi] for icept in comparison_reg_dict[roi]['all_intercepts']]
            axs[row, col].scatter(comparison_betas, comparison_icepts, c='red', label=dictdescrip2)

        axs[row, col].set_xlabel('betas', color='white')
        axs[row, col].set_ylabel('icepts', color='white')
        axs[row, col].set_title(roi[:2], color='white')
        axs[row, col].tick_params(colors='white')
    axs[0, 0].legend(fontsize = 'medium')
    
    if len(comptype) != 0: comptype = f'\n{comptype}' 

    fig.suptitle(f'Regression betas to intercepts of subject 1, {feat_type}{comptype}', fontsize=16, y=1, color='white')

    plt.tight_layout()
    plt.show()
    
    

    
    
def reg_plots(reg_dict, dictdescrip1 = '', icept_correct = None, feat_type = None, 
              beta_hist:bool = True, beta_icept:bool = True, comparison_reg_dict = None, 
              dictdescrip2 = '', comptype = ''):
            
    if beta_hist:
        plot_roi_beta_distribution(reg_dict = reg_dict, dictdescrip1 = dictdescrip1, icept_correct = icept_correct, 
                                   feat_type = feat_type, comparison_reg_dict = comparison_reg_dict, 
                                   dictdescrip2 = dictdescrip2, comptype = comptype)
    if beta_icept:
        plot_beta_to_icept(reg_dict = reg_dict, dictdescrip1 = dictdescrip1, comparison_reg_dict = comparison_reg_dict, 
                           feat_type = feat_type, dictdescrip2 = dictdescrip2, comptype = comptype)