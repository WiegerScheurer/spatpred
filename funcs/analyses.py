import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import os
import pickle
import nibabel as nib
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from funcs.rf_tools import rsquare_selection, nsd_R2_dict
from funcs.utility import numpy2coords, coords2numpy, filter_array_by_size, find_common_rows, get_zscore, mean_center, print_dict_structure, _sort_by_column

# This function is dangerous, it should not be applied to the full dataset, but a subset of it.
# is not yet adapted to the right input
def _scale_betas(session_betas=None):
    # check if session_betas is a nifti file
    if type(session_betas) == str:
        session_np = nib.load(session_betas).get_fdata(caching='unchanged')
    else:
        session_np = session_betas

    # scale the beta values
    session_np = session_np / 300

    # Divide every voxel over their grand mean across all sessions for that specific voxel
    voxel_means = np.mean(session_np, axis=3, keepdims=True)
    session_np = session_np / voxel_means

    return session_np


def _optimize_rsquare(R2_dict_hrf, subject, dataset, this_roi, R2_threshold, verbose:int, stepsize):
    top_n = 1
    while True:
        top_n += stepsize
        if verbose:
            print(f'The top{top_n} R2 values are now included')
        highR2 = rsquare_selection(R2_dict_hrf, top_n, n_subjects=8, dataset=dataset)
        lowest_val = highR2[subject][this_roi][0,3]
        if verbose:
            print(lowest_val)
        if lowest_val < R2_threshold:
            break
    # Return the optimal top_n value, which is one less than the value that caused lowest_val to fall below R2_threshold
    return top_n - 1

# Okay this one is the actual good function. The other should be deleted and never be used again. 
def get_hrf_dict(subjects, voxels, prf_region='center_strict', min_size=0.1, max_size=1,
                 prf_proc_dict=None, max_voxels=None, plot_sizes='n', verbose:bool=False,
                 vismask_dict=None, minimumR2:int=100):
    hrf_dict = {}
    R2_dict_hrf = nsd_R2_dict(vismask_dict, glm_type = 'hrf')
    
    
    for subject in [subjects]:
        hrf_dict[subject] = {}

        # Load beta dictionaries for each session
        beta_sessions = []
        for file_name in sorted(os.listdir(f'/home/rfpred/data/custom_files/{subject}/{prf_region}/')):
            if file_name.startswith("beta_dict") and file_name.endswith(".pkl"):
                with open(f'/home/rfpred/data/custom_files/{subject}/{prf_region}/{file_name}', 'rb') as fp:
                    
                    beta_sessions.append(pickle.load(fp)[subject])

        rois = list(beta_sessions[0].keys())

        for n_roi, roi in enumerate(rois):
            hrf_dict[subject][roi] = {}
            
            # Determine the subject, roi specific optimal top number of R2 values to filter the voxels for
            optimal_top_n_R2 = _optimize_rsquare(R2_dict_hrf, 'subj01','nsd', roi, minimumR2, False, 250)
            print(f'Voxels in {roi[:2]} with a minimum R2 of {minimumR2} is approximately {optimal_top_n_R2}')
            # Fetch this specific number of selected top R2 values for this roi
            highR2 = rsquare_selection(R2_dict_hrf, optimal_top_n_R2, n_subjects = 8, dataset = 'nsd')[subject][roi]
            # print(f'The average R2 value for {roi}') # This does not make sense, because not filtered yet.
            voxel_mask = voxels[subject][roi] # So this is not the binary mask, but the prf-selection made with the heatmap function
            
            # if max_voxels is None or n_roi > 0:
                # vox_n_cutoff = numpy2coords(voxel_mask).shape[0]
                
            # This if statement is to allow for a size-based selection of voxels
            if min_size is not None and max_size is not None:
                preselect_voxels = numpy2coords(voxel_mask, keep_vals = True) # Get the voxel coordinates based on the prf selection
                # This is another array with coordinates on the first 3 columns and then a selected size on the 4th column
                size_selected_voxels = filter_array_by_size(prf_proc_dict[subject]['proc'][roi]['size'], min_size, max_size)
                
                joint_ar_prf = find_common_rows(size_selected_voxels, preselect_voxels, keep_vals = True) # Keep_vals keeps the values of the first array
                joint_ar_R2 = find_common_rows(joint_ar_prf, highR2, keep_vals = True) # Select based on the top R2 values
                if verbose:
                    print(f'This is joint_ar_R2 {joint_ar_R2[10:15,:]}')
                available_voxels = joint_ar_R2.shape[0] # Check how many voxels we end up with
                print(f'Found {available_voxels} voxels in {roi[:2]} with pRF sizes between {min_size} and {max_size}')
                
                selected_R2_vals = find_common_rows(highR2, joint_ar_R2, keep_vals = True)#[:,3] # Get a list of the R2 values for the selected voxels
                if verbose:
                    print(f'This is the final r2 vals {selected_R2_vals[10:15,:]}')

                # Check whether the amount of voxels available is more than a potential predetermined limit
                if max_voxels is not None and available_voxels > max_voxels:
                    
                    top_n_R2_voxels = _sort_by_column(selected_R2_vals, 3, top_n = 1000)[:max_voxels, :] # Sort the R2 values and select the top n
                    size_selected_voxels_cut = find_common_rows(joint_ar_R2, top_n_R2_voxels, keep_vals = True) # Get the pRF sizes of these voxels
                    print(f'The amount of voxels are manually restricted to {max_voxels} out of {available_voxels}')
                else: size_selected_voxels_cut = joint_ar_R2                
                
                final_R2_vals = find_common_rows(highR2, size_selected_voxels_cut, keep_vals = True) # Get a list of the R2 values for the selected voxels
                
                print(f'of which the average R2 value is {np.mean(final_R2_vals[:,3])}\n')

                # size_slct = size_selected_voxels_cut
                hrf_dict[subject][roi]['roi_sizes'] = size_selected_voxels_cut # This is to be able to plot them later on
                hrf_dict[subject][roi]['R2_vals'] = final_R2_vals # Idem dito for the r squared values

                n_voxels = size_selected_voxels_cut.shape[0]
                if verbose:
                    print(f'\tAmount of voxels in {roi[:2]}: {n_voxels}')

                # And the first three columns are the voxel indices
                array_vox_indices = size_selected_voxels_cut[:, :3]

                # Convert array of voxel indices to a set of tuples for faster lookup
                array_vox_indices_set = set(map(tuple, array_vox_indices))

                # Create a new column filled with zeros, to later fill with the voxelnames in the betasession files, and meanbeta values
                new_column = unscaled_betas = np.zeros((size_selected_voxels_cut.shape[0], 1))

                # Add the new column to the right of size_selected_voxels_cut
                find_vox_ar = np.c_[size_selected_voxels_cut, new_column].astype(object)

                # Iterate over the dictionary
                for this_roi, roi_data in beta_sessions[0].items():
                    for voxel, voxel_data in roi_data.items():
                        # Check if the voxel's vox_idx is in the array
                        if voxel_data['vox_idx'] in array_vox_indices_set:
                            if verbose:
                                print(f"Found {voxel_data['vox_idx']} in array for {this_roi}, {voxel}")

                            # Find the row in find_vox_ar where the first three values match voxel_data['vox_idx']
                            matching_rows = np.all(find_vox_ar[:, :3] == voxel_data['vox_idx'], axis=1)

                            # Set the last column of the matching row to voxel
                            find_vox_ar[matching_rows, -1] = voxel

            mean_betas = np.zeros((final_R2_vals.shape))
            
            # Check whether the entire fourth column is now non-zero:
            if verbose:
                print(f'\tChecking if all selected voxels are present in beta session file: {np.all(find_vox_ar[:, 4] != 0)}\n')
            for vox_no in range(n_voxels):
                # Get the xyz coordinates of the voxel
                vox_xyz = find_vox_ar[vox_no, :3]
                vox_name = find_vox_ar[vox_no, 4]
                xyz_to_name = np.hstack((vox_xyz.astype('int'), vox_name))
                if verbose:
                    print(f'This is voxel numero: {vox_no}')
                    print(f'The voxel xyz are {vox_xyz}')
                
                hrf_betas = []
                for session_data in beta_sessions:
                    if verbose:
                        print(f"There are {len(session_data[roi]['voxel1']['beta_values'])} in this beta batch")
                    these_betas = session_data[roi][vox_name]['beta_values']
                    # Flatten the numpy array and convert it to a list before extending hrf_betas
                    hrf_betas.extend(these_betas.flatten().tolist())
                
                # Reshape hrf betas into 40 batches of 750 values
                betas_reshaped = np.array(hrf_betas).reshape(-1, 750) #, np.array(hrf_betas).shape[1])

                # Initialize an empty array to store the z-scores
                betas_normalised = np.empty_like(betas_reshaped)

                # Calculate the z-scores for each batch
                for i in range(betas_reshaped.shape[0]):
                    betas_mean = np.mean(betas_reshaped[i])
                    betas_normalised[i] = get_zscore(((betas_reshaped[i] / betas_mean) * 100), print_ars='n')

                # Flatten z_scores back into original shape
                hrf_betas_z = betas_normalised.flatten()
                mean_beta = np.mean(hrf_betas_z)
                hrf_dict[subject][roi][vox_name] = {
                    'xyz': list(vox_xyz.astype('int')),
                    'size': size_selected_voxels_cut[vox_no,3],
                    'R2': final_R2_vals[vox_no,3],
                    'hrf_betas': hrf_betas,
                    'hrf_betas_z': hrf_betas_z,
                    'mean_beta': mean_beta
                    }
                unscaled_betas[vox_no] = mean_beta
            mean_betas[:, :3] = size_selected_voxels_cut[:,:3]
            mean_betas[:, 3] = get_zscore(unscaled_betas, print_ars='n').flatten()
            
            hrf_dict[subject][roi]['mean_betas'] = mean_betas # Store the mean_beta values for each voxel in the roi


            n_betas = len(hrf_dict[subject][roi][vox_name]['hrf_betas'])
            if verbose:
                print(f'\tProcessed images: {n_betas}')
            
    plt.style.use('default')

    if plot_sizes == 'y':
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Create a figure with 2x2 subplots
        axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
        cmap = plt.get_cmap('gist_heat')  # Get the 'viridis' color map
        for i, roi in enumerate(rois):
            sizes = hrf_dict[subject][roi]['roi_sizes'][:,3]
            color = cmap(i / len(rois))  # Get a color from the color map
            sns.histplot(sizes, kde=True, ax=axs[i], color=color, bins = 10)  # Plot on the i-th subplot
            axs[i].set_title(f'RF sizes for {roi[:2]} (n={sizes.shape[0]})')  # Include the number of voxels in the title
            axs[i].set_xlim([min_size-.1, max_size+.1])  # Set the x-axis limit from 0 to 2
        fig.suptitle(f'{prf_region}', fontsize=18)
        plt.tight_layout()
        plt.show()
        
    import re
    xyz_to_voxname = {tuple(vox['xyz']): name for name, vox in hrf_dict[subject][rois[0]].items()}
    # Example string
    s = 'voxel130'

    # Use regex to find the numerical values after 'voxel'
    match = re.search('voxel(\d+)', s)

    if match:
        # If a match is found, group(1) will return the first group in the match, which is the numerical value
        num = int(match.group(1))
        print(num)    
            
    
                
    return hrf_dict, find_vox_ar


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

##############################
# This is what Micha wrote for the example pseudocode:
def _get_coefs(X,y,model):
    pass

def _score_model(X,y,model):
    
    #
    for cv_loop in cv():
        x_train,x_test,y_train,y_test=this_cv 
        
        model.fit(x_train,y_train)
        this_score=model.score(x_test,y_test)
        scores.append(this_score)
        pass
    
            
    #################################
        

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
                                 multi_feats:bool = False, rms:bool = False, sc:bool = False, ce:bool = False,
                                 sc_large:bool = False):
    
        # Load in the new RMS dict (where RMS is calculated after cropping, thus cropping prior to RMS)
    with open('./data/custom_files/all_visfeats_rms_crop_prior.pkl', 'rb') as fp:
        visfeats_rms_crop_prior = pickle.load(fp)
    
    with open('/home/rfpred/data/custom_files/all_visfeats_scce.pkl', 'rb') as fp:
        visfeats_scce = pickle.load(fp)
    
    with open('/home/rfpred/data/custom_files/all_visfeats_scce_large.pkl', 'rb') as fp:
        visfeats_scce_large = pickle.load(fp)
    
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
            if sc_large:
                sc_X = visfeats_scce_large[subject][f'scce{irrel}']['sc_z']
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