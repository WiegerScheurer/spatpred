import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def multivariate_regression(X, y_matrix):
    # Add a constant term to the independent variable matrix
    X_with_constant = sm.add_constant(X)

    # Reshape y_matrix to (n_imgs, n_voxels)
    n_imgs, n_voxels = y_matrix.shape
    y_matrix_reshaped = y_matrix.reshape(n_imgs, n_voxels, 1)

    # Fit the multivariate regression model
    model = sm.OLS(y_matrix_reshaped, X_with_constant)
    results = model.fit()

    # Extract beta coefficients and intercepts
    beta_values = results.params[:-1, :]
    intercept_values = results.params[-1, :]

    return beta_values, intercept_values

def regression_dict_multivariate(subject, feat_type, voxels, hrfs, feat_vals, n_imgs='all'):
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
        
        y_matrix = np.array([hrfs[subject][roi][f'voxel{voxel + 1}']['hrf_betas'] for voxel, xyz in enumerate(vox_indices)]).T #/ 300
        # y_matrix = np.array([hrfs[roi][tuple(vox_idx)][:n_imgs] for vox_idx in vox_indices]).T / 300

        # Perform multivariate regression
        beta_values, intercept_values = multivariate_regression(X, y_matrix)

        for voxel, vox_idx in enumerate(vox_indices):
            reg_dict[roi][f'vox{voxel}'] = {
                'xyz': list(vox_idx),
                'beta': beta_values[:, voxel],
                'icept': intercept_values[voxel]
            }

    return reg_dict, X, y_matrix

def plot_roi_beta_distribution(reg_dict):
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    colors = sns.color_palette('ocean', n_colors=len(reg_dict))

    for i, (roi, voxels) in enumerate(reg_dict.items()):
        beta_values = np.concatenate([voxel_data['beta'] for voxel_data in voxels.values()])
        sns.histplot(beta_values, kde=True, ax=axes[i], color=colors[i], label=f'{roi} ROI')

        axes[i].set_title(f'Distribution of Beta Values for {roi[:2]}')
        axes[i].set_ylabel('Occurrence freq', weight = 'normal', fontsize = 12)
        axes[i].set_xlim(-1, 2)  # Set the same x range for all subplots

        axes[i].set_xticks(np.arange(-1, 10, .5))  # Set the ticks to be more frequent


    axes[-1].set_xlabel('Beta values', weight = 'normal', fontsize = 12)
    fig.suptitle('Multivariate regression approach (Subject 1, all images, HRF beta)', fontsize=16, y=1)

    plt.tight_layout()
    plt.show()
