import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy


# Function to create a dictionary containing all the relevant HRF signal info for the relevant voxels.
def get_hrf_dict(subjects, voxels):
    
    hrf_dict = {}
    
    for subject in [subjects]:
        hrf_dict[subject] = {}


        # Get a list of files in the directory
        files = os.listdir(f'/home/rfpred/data/custom_files/{subject}')

        # Filter files that start with "beta_dict" and end with ".pkl"
        filtered_files = [file for file in files if file.startswith("beta_dict") and file.endswith(".pkl")]

        # Sort files based on the first number after 'beta_dict'
        sorted_files = sorted(filtered_files, key=lambda x: int(''.join(filter(str.isdigit, x.split('beta_dict')[1]))))

        # Print the sorted file names
        for n_file, file_name in enumerate(sorted_files):
            print(file_name)
                
            # Load in the boolean mask for inner circle voxel selection per roi.
            with open(f'/home/rfpred/data/custom_files/subj01/{file_name}', 'rb') as fp:
                beta_session = pickle.load(fp)
            
            rois = list(beta_session[subject].keys())
            
            if n_file == 0:
                hrf_dict[subject] = copy.deepcopy(beta_session[subject])
            for roi in rois:
                # hrf_dict[subject][roi] = {}
                n_voxels = len(beta_session[subject][roi])
                # print(n_voxels)
                
                
                voxel_mask = voxels[subject][roi] # These is the boolean mask for the specific subject, roi
                
                vox_indices = np.zeros([n_voxels, 3], dtype = int) # Initiate an empty array to store vox indices
                
                for coordinate in range(vox_indices.shape[1]): # Fill the array with the voxel coordinates as indices
                    vox_indices[:, coordinate] = np.where(voxel_mask == 1)[coordinate]
                    
                for voxel in range(len(beta_session[subject][roi])):
                    hrf_betas_ses = copy.deepcopy(beta_session[subject][roi][f'voxel{voxel + 1}'])
                    # print(f'Processing voxel: {voxel + 1}')
                    
                    if n_file == 0:
                        # hrf_dict[subject][roi][f'voxel{voxel + 1}'] = {}
                        total_betas = hrf_betas_ses
                        hrf_dict[subject][roi][f'voxel{voxel+1}'] = {
                            'xyz': list(vox_indices[voxel]),
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
                        'hrf_betas': total_betas,
                        'hrf_betas_z': 0,
                        'hrf_rsquared': 0,
                        'hrf_rsquared_z': 0
                    }
            print(len(hrf_dict[subject][roi][f'voxel{voxel+1}']['hrf_betas']))
            
            
    with open(f'./data/custom_files/{subjects}hrf_dict.pkl', 'wb') as fp:
        pickle.dump(hrf_dict, fp)
    
            
    return hrf_dict


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

    # Squeeze the last dimension out of y_matrix_reshaped
    y_matrix_squeezed = np.squeeze(y_matrix_reshaped, axis=-1)

    # Calculate R-squared values
    ss_res = np.sum((y_matrix_squeezed - results.fittedvalues)**2, axis=0)
    ss_tot = np.sum((y_matrix_squeezed - np.mean(y_matrix_squeezed, axis=0))**2, axis=0)
    rsquared_values = 1 - ss_res / ss_tot

    return beta_values, intercept_values, rsquared_values

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
        beta_values, intercept_values, rsquared_values = multivariate_regression(X, y_matrix)

        for voxel, vox_idx in enumerate(vox_indices):
            reg_dict[roi][f'vox{voxel}'] = {
                'xyz': list(vox_idx),
                'beta': beta_values[:, voxel],
                'R2': rsquared_values[voxel],
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
