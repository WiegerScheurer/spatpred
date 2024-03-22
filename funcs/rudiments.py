###################### IMGPROC.PY ##################

# Function to calculate rms scores for image independent of design matrix. Faster option, later should be used to fill the
# dictionaries based on the design matrices. 
# This is the sequential version of the parallel one above, not useful so delete.

def rms_all_seq(start, n, ecc_max = 1):
    rms_vec = []
    img_vec = range(start, start + n)
    dim = show_stim(hide = 'y')[0].shape[0]
    x = y = (dim + 1)/2
    radius = ecc_max * (dim / 8.4)
    mask_w_in = css_gaussian_cut(dim, x, y, radius)
    rf_mask_in = make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
    for i in img_vec:
        ar_in = show_stim(img_no = start + i, hide = 'y')[0]  
        
        rms_vec.append(get_rms_contrast_lab(ar_in, mask_w_in, rf_mask_in, normalise = True, plot = 'n'))
        
    rms_dict = pd.DataFrame({
        'rms': rms_vec
    })
    
    rms_dict.set_index(np.array(img_vec))
    return rms_dict
        
        
# This one works, but gives a different (yet correlated) rms value, for some reason. Figure out why

def calculate_rms_contrast_circle(image_array, center, radius, hist = 'n', circ_plot = 'n'):
    """
    Calculate the Root Mean Square (RMS) contrast and fit a Weibull distribution to pixel intensities
    within a circular patch in a color image.

    Parameters:
    - image_array (numpy.ndarray): Input color image array of shape (height, width, channels).
    - center (tuple): Center coordinates of the circular patch (x, y).
    - radius (int): Radius of the circular patch.

    Returns:
    - tuple: (RMS contrast value within the circular patch, Weibull parameters, image with circle drawn,
              histogram plot, Weibull fit plot)
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Extract circular patch
    mask = np.zeros_like(gray_image)
    cv2.circle(mask, center, radius, 1, thickness=-1)  # Filled circle as a mask
    patch_pixels = gray_image[mask == 1]

    # Calculate mean intensity
    mean_intensity = np.mean(patch_pixels)

    # Calculate RMS contrast within the circular patch
    rms_contrast = np.sqrt(np.mean((patch_pixels - mean_intensity)**2))

    # Alternative weibull fit, centered around 0 (so subtracting the mean of all intensities)
    # Works very badly, I think the problem only arises when pRFs are too small
    # centered_patch_pixels = patch_pixels - mean_intensity
    # weibull_params = weibull_min.fit(centered_patch_pixels)

    weibull_params = None
    image_with_circle = None

    if hist == 'y':
        # Plot contrast histogram
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(patch_pixels, bins=50, density=True, color='lightblue', alpha=0.7)
        plt.title('Contrast Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        # Plot Weibull fit
        
        # Fit a Weibull distribution to pixel intensities
        weibull_params = weibull_min.fit(patch_pixels)
        plt.subplot(1, 2, 2)
        plt.hist(patch_pixels, bins=50, density=True, color='lightblue', alpha=0.7)
        x_range = np.linspace(min(patch_pixels), max(patch_pixels), 100)
        plt.plot(x_range, weibull_min.pdf(x_range, *weibull_params), 'r-', lw=2, label='Weibull Fit')
        plt.title('Contrast Histogram with Weibull Fit')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
        
    if circ_plot == 'y':
            # Draw circle on the original image
        image_with_circle = image_array.copy()
        cv2.circle(image_with_circle, center, radius, (0, 255, 0), thickness=2)  # Green circle

        # Display the image with the circle
        fig, ax = plt.subplots(figsize=(8.8, 8.8))
        ax.imshow(image_with_circle)
        ax.set_title('Natural Scene with highlighted pRF')
        ax.axis('off')  # Turn off axis
        plt.show()
        

    return rms_contrast, weibull_params, image_with_circle, mask, patch_pixels, mean_intensity

# These two functions are coupled to run the feature computations in parallel.
# This saves a lot of time. Should be combined with the feature_df function to assign
# the values to the corresponding trials.
def rms_single(args, ecc_max = 1):
    i, start, n = args
    
    dim = show_stim(hide = 'y')[0].shape[0]
    x = y = (dim + 1)/2
    radius = ecc_max * (dim / 8.4)
    mask_w_in = css_gaussian_cut(dim, x, y, radius)
    rf_mask_in = make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
    
    ar_in = show_stim(img_no = i, hide = 'y')[0]
    
    if i % 100 == 0:
        print(f"Processing image number: {i} out of {n + start}")
    return get_rms_contrast_lab(ar_in, mask_w_in, rf_mask_in, normalise = True, plot = 'n')

def rms_all(start, n, ecc_max = 1):
    img_vec = list(range(start, start + n))

    # Create a pool of worker processes
    with Pool() as p:
        rms_vec = p.map(rms_single, [(i, start, n) for i in img_vec])

    rms_dict = pd.DataFrame({
        'rms': rms_vec
    })

    rms_dict = rms_dict.set_index(np.array(img_vec))
    return rms_dict

# Code to acquire the hrf parameters for each subject, roi, voxel
# Importantly, it allows working with the data without crashing (though only for max 3 sessions at a time). 
# It loads in the nifti files, extracts the required data, overwrites it.
def get_betas(subjects, voxels, start_session, end_session):
    beta_dict = {}
    
    if subjects == 'all':
        subjects = [f'subj{i:02d}' for i in range(1, 9)]
    else:
        subjects = [subjects]
    for subject in subjects:
        beta_dict[subject] = {}
        
        rois = list(voxels[subject].keys())

        hrf_betas = {}
        
        for session in range(start_session, end_session + 1):
            session += 1
            if session < 10:
                session_str = f'0{session}'
            else: session_str = f'{session}'
            
            # session_nifti = betas_ses1 # Uncomment to check functionality of the code, if betas_ses1 has been loaded before.
            session_nifti = (nib.load(f'/home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/betas_session{session_str}.nii.gz')).get_fdata(caching = 'unchanged')
            n_imgs = session_nifti.shape[3]
        
            print(f'Working on session: {session} of subject: {subject}')
            for roi in rois: 
                
                if session == (start_session + 1):
                    hrf_betas[roi] = {}
                    # beta_dict[subject][roi] = {}
        
                voxel_mask = voxels[subject][roi] # These is the boolean mask for the specific subject, roi
                n_voxels = np.sum(voxel_mask).astype('int') # This is the amount of voxels in this roi
                vox_indices = np.zeros([n_voxels, voxel_mask.ndim], dtype = int) # Initiate an empty array to store vox indices                
                
                for coordinate in range(vox_indices.shape[1]): # Fill the array with the voxel coordinates as indices
                    vox_indices[:, coordinate] = np.where(voxel_mask == 1)[coordinate]
                    
                for voxel in range(n_voxels):
                    vox_idx = vox_indices[voxel] # Get the voxel indices for the current voxel
                
                    hrf_betas_ses = (np.array(session_nifti[tuple(vox_idx)]).reshape(n_imgs, 1))/300 # Divide by 300 to return to percent signal change units.
                    
                    if session == (start_session + 1):
                        hrf_betas[roi][f'voxel{voxel + 1}'] = hrf_betas_ses
                    else:    
                        total_betas = np.append(hrf_betas[roi][f'voxel{voxel + 1}'], hrf_betas_ses)
                        
                        hrf_betas[roi][f'voxel{voxel + 1}'] = total_betas
                    
            with open('./data/custom_files/subj01/intermediate_hrf_save.pkl', 'wb') as fp:
                pickle.dump(hrf_betas, fp)
                print('     - Back-up saved to intermediate_hrf_save.pkl\n')
                    
        beta_dict[subject] = hrf_betas               
        
    with open(f'./data/custom_files/subj01/beta_dict{start_session}_{end_session}.pkl', 'wb') as fp:
        pickle.dump(beta_dict, fp)
        print('     - Back-up saved to beta_dict{start_session}_{end_session}.pkl\n')        
                
    return beta_dict



# Function that calculates rms but based on a RGB to LAB conversion, which follows the CIELAB colour space
# This aligns best with the way humans perceive visual input. 
def get_rms_contrast_lab(rgb_image, mask_w_in, rf_mask_in, normalise = True, plot = 'n'):
    # Convert RGB image to LAB colour space
    lab_image = color.rgb2lab(rgb_image)
    
    ar_in = lab_image[:, :, 0] # Extract the L* channel for luminance values, set as input array

    if normalise == True:
        ar_in = ar_in/np.max(ar_in)
    
    square_contrast=np.square((ar_in-(ar_in[rf_mask_in].mean())))

    msquare_contrast=(mask_w_in*square_contrast).sum()
    
    if plot == 'y':
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        plt.subplots_adjust(wspace=0.01)

        axs[1].set_title(f'rms = {np.sqrt(msquare_contrast):.2f}')
        axs[0].imshow(square_contrast, cmap = 'gist_gray')
        axs[0].axis('off') 
        axs[1].imshow(mask_w_in*square_contrast, cmap = 'gist_gray')
        axs[1].axis('off') 
        
    return (np.sqrt(msquare_contrast))
   
   
   



################################## RF_TOOLS ##################################



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
            
        # FIX THIS STILL!!! I think it works now, but I need to check it.
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

############################ analyses.py

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



def regression_dict(subject, feat_type, voxels, hrfs, feat_vals, n_imgs = 'all'):
    reg_dict = {}
    # Set the amount of images to regress over in case all images are available.
    if n_imgs == 'all':
        n_imgs = len(feat_vals)
        
    X = np.array(feat_vals[feat_type][:n_imgs]).reshape(n_imgs, 1) # Set the input matrix for the regression analysis
    
    # This function will run the multiple regression analysis for each voxel, roi, image, for a subject.
    rois = list(voxels[subject].keys())
    

    for roi in rois:
        reg_dict[roi] = {}
        voxel_mask = voxels[subject][roi] # These is the boolean mask for the specific subject, roi
        n_voxels = np.sum(voxel_mask).astype('int') # This is the amount of voxels in this roi
        vox_indices = np.zeros([n_voxels, voxel_mask.ndim], dtype = int) # Initiate an empty array to store vox indices
        
        for coordinate in range(vox_indices.shape[1]): # Fill the array with the voxel coordinates as indices
            vox_indices[:, coordinate] = np.where(voxel_mask == 1)[coordinate]
            
        for voxel in range(n_voxels):
            vox_idx = vox_indices[voxel] # Get the voxel indices for the current voxel
            # y = (np.array(hrfs[tuple(vox_idx)][:n_imgs]).reshape(n_imgs, 1))/300 # Set the output matrix for the regression analysis
            y = hrfs[subject][roi][f'voxel{voxel + 1}']['hrf_betas']
            beta, icept = multiple_regression(X, y)
            reg_dict[roi][f'vox{voxel}'] = {
            'xyz': list(vox_idx),
            'beta': beta,
            'icept': icept
            }
            
    return reg_dict, X, y



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

def regression_dict_multivariate(subject, feat_type, voxels, hrfs, feat_vals, n_imgs='all', z_scorey:bool = False, meancentery:bool = False):
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

        if z_scorey:
            
            
            y_matrix = get_zscore(y_matrix, print_ars = 'n')
            
        if meancentery:
            y_matrix = mean_center(y_matrix, print_ars = 'n')
            
        # Perform multivariate regression
        beta_values, intercept_values, rsquared_value, reg_model = multivariate_regression(X, y_matrix, z_scorey = z_scorey, meancentery = meancentery)
        
        
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

################################# USEFUL SCHEISSE

# Saving dictionary with visual feats

for subj in visfeats_rms:
    # Mean center the 'rms' values
    visfeats_rms[subj]['rms']['rms_mc'] = mean_center(visfeats_rms[subj]['rms']['rms'], print_ars = 'n')
    # Mean center the 'rms_irrelevant' values
    visfeats_rms[subj]['rms_irrelevant']['rms_mc'] = mean_center(visfeats_rms[subj]['rms_irrelevant']['rms'], print_ars = 'n')
    
    
# Save a dictionary 
with open(f'./data/custom_files/all_visfeats_rms8.pkl', 'wb') as fp:
    pickle.dump(visfeats_rms, fp)