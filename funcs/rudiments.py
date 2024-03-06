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

