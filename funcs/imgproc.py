import os
from posixpath import dirname
import matplotlib.pyplot as plt
import random
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
import cv2
from scipy.stats import weibull_min
from matplotlib.ticker import MultipleLocator
from skimage import color
from multiprocessing import Pool
from funcs.rf_tools import get_mask, make_circle_mask, css_gaussian_cut, make_gaussian_2d

# Function to show a randomly selected image of the nsd dataset
def show_stim(hide = 'n', img_no = 'random'):
    # Example code to show how to access the image files, these are all 73000 of them, as np.arrays
    # I keep it like this as it might be useful to also store the reconstructed images with the autoencoder
    # using a .hdf5 folder structure, but I can change this later on.

    stim_dir = '/home/rfpred/data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/'
    stim_files = os.listdir(stim_dir)

    with h5py.File(f'{stim_dir}{stim_files[0]}', 'r') as file:
        img_brick_dataset = file['imgBrick']
        
        if img_no == 'random':
            image_no = random.randint(0,img_brick_dataset.shape[0])
        else: image_no = img_no
        
        test_image = img_brick_dataset[image_no]
            
    if hide == 'n':
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        plt.title(f'Image number {image_no}')
        plt.axis('off')
        plt.show()
        
    return test_image, image_no

# Create design matrix containing ordered indices of stimulus presentation per subject
def get_imgs_designmx():
    
    subjects = os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata')
    exp_design = '/home/rfpred/data/natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat'
    
    # Load MATLAB file
    mat_data = loadmat(exp_design)

    # Order of the presented 30000 stimuli, first 1000 are shared between subjects, rest is randomized (1, 30000)
    # The values take on values betweeon 0 and 1000
    img_order = mat_data['masterordering']-1

    # The sequence of indices from the img_order list in which the images were presented to each subject (8, 10000)
    # The first 1000 are identical, the other 9000 are randomly selected from the 73k image set. 
    img_index_seq = (mat_data['subjectim'] - 1) # Change from matlab to python's 0-indexing
    
    # Create design matrix for the subject-specific stimulus presentation order
    stims_design_mx = {}
    stim_list = np.zeros((img_order.shape[1]))
    for n_sub, subject in enumerate(sorted(subjects)):
    
        for stim in range(0, img_order.shape[1]):
            
            idx = img_order[0,stim]
            stim_list[stim] = img_index_seq[n_sub, idx]
            
        stims_design_mx[subject] = stim_list.astype(int)
    
    return stims_design_mx
    
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

# This function creates a dataframe containing the rms contrast values for each image in the design matrix
# This way you can chronologically map the feature values per subject based on the design matrix image order
def feature_df(subject, feature, feat_per_img, designmx):

    if feature == 'rms': 
        
        ices = list(designmx[subject])
        rms_all = feat_per_img['rms'][ices]
        
        df = pd.DataFrame({'img_no': ices,
                           'rms':rms_all})
        
    df = df.set_index(np.array(range(0, len(df))))
        
    return df
    
# call it like this:
# feature_df_s1 = feature_df('subject', 'rms', all_feats, designmx = dmx)


# Function to get rms contrast for input image, mask patch, and weighted mask
def get_rms_contrast(ar_in,mask_w_in,rf_mask_in,normalise=True, plot = 'n'):

    """
    rms contrast, computing contrast with respect to uniform mean
    Args:

    ar_in (_type_): image array in

    mask_w_in (_type_): weighted receptive field (e.g. guass), sums to one

    rf_mask_in (_type_): boolean mask of receptive field

    normalise (bool, optional): _description_. Defaults to True.
    """
    ar_gray = np.mean(ar_in, axis = -1)
    ar_in = ar_gray
    
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

    return(np.sqrt(msquare_contrast))

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
   
   
# Function to get contrast features based on the design matrix of a subject. 
# Extend the function so it does so by mapping the precomputed rms values with the function below.  
def get_contrast_df(n_images = None, start_img_no = 0 ,roi = 'V1', subject = 'subj01', ecc_max = 1, ecc_strict = 'y', 
                     prf_proc_dict = None, binary_masks = None, rf_type = 'prf', contrast_type = 'rms_lab'):
    
    designmx = get_imgs_designmx()
    
    if n_images == 'all':
        n_images = len(designmx['subj01'])
    
    indices, rms_list, image_id_list= [], [], []
      
    img_vec = designmx[subject][start_img_no : n_images + start_img_no]  
    
    # for img_no in range(start_img_no, n_images + start_img_no):
    for n_img, img_id in enumerate(img_vec):
        ar_in = show_stim(img_no = img_id, hide = 'y')[0]
        
        if rf_type == 'prf':
            # Acquire mask based on subject, roi, outline. Type is cut_gaussian by default, based on NSD paper
            rf_info = get_mask(dim = 425, subject = subject, binary_masks = binary_masks, 
                                            prf_proc_dict = prf_proc_dict, type='cut_gaussian', roi=roi,
                                            plot = 'n', excl_reason = 'n', sigma_min=0, sigma_max = 4.2, 
                                            ecc_max = ecc_max, ecc_strict = ecc_strict)
            
            # Get the location and radius of the patch
            x, y = rf_info['x'].astype('int'), rf_info['y'].astype('int')
            radius = rf_info['pix_radius'].astype('int')
            mask_w_in = rf_info['mask']
            
            # Create boolean mask of exact same size as weighted pRF patch
            rf_mask_in = make_circle_mask(425, x, y, radius, fill = 'y', margin_width = 0)
            
        elif rf_type == 'center':
            dim = ar_in.shape[0]
            x = y = (dim + 1)/2
            radius = ecc_max * (dim / 8.4)
            mask_w_in = css_gaussian_cut(dim, x, y, radius)
            rf_mask_in = make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0)
            
        # Get root mean square contrast of image and add to list
        indices.append(n_img)
        
        if contrast_type == 'rms_lab':
            rms_list.append(get_rms_contrast_lab(ar_in, mask_w_in, rf_mask_in, normalise = True, plot = 'n'))
        elif contrast_type == 'rms':
            rms_list.append(get_rms_contrast(ar_in, mask_w_in, rf_mask_in, normalise = True, plot = 'n'))
            
        # rms_list.append(get_rms_contrast(ar_in, mask_w_in, rf_mask_in, normalise = True))
        image_id_list.append(designmx[subject][n_img])
        # roi_list.append(roi)
        # subject_list.append(subject)
        if n_img % 10 == 0:
            print(f"Processing image number: {n_img} out of {n_images + start_img_no}")

    contrast_df = pd.DataFrame({
        'rms': rms_list,
        'image_id': image_id_list
        # 'roi': roi_list,
        # 'subject': subject_list
    })
    
    # Remove this roi thing, or well, only when rf_type = 'center'
    # contrast_df.insert(2, 'roi', [roi] * contrast_df.shape[0])
    contrast_df.insert(2, 'subject', [subject] * contrast_df.shape[0])
    contrast_df.insert(3, 'central_radius', [ecc_max] * contrast_df.shape[0])
    
    contrast_df = contrast_df.set_index(np.array(indices))
    
    return contrast_df

# This function applies a gaussian filter to the loaded image
def get_img_prf(image, x = None, y = None, sigma = None, type = 'gaussian', heatmask = None, 
                binary_masks = None, prf_proc_dict = None, roi = 'V1', sigma_min=1, sigma_max=25, ecc_max = 4.2,
                rand_seed = None, invert = 'n', central = 'n', filter_dict = None, grid = 'n', fill = 'y'):
    # arguments can be specified manually, or generated randomly if none are given
    # when entered manually there is no specification of parameters (yet)
    # I have to think about whether this is actually relevant, don't think so.
    # It's nothing more than a check whether it works, which it does.
    dim = image.shape[0]
    masked_arr = np.zeros(image.shape) # Create empty array for masked image
    if type == 'heatmask':
        # prf_mask = np.mean(heatmask, axis=2)
        prf_mask = heatmask
    else:
        if x is None and y is None and sigma is None:
            prf_info = get_mask(dim = dim, subject = 'subj01', plot = 'n', 
                                binary_masks = binary_masks, prf_proc_dict = prf_proc_dict, 
                                type = type, sigma_min=sigma_min, sigma_max=sigma_max, ecc_max = ecc_max,
                                rand_seed = rand_seed, filter_dict = filter_dict, grid = grid)

        
        x, y, sigma = prf_info['x'], prf_info['y'], prf_info['pix_radius']
        masked_arr = np.zeros(image.shape) # Create empty array for masked image

        pix_radius = sigma #* (image.shape[0]/8.4)
        
        if central == 'y':
            x = y = ((dim + 1) / 2)
            pix_radius = (ecc_max * (dim / 8.4))

            
        if type == 'gaussian':
            prf_mask = make_gaussian_2d(dim, x, y, pix_radius)
        elif type == 'circle':
            prf_mask = make_circle_mask(dim, x, y, pix_radius)
        elif type == 'full_gaussian':
            prf_mask = make_gaussian_2d(dim, x, y, pix_radius)
        elif type == 'cut_gaussian':
            prf_mask = css_gaussian_cut(dim, x, y, pix_radius)
        # elif type == 'outline':
        #     prf_mask = (make_circle_mask(image.shape[0], x, y, pix_radius, fill = 'n'))
        elif type == 'outline':
            dim = image.shape[0]
            x = y = ((dim + 2)/2)
            # x_deg = y_deg = prf_angle = prf_ecc = prf_expt = 0
            # deg_radius = prf_size = ecc_max
            prf_mask = (make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), ecc_max * (dim / 8.4), fill = fill))
        else:
            raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian', and 'outline'.")

    # Apply the mask per layer of the input image using matrix multiplication
    for colour in range(image.shape[2]):
        if invert == 'n':
            masked_arr[:,:,colour] = image[:,:,colour] * prf_mask
        elif invert == 'y':
            masked_arr[:,:,colour] = image[:,:,colour] * (1 - prf_mask)
            

    # Normalize the masked image according to the RGB range of 0-255
    masked_img = masked_arr / 255

    # x, y = prf_info['x'].astype('int'), prf_info['y'].astype('int')
    # radius = prf_info['pix_radius'].astype('int')
    mask_w_in = prf_info['mask']
    
    # Create boolean mask of exact same size as weighted pRF patch
    rf_mask_in = make_circle_mask(425, x, y, sigma, fill = 'y', margin_width = 0)
    rms_contrast = get_rms_contrast(image,mask_w_in,rf_mask_in,normalise=True)
    # rms_contrast, weibull_pars, gray_image, mask, patch_pixels, mean_intensity = calculate_rms_contrast_circle(image, center = center, radius = prf_info['pix_radius'].astype(int), hist = 'y', circ_plot = 'y')
    # sc, loc, ce = weibull_pars # Extract the spatial coherence (shape), location (on x-axis), and contrast energy (width, scale)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(masked_img, cmap='bone', origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
    ax.set_title(f'Region Of Interest: {roi}\n'
                f'Voxel: [{prf_info["x_vox"]}, {prf_info["y_vox"]}, {prf_info["z_vox"]}]\n'
                f'pRF x,y,σ: {round(prf_info["x_deg"], 1), round(prf_info["y_deg"], 1), round(prf_info["deg_radius"], 1)}\n'
                f'Angle: {round(prf_info["angle"], 2)}°\nEccentricity: {round(prf_info["eccentricity"], 2)}°\n'
                f'Exponent: {round(prf_info["exponent"], 2)}\nSize: {round(prf_info["size"], 2)}°\n'
                f'Explained pRF variance (R2): {round(prf_info["R2"], 2)}%\n'
                f'Root Mean Square (RMS) contrast of patch: {round(rms_contrast, 2)}')
                # f'Contrast Energy (CE) of patch (Weibull width): {round(ce, 2)}\n'
                # f'Spatial Coherence (SC) of patch (Weibull shape): {round(sc, 2)}\n'
                # f'Weibull location on x-axis: {round(loc, 2)}')
    ax.set_xlabel('Horizontal Degrees of Visual Angle')
    ax.set_ylabel('Vertical Degrees of Visual Angle')

    if grid == 'y':
        ax.grid(which='both', linestyle='--', linewidth=0.5, color='black')

    # Set ticks at every 0.1 step
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    
    return prf_info #, weibull_pars

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
        
        for session in range(start_session, end_session):
            session += 1
            if session < 10:
                session = f'0{session}'
            
            # session_nifti = betas_ses1 # Uncomment to check functionality of the code, if betas_ses1 has been loaded before.
            session_nifti = (nib.load(f'/home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/betas_session{session}.nii.gz')).get_fdata(caching = 'unchanged')
            n_imgs = session_nifti.shape[3]
        
            print(f'Working on session: {session} of subject: {subject}')
            for roi in rois: 
                
                if session == '01':
                    hrf_betas[roi] = {}
                    # beta_dict[subject][roi] = {}
        
                voxel_mask = voxels[subject][roi] # These is the boolean mask for the specific subject, roi
                n_voxels = np.sum(voxel_mask).astype('int') # This is the amount of voxels in this roi
                vox_indices = np.zeros([n_voxels, voxel_mask.ndim], dtype = int) # Initiate an empty array to store vox indices                
                
                for coordinate in range(vox_indices.shape[1]): # Fill the array with the voxel coordinates as indices
                    vox_indices[:, coordinate] = np.where(voxel_mask == 1)[coordinate]
                    
                for voxel in range(n_voxels):
                    vox_idx = vox_indices[voxel] # Get the voxel indices for the current voxel
                
                    hrf_betas_ses = (np.array(session_nifti[tuple(vox_idx)]).reshape(n_imgs, 1))/300
                    
                    if session == '01':
                        hrf_betas[roi][f'voxel{voxel + 1}'] = hrf_betas_ses
                        
                    total_betas = np.append(hrf_betas[roi][f'voxel{voxel + 1}'], hrf_betas_ses)
                    
                    hrf_betas[roi][f'voxel{voxel + 1}'] = total_betas
                
            with open('./data/custom_files/intermediate_hrf_save.pkl', 'wb') as fp:
                pickle.dump(hrf_betas, fp)
                print('     - Back-up saved to intermediate_hrf_save.pkl\n')
                    
        beta_dict[subject] = hrf_betas               
        
    with open('./data/custom_files/beta_dict.pkl', 'wb') as fp:
        pickle.dump(beta_dict, fp)
        print('     - Back-up saved to beta_dict.pkl\n')        
                
    return beta_dict
