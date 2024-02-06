import os
import matplotlib.pyplot as plt
import random
import h5py
import numpy as np
from scipy.io import loadmat
import cv2
from scipy.stats import weibull_min


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
    
    
    


def calculate_rms_contrast_circle(image_array, center, radius):
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

    # Draw circle on the original image
    image_with_circle = image_array.copy()
    cv2.circle(image_with_circle, center, radius, (0, 255, 0), thickness=2)  # Green circle

    # Calculate mean intensity
    mean_intensity = np.mean(patch_pixels)

    # Calculate RMS contrast within the circular patch
    rms_contrast = np.sqrt(np.mean((patch_pixels - mean_intensity)**2))

    # Alternative weibull fit, centered around 0 (so subtracting the mean of all intensities)
    # Works very badly, I think the problem only arises when pRFs are too small
    # centered_patch_pixels = patch_pixels - mean_intensity
    # weibull_params = weibull_min.fit(centered_patch_pixels)


    # Fit a Weibull distribution to pixel intensities
    weibull_params = weibull_min.fit(patch_pixels)

    # Plot contrast histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(patch_pixels, bins=50, density=True, color='lightblue', alpha=0.7)
    plt.title('Contrast Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Plot Weibull fit
    plt.subplot(1, 2, 2)
    plt.hist(patch_pixels, bins=50, density=True, color='lightblue', alpha=0.7)
    x_range = np.linspace(min(patch_pixels), max(patch_pixels), 100)
    plt.plot(x_range, weibull_min.pdf(x_range, *weibull_params), 'r-', lw=2, label='Weibull Fit')
    plt.title('Contrast Histogram with Weibull Fit')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Display the image with the circle
    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    ax.imshow(image_with_circle)
    ax.set_title('Natural Scene with highlighted pRF')
    ax.axis('off')  # Turn off axis
    plt.show()

    return rms_contrast, weibull_params, image_with_circle, mask, patch_pixels, mean_intensity


def calculate_rms_contrast_color(image_array):
    """
    Calculate the Root Mean Square (RMS) contrast of a color image.

    Parameters:
    - image_array (numpy.ndarray): Input image array of shape (height, width, channels).

    Returns:
    - float: RMS contrast value.
    """
    # Convert the image to grayscale
    gray_image = np.mean(image_array, axis=-1)

    # Calculate mean intensity
    mean_intensity = np.mean(gray_image)

    # Calculate RMS contrast
    rms_contrast = np.sqrt(np.mean((gray_image - mean_intensity)**2))

    return rms_contrast