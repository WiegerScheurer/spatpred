import os
import matplotlib.pyplot as plt
import random
import h5py
import numpy as np
# from funcs.imgproc import show_stim
from scipy.io import loadmat


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
    