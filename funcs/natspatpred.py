import os
import sys
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns
import nibabel as nib
import pickle
import torchvision.models as models
import nibabel as nib
import h5py
import scipy.stats.mstats as mstats
import copy
import ipywidgets as widgets
import sklearn as sk

from skimage import color
from nilearn import plotting
from scipy.ndimage import binary_dilation
from PIL import Image
from importlib import reload
from scipy.io import loadmat
from matplotlib.ticker import MultipleLocator, NullFormatter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from colorama import Fore, Style
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from tqdm.notebook import tqdm

from matplotlib.lines import Line2D
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error    
from IPython.display import display
from math import sqrt
from scipy.special import softmax
from matplotlib.ticker import MaxNLocator
from typing import Union, Dict, List, Tuple, Optional

# print(sys.path)
print('soepstengesl')

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

from unet_recon.inpainting import UNet
from funcs.analyses import univariate_regression


class VoxelSieve:
    """
    A class used to represent a Voxel Sieve, filtering out all voxels that do not
    meet all requirements defined in the VoxelSieve initiation method.

    Attributes
    ----------
    size : ndarray
        The size of the pRFs.
    ecc : ndarray
        The eccentricity of the pRFs.
    angle : ndarray
        The angle of the pRFs.
    prf_R2 : ndarray
        The R2 value of the voxel from pRF.
    nsd_R2 : ndarray
        The R2 value of the voxel from NSD.
    vox_pick : ndarray
        Boolean array indicating which voxels meet the criteria.
    xyz : ndarray
        The x, y, z coordinates of the selected voxels.

    Methods
    -------
    __init__(self, prf_dict: Dict, roi_masks: Dict, NSP, subject: str, roi: str, max_size: float, min_size: float, patchbound: float, min_nsd_R2: int, min_prf_R2: int)
        Initializes the VoxelSieve instance with the given parameters.
    """
    def __init__(self, NSP, prf_dict: Dict, roi_masks: Dict, subject: str, roi: str, max_size: float, min_size: float, patchbound: float, min_nsd_R2: int, min_prf_R2: int):
        self.size = prf_dict[subject]['proc'][f'{roi}_mask']['size'][:,3]
        self.ecc = prf_dict[subject]['proc'][f'{roi}_mask']['eccentricity'][:,3]
        self.angle = prf_dict[subject]['proc'][f'{roi}_mask']['angle'][:,3]
        self.prf_R2 = prf_dict[subject]['proc'][f'{roi}_mask']['R2'][:,3]
        self.R2_dict = NSP.cortex.nsd_R2_dict(roi_masks, glm_type='hrf')
        self.nsd_R2 = self.R2_dict[subject]['R2_roi'][f'{roi}_mask'][:,3]
        self.sigmas, self.ycoor, self.xcoor = NSP.cortex.calculate_pRF_location(self.size, self.ecc, self.angle, (425,425))
        self.patchbound = patchbound
        self.vox_pick = (self.size < max_size) & (self.ecc+self.size < patchbound) * (self.size > min_size) & (self.nsd_R2 > min_nsd_R2) & (self.prf_R2 > min_prf_R2)
        self.xyz = prf_dict[subject]['proc'][f'{roi}_mask']['size'][:, :3][self.vox_pick]
                

class DataFetch():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    # Function to get the pRF-based voxel selections
    # IMPROVE: make sure that it also works for all subjects later on. Take subject arg, clean up paths.
    def prf_selections(self):
        prf_selection_paths = [
            './data/custom_files/subj01/prf_mask_center_strict.pkl',
            './data/custom_files/subj01/prf_mask_central_strict_l.pkl',
            './data/custom_files/subj01/prf_mask_central_halfloose.pkl',
            './data/custom_files/subj01/prf_mask_central_loose.pkl',
            './data/custom_files/subj01/prf_mask_periphery_strict.pkl'
        ]
        return {os.path.basename(file): self.fetch_file(file) for file in prf_selection_paths}
    
    # General file fetching function 
    def fetch_file(self, file_path:str):
        """
        General function to acquire saved data from various file types
        file_type: str, the types of files to be fetched, either features or prf_selections
        """
        _, ext = os.path.splitext(file_path)
        
        # Check if file is of .h5 type
        if ext == '.h5':
            with h5py.File(file_path, 'r') as hf:
                data = hf.keys()
                return {key: np.array(hf[key]).flatten() for key in data}
        # Check if file is of .pkl type
        elif ext == '.pkl':
            with open(file_path, 'rb') as fp:
                return pickle.load(fp)
            
            
    # Function to load in nifti (.nii.gz) data and create some useful variables 
    def get_dat(self, path:str):
        full_dat = nib.load(path)
        dat_array = full_dat.get_fdata()
        
        # Calculate the range of values
        flat_arr = dat_array[~np.isnan(dat_array)]
        dat_dim = dat_array.shape

        return full_dat, dat_array, dat_dim, {'min': round(np.nanmin(flat_arr),7), 'max': np.nanmax(flat_arr), 'mean': round(np.nanmean(flat_arr),5)}
        
    # Hard coded, make flexible    
    # Function to load in all the computed predictability estimates, created using the get_pred.py and pred_stack.sh scripts.
    def load_pred_estims(self, subject = None, start = None, n_files = None, verbose:bool = False):
        dict_list = []

        # Get a list of files in the directory
        files = os.listdir(f'/home/rfpred/data/custom_files/{subject}/pred/')

        # Filter files that start with "beta_dict" and end with ".pkl"
        filtered_files = [file for file in files if file.startswith('light_payloads') and file.endswith(".h5")]
        
        # Sort files based on the first number after 'beta_dict'
        sorted_files = sorted(filtered_files, key=lambda x: int(''.join(filter(str.isdigit, x.split('light_payloads')[1]))))

        # Load in the .h5 files
        for file_no, file in enumerate(sorted_files):
            if verbose:
                print(f'Now loading file {file_no + 1} of {len(sorted_files)}')
            # load in back dictionary
            with h5py.File(f'/home/rfpred/data/custom_files/{subject}/pred/{file}', 'r') as hf:
                data = hf.keys()
                    
                dict = {key: np.array(hf[key]) for key in data}
            
            dict_list.append(dict)
                
        return dict_list
        
        
    # What I Now need to figure out is whether it is doable to just save the aggregated version of this, or 
    # that it's quick enough to just stack them on the spot.
    def _stack_betas(self, subject:str, roi:str, verbose:bool, n_sessions:int) -> np.ndarray:
        """Hidden method to stack the betas for a given subject and roi

        Args:
            subject (str): The subject to acquire the betas for
            roi (str): The region of interest
            verbose (bool): Print out the progress
            n_sessions (int): The amount of sessions for which to acquire the betas

        Returns:
            np.ndarray: A numpy array with dimensions (n_voxels, n_betas) of which the first 3 columns
                represent the voxel coordinates and the rest the betas for each chronological trial
        """      
        with tqdm(total=n_sessions, disable=not verbose) as pbar:
            for session in range(1, 1+n_sessions):
                session_str = f'{session:02d}'
                betapath = f'/home/rfpred/data/custom_files/{subject}/betas/{roi}/'

                if session == 1:
                    init_sesh = np.load(f'{betapath}beta_stack_session{session_str}.npy')
                    stack = np.hstack((init_sesh[:,:3], self.nsp.utils.get_zscore(init_sesh[:,3:], print_ars='n')))
                else:
                    stack = np.hstack((stack,  self.nsp.utils.get_zscore(np.load(f'{betapath}beta_stack_session{session_str}.npy'), print_ars='n')))

                if verbose:
                    pbar.set_description(f'NSD session: {session}')
                    pbar.set_postfix({f'{roi} betas': f'{stack.shape} {round(self.nsp.utils.inbytes(stack)/1000000000, 3)}gb'}, refresh=True)

                pbar.update()
        return stack
            
class Utilities():

    def __init__(self):
        pass
        
    # Utility function to visualize dictionary structures
    def print_dict_structure(self, d, indent=0):
        for key, value in d.items():
            print(' ' * indent + str(key))
            if isinstance(value, dict):
                self.print_dict_structure(value, indent + 4)
                
    def print_large(self, item):
        with np.printoptions(threshold=np.inf):
            print(item)
            
    # Function to create the Gaussian image
    def make_gaussian_2d(self, size, center_row, center_col, sigma):
        rows = np.arange(size)
        cols = np.arange(size)
        rows, cols = np.meshgrid(rows, cols)
        exponent = -((rows - center_row)**2 / (2 * sigma**2) + (cols - center_col)**2 / (2 * sigma**2))
        gaussian = np.exp(exponent)
        return gaussian        
    
    # Function to create a circle mask
    def make_circle_mask(self, size, center_row, center_col, radius, fill='y', margin_width=1):
        rows = np.arange(size)
        cols = np.arange(size)
        rows, cols = np.meshgrid(rows, cols)

        # Calculate the distance from each point to the center
        distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)

        # Create a binary mask where values within the radius are set to 1, others to 0
        circle_mask = np.where(distances <= radius, 1, 0)

        # Create a dilated version of the binary mask to add a margin
        dilated_circle_mask = binary_dilation(circle_mask, iterations=margin_width)

        # Subtract the dilated version to create the outline
        outline_circle_mask = circle_mask - dilated_circle_mask

        if fill == 'y':
            return circle_mask
        elif fill == 'n':
            return -outline_circle_mask

    # def css_gaussian_cut(self, size, center_row, center_col, sigma):
    #     rows = np.arange(size)
    #     cols = np.arange(size)
    #     rows, cols = np.meshgrid(rows, cols)

    #     distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
    #     mask = np.where(distances <= sigma, 1, 0)

    #     exponent = -((rows - center_row)**2 / (2 * sigma**2) + (cols - center_col)**2 / (2 * sigma**2))
    #     gaussian = np.exp(exponent)
    #     gaussian *= mask
    #     return gaussian
    # import numpy as np

    def css_gaussian_cut(self, size:np.ndarray, center_row:np.ndarray, center_col:np.ndarray, sigma:np.ndarray):
        # Ensure all inputs are numpy arrays and have the same shape
        size = np.asarray(size).reshape(-1, 1, 1)
        center_row = np.asarray(center_row).reshape(-1, 1, 1)
        center_col = np.asarray(center_col).reshape(-1, 1, 1)
        sigma = np.asarray(sigma).reshape(-1, 1, 1)

        # Create a meshgrid for rows and cols
        rows = np.arange(size.max())
        cols = np.arange(size.max())
        rows, cols = np.meshgrid(rows, cols)

        # Calculate distances, mask, and exponent for all inputs at once using broadcasting
        distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
        mask = np.where(distances <= sigma, 1, 0)

        exponent = -((rows - center_row)**2 / (2 * sigma**2) + (cols - center_col)**2 / (2 * sigma**2))
        gaussian = np.exp(exponent)
        gaussian *= mask

        # Trim each 2D array in the stack to its corresponding size
        gaussian = np.array([gaussian[i, :s, :s] for i, s in enumerate(size.flatten())])

        return gaussian
    
    def get_zscore(self, data, print_ars = 'y'):
        mean_value = np.mean(data)
        std_dev = np.std(data)

        # Calculate z-scores
        z_scores = (data - mean_value) / std_dev

        if print_ars == 'y':
            print("Original array:", data)
            print("Z-scores:", z_scores)
            
        return z_scores

    def cap_values(self, array = None, lower_threshold = None, upper_threshold = None):
        
        if upper_threshold is None:
            upper_threshold = np.max(array)
        else:
            # Identify values above the upper threshold
            above_upper_threshold = array > upper_threshold
            
            # Identify the highest value below the upper threshold
            highest_below_upper_threshold = array[array <= upper_threshold].max()

            # Replace values above the upper threshold with the highest value below the upper threshold
            array[above_upper_threshold] = highest_below_upper_threshold

        if lower_threshold is None:
            lower_threshold = np.min(array)
        else:
            # Identify values below the lower threshold
            below_lower_threshold = array < lower_threshold

            # Identify the lowest value above the lower threshold
            lowest_above_lower_threshold = array[array >= lower_threshold].min()

            # Replace values below the lower threshold with the lowest value above the lower threshold
            array[below_lower_threshold] = lowest_above_lower_threshold

        return array

    def mean_center(self, data, print_ars = 'y'):
        mean_value = np.mean(data)

        # Mean centering
        centered_data = data - mean_value

        if print_ars == 'y':
            print("Original array:", data)
            print("Centered data:", centered_data)
            
        return centered_data

    # Function to generate a bell-shaped vector
    def generate_bell_vector(self, n, width, location, kurtosis=0, plot = 'y'):
        x = np.linspace(0, 1, n)
        y = np.exp(-0.5 * ((x - location) / width) ** 2)
        
        if kurtosis != 0:
            y = y ** kurtosis
        
        y /= np.sum(y)  # Normalize the vector to sum up to 1
        
        if plot == 'y':
            plt.scatter(x, y)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Bell-Shaped Vector')
            plt.show()
            
        return y

    def calculate_sigma(self, eccentricity, angle, visual_stimulus_size=8.4):
        # Convert polar coordinates to Cartesian coordinates
        x = eccentricity * np.cos(angle)
        y = eccentricity * np.sin(angle)

        # Calculate the scaling factor based on visual stimulus size and eccentricity range
        eccentricity_range = 1000  
        scaling_factor = visual_stimulus_size / eccentricity_range

        # Calculate sigma
        sigma = np.sqrt(x**2 + y**2) * scaling_factor

        return sigma, x, y
        
    def numpy2coords(self, boolean_array, keep_vals:bool = False):
        # Get the coordinates of the True values in the boolean array
        coordinates = np.array(np.where(boolean_array))
        
        if keep_vals:
            # Get the values at the coordinates
            values = boolean_array[coordinates[0], coordinates[1], coordinates[2]]
            
            # Add the values as a fourth row to the coordinates
            coordinates = np.vstack([coordinates, values])
        
        # Transpose the coordinates to get them in the correct shape
        coordinates = coordinates.T
        
        return coordinates

    def coords2numpy(self, coordinates, shape, keep_vals:bool = False):
        # Create an array with the same shape as the original array
        array = np.zeros(shape, dtype=float if keep_vals else bool)
        
        if keep_vals:
            # Set the cells at the coordinates to their corresponding values
            array[tuple(coordinates[:,:3].astype('int').T)] = coordinates[:,3]
        else:
            # Set the cells at the coordinates to True
            # if coordinates
            # array[tuple(coordinates.T)] = True
                array[tuple(coordinates[:,:3].astype('int').T)] = True
        
        return array

    def find_common_rows(self, values_array, mask_array, keep_vals:bool = False):
        cols_vals = values_array.shape[1] - 1
        cols_mask = mask_array.shape[1] - 1
        set1 = {tuple(row[:cols_vals]): row[cols_vals] for row in values_array}
        set2 = set(map(tuple, mask_array[:,:cols_mask]))
        
        common_rows = np.array([list(x) + ([set1[x]] if keep_vals else []) for x in set1.keys() & set2])
        return common_rows    

    def sort_by_column(self, array:np.ndarray, column_index:int, top_n:Union[int, str]):
        """Function to sort a numpy array based on one of the column values

        Args:
            array (np.ndarray): Input array
            column_index (int): The index of the column to sort by
            top_n (np.ndarray): The top number of rows to return

        Returns:
            np.array: sorted array containing only the top_n rows
        """                
        # Get the column
        column = array[:, column_index]

        # Get the indices that would sort the column
        sorted_indices = np.argsort(column)

        if top_n == 'all':
            cut_off = array.shape[0]
        else: cut_off = top_n

        # Reverse the indices to sort in descending order and get the top_n indices
        top_indices = sorted_indices[::-1][:cut_off]

        # Sort the entire array by these indices
        sorted_array = array[top_indices]

        return sorted_array

# DELETE
    # Function to return the voxel coordinates based on the parameter represented in the 4th column
    def filter_array_by_size(self, array, size_min, size_max):
        filtered_array = array[(array[:, 3] >= size_min) & (array[:, 3] <= size_max)]
        return filtered_array

    def ecc_angle_to_coords(self, ecc, angle, dim = 425):
        """_summary_

        Args:
            ecc (_type_): _description_
            angle (_type_): _description_
            dim (int, optional): _description_. Defaults to 425.

        Returns:
            _type_: _description_
        """        
        y = ((1 + dim) / 2) - (ecc * np.sin(np.radians(angle)) * (dim / 8.4)) #y in pix (c_index)
        x = ((1 + dim) / 2) + (ecc * np.cos(np.radians(angle)) * (dim / 8.4)) #x in pix (r_index)
        
        x = ecc * np.cos(np.radians(angle))
        y = ecc * np.sin(np.radians(angle))
        return x, y

    def voxname_for_xyz(self, xyz_to_voxname:np.array, x:int, y:int, z:int):
        # Create a boolean mask that is True for rows where the first three columns match val1, val2, val3
        mask = (xyz_to_voxname[:, 0] == x) & (xyz_to_voxname[:, 1] == y) & (xyz_to_voxname[:, 2] == z)

        # Use the mask to select the matching row(s) and the fourth column
        voxname = xyz_to_voxname[mask, 3]

        # If there is only one matching row, voxname will be a one-element array
        # You can get the element itself with:
        if voxname.size == 1:
            voxname = voxname[0]

        return voxname
    
    # Function to create a list solely containing roi-based voxels
    def roi_filter(self, roi_mask, input_array, nan2null:bool=False):
        roi_ices = np.argwhere(roi_mask != 0)

        # Create list that only contains the voxels of the specific roi
        roi_ar = np.column_stack((roi_ices, input_array[roi_ices[:, 0], roi_ices[:, 1], roi_ices[:, 2]]))

        # Turn the nan values into zeros for the angle parameter
        if nan2null:
            output_roi = np.nan_to_num(roi_ar, nan=0)
            
        # Filter away the nan values
        output_roi = roi_ar[~np.isnan(roi_ar).any(axis=1)]
        rounded_output_roi = np.round(roi_ar, 5)
        
        # Set print options to control precision and suppress scientific notation
        np.set_printoptions(precision=5, suppress=True)
        
        return rounded_output_roi
    
    # Get the layer names of a neural network
    def get_layer_names(self, model):
        # Initialize an empty list to store the layer names
        layer_names = ['input']

        # Initialize counters for each type of layer
        conv_counter = 0
        relu_counter = 0
        maxpool_counter = 0
        dropout_counter = 0
        linear_counter = 0

        # Iterate over the named modules of the model
        for name, module in model.named_modules():
            # Check the type of the module and update the corresponding counter
            if isinstance(module, torch.nn.Conv2d):
                conv_counter += 1
                layer_names.append(f'Conv2d_{conv_counter}')
            elif isinstance(module, torch.nn.ReLU):
                relu_counter += 1
                layer_names.append(f'ReLU_{relu_counter}')
            elif isinstance(module, torch.nn.MaxPool2d):
                maxpool_counter += 1
                layer_names.append(f'MaxPool2d_{maxpool_counter}')
            elif isinstance(module, torch.nn.Dropout):
                dropout_counter += 1
                layer_names.append(f'Dropout_{dropout_counter}')
            elif isinstance(module, torch.nn.Linear):
                linear_counter += 1
                layer_names.append(f'Linear_{linear_counter}')

        return layer_names
    
    # Check the memory size of an object
    def inbytes(self, thing):
        return sys.getsizeof(thing)
    
    # Function get the min and max x,y values in order to acquire a perfect square crop of the RF mask.
    def get_bounding_box(self, mask):
        # Get the indices where the mask is True
        y_indices, x_indices = np.where(mask)

        # Get the minimum and maximum indices along each axis
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        return x_min, x_max, y_min, y_max
    
    def make_img_3d(self, mask_in):
        """for 2d array, copy to make 3-dimensional"""
        return(np.repeat(mask_in[:,:,np.newaxis],3,axis=2))
    
    # This function is from the unet notebook, it is used to create the eval_mask
    def scale_square_mask(self, mask_in:np.ndarray, scale_fact=np.sqrt(1.5), mask_val=1, min_size=50):
        """given a square mask, scale width and height with a given factor

        in:
        - mask_in: ndarray, (2d or 3d)
            boolean-type mask image
        - mask_val: float/int/bool (default:1)
            the value to look for as the definition of in the circle of the mask.
        - min_size: int
            minimum size of the square mask.

        out:
        -scaled_mask: ndarray
            like the square input mask, but now with a square outline around the mask
        """

        def _do_scaling(_mask_in:np.ndarray, scale_fact=np.sqrt(2), mask_val=1, min_size=50):
            """inner function doing the actual scaling"""
            mask_out=copy.deepcopy(_mask_in)
            nz_rows,nz_cols=np.nonzero(_mask_in==mask_val)
            nz_r,nz_c=np.unique(nz_rows),np.unique(nz_cols)
            # determine square masks that spans the circle
            width, height = nz_r[-1]-nz_r[0], nz_c[-1]-nz_c[0]

            # make actual spanning mask a bit larger (delta determined by scale_fact or min_size)
            ideal_delta_w = max(np.round(((width*scale_fact) - width)*.5), (min_size - width) // 2)
            ideal_delta_h = max(np.round(((height*scale_fact) - height)*.5), (min_size - height) // 2)

            # Adjust deltas based on mask's proximity to image borders
            delta_w_left = min(ideal_delta_w, nz_c[0])
            delta_w_right = min(ideal_delta_w, mask_out.shape[1] - nz_c[-1] - 1)
            delta_h_top = min(ideal_delta_h, nz_r[0])
            delta_h_bottom = min(ideal_delta_h, mask_out.shape[0] - nz_r[-1] - 1)

            # If mask is near the border, expand on the other side
            if delta_w_left < ideal_delta_w:
                delta_w_right = max(ideal_delta_w * 2 - delta_w_left, delta_w_right)
            if delta_w_right < ideal_delta_w:
                delta_w_left = max(ideal_delta_w * 2 - delta_w_right, delta_w_left)
            if delta_h_top < ideal_delta_h:
                delta_h_bottom = max(ideal_delta_h * 2 - delta_h_top, delta_h_bottom)
            if delta_h_bottom < ideal_delta_h:
                delta_h_top = max(ideal_delta_h * 2 - delta_h_bottom, delta_h_top)

            mask_out[int(nz_r[0]-delta_h_top):int(nz_r[-1]+delta_h_bottom),
                    int(nz_c[0]-delta_w_left):int(nz_c[-1]+delta_w_right)] = mask_val
            # set values to 1, square mask
            return(mask_out)

        # switch dealing with RGB [colmns,rows,colours] vs grayscale images [columns,rows]
        if mask_in.ndim==3:
            mask_scaled=_do_scaling(mask_in[:,:,0],scale_fact=scale_fact, mask_val=mask_val, min_size=min_size)
            return(self.make_img_3d(mask_scaled))
        elif mask_in.ndim==2:
            return(_do_scaling(mask_in, scale_fact=scale_fact, mask_val=mask_val, min_size=min_size))
        else:
            raise ValueError('can only understand 3d (RGB) or 2d array images!')
        
    # Function to inspect the numeric range of an object
    def inspect_dat(self, data):
        print(f'Lowest value: {np.min(data)}')
        print(f'Highest value: {np.max(data)}')
        print(f'Mean value: {np.mean(data)}')
        print(f'Standard deviation: {np.std(data)}')
        return 
    
# This class only contains strictly non-functional methods, they form no part of the main functionality of the NSP.
# However, they do serve as useful tools to explore the data and a basis for optimal decision-making regarding the procedures.
class Explorations():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    # Create a plot in which the different calculations (CSS,nonlinear vs. linear) for pRF radius are compared
    # This serves solely as a visual inspection of what definition aligns better with the established literature
    def compare_radius(self, prf_dictionary, size_key='size', sigma_key='lin_sigma', x_lim=(0, 20), y_lim=(0, 8), ci = 95):
        """
        Plot pRF data for each ROI side by side.
        Parameters:
        - prf_plots_dict: Dictionary containing pRF data.
        - size_key: Key for accessing pRF size data in the dictionary.
        - sigma_key: Key for accessing linear sigma data in the dictionary.
        - x_lim: Tuple specifying the x-axis limits.
        - y_lim: Tuple specifying the y-axis limits.
        """
        # Set seaborn style to remove background grid
        sns.set_style("white")

        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Titles for subplots
        subplot_titles = ['pRF size as radius', 'Linear sigma as radius']

        # Loop over subplots
        for k, ax in enumerate(axes):
            # Set subplot title
            ax.set_title(subplot_titles[k])

            # Define a list of bright colors for each ROI
            bright_palette = sns.color_palette('bright', n_colors=len(prf_dictionary['subj01']['proc']))

            legend_handles = []
            all_subs = {'y': {'V1': np.empty(0), 'V2': np.empty(0), 'V3': np.empty(0), 'V4': np.empty(0)},
                        'x': {'V1': np.empty(0), 'V2': np.empty(0), 'V3': np.empty(0), 'V4': np.empty(0)}}

            for subject, dict_shelve in prf_dictionary.items():
                for j, (roi, data) in enumerate(dict_shelve['proc'].items()):
                    ecc_values = data.get('eccentricity', [])

                    if k == 0:
                        size_values = data.get(size_key, [])
                        y_data = size_values[:, 3]  # Assuming 4th column of prf_size
                        label_prefix = 'pRF size'
                    else:
                        sigma_values = data.get(sigma_key, [])
                        y_data = sigma_values[:, 3]  # Assuming 4th column of linear sigma
                        label_prefix = 'Linear sigma'

                    # Filter values within the desired range
                    valid_indices = (
                        (0 <= ecc_values[:, 3]) & (ecc_values[:, 3] <= 20) &
                        (0 <= y_data) & (y_data <= 20)
                    )

                    ecc_values = ecc_values[valid_indices]
                    y_data = y_data[valid_indices]

                    x_data = ecc_values[:, 3]  # Assuming 4th column of prf_ecc

                    all_subs['y'][f'{roi[:2]}'] = np.concatenate([all_subs['y'][f'{roi[:2]}'], y_data])
                    all_subs['x'][f'{roi[:2]}'] = np.concatenate([all_subs['x'][f'{roi[:2]}'], x_data])

            for n_col, roi in enumerate(['V1', 'V2', 'V3', 'V4']):
                # Collect legend handles for each ROI
                legend_handles.append(Line2D([0], [0], color=bright_palette[n_col], lw=2, label=f'{roi}'))

                # Plot all subjects' data for the current ROI
                sns.regplot(x=all_subs['x'][f'{roi}'], y=all_subs['y'][f'{roi}'],
                            scatter_kws={'alpha': 0, 's': 1, 'color': bright_palette[n_col]},
                            line_kws={'alpha': 1, 'color': bright_palette[n_col]},
                            truncate=False, ci=ci, ax=ax)

            sns.despine()

            # Set the limits of the x and y axes
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_xticks(np.arange(0, x_lim[1]+1, 5))
            ax.set_yticks(range(y_lim[1]+1))

            # Add legend outside the subplot
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xlabel('pRF eccentricity (deg)')
            ax.set_ylabel(f'{label_prefix} (deg)')

        # Adjust layout to prevent clipping of titles and labels
        plt.tight_layout()

        plt.show()
        return all_subs
    
    def prf_plots_new(self, subject:str):
        prf_info = ['angle', 'eccentricity', 'size']
        prf_plot_dict = {}
        
        # Load data for angle, eccentricity, and size
        for idx in prf_info:
            _, prf_plot_dict[idx], _, prf_range = self.nsp.datafetch.get_dat(f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/prf_{idx}.nii.gz')
        
        # Calculate the common boolean mask based on exclusion criteria
        common_mask = (~np.isnan(prf_plot_dict['eccentricity'])) & (prf_plot_dict['eccentricity'] < 1000) & (prf_plot_dict['size']<1000)  # Adjust conditions as needed

        # Apply the common boolean mask to all relevant arrays
        for key in prf_info:
            prf_plot_dict[key] = prf_plot_dict[key][common_mask]

        # Calculate sigma, x, and y
        sigma_array, x, y = self.nsp.utils.calculate_sigma(prf_plot_dict['eccentricity'], prf_plot_dict['angle'])

        # Add sigma_array, x, and y to prf_plot_dict
        prf_plot_dict.update({'sigma': sigma_array, 'x': x, 'y': y})

        # Calculate pRF location for each voxel
        prf_plot_dict['sigma_px'], prf_plot_dict['r_index'], prf_plot_dict['c_index'] = self.nsp.cortex.calculate_pRF_location(
            prf_plot_dict['size'], prf_plot_dict['eccentricity'], prf_plot_dict['angle']
        )

        # Plot histograms for all dictionary elements excluding NaN and large values
        for key, value in prf_plot_dict.items():
            if key in ['x', 'y']:
                # Skip histograms for 'x' and 'y'
                continue
            
            plt.figure()
            
            # Determine adaptive binning based on the range of valid values
            num_bins = min(50, int(np.sqrt(len(value))))  # Adjust as needed
            
            # Apply logarithmic scale to the data for better visibility
            plt.hist(np.log1p(value.flatten()), bins=num_bins, color='red', alpha=0.7)  # Using np.log1p to handle zero values
            plt.title(f'Histogram for {key} (excluding NaN and values > 950)')
            plt.xlabel(f'Log({key} + 1)')  # Adding 1 to avoid log(0)
            
            plt.ylabel('Frequency')
            plt.show()

        # Scatter plot for 'x' and 'y' as coordinates
        plt.figure()
        plt.scatter(prf_plot_dict['x'], prf_plot_dict['y'], c=prf_plot_dict['size'], cmap='cividis', s=prf_plot_dict['size'], alpha=0.7)
        plt.colorbar(label='Size (degrees of visual angle)')
        plt.title('pRF locations based on Eccentricity-Angle, coloured for pRF size')
        plt.xlabel('Eccentricity (degrees)')
        plt.ylabel('Angle (degrees)')
        plt.show()
        
        return prf_plot_dict
    
    # Function to compare the relative nsd R2 values (over all sessions) per pRF size
    # make sure this one also works for the prf R2 values. 
    def rsq_to_size(self, prf_dict:dict, roi_masks:dict, rsq_type='nsd'):        
        sns.set_style("white")
        bright_palette = sns.color_palette('bright', n_colors=len(prf_dict['subj01']['proc']))
        if rsq_type == 'nsd':
            rsq_all = self.nsp.cortex.nsd_R2_dict(roi_masks)

        # Create subplots for each ROI
        _, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for i, roi in enumerate(list(prf_dict['subj01']['proc'].keys())):
            data = {'RSQ': [], 'Size': []}

            for subject in list(prf_dict.keys()):
                
                if rsq_type == 'nsd':
                    
                    rsq = rsq_all[subject]['R2_roi'][roi][:, 3]
                elif rsq_type == 'prf':
                    rsq = prf_dict[subject]['proc'][roi]['R2'][:, 3]
                
                size = prf_dict[subject]['proc'][roi]['size'][:, 3]

                valid_indices = (0 <= size) & (size <= 8.5) & (rsq > 0)

                rsq = rsq[valid_indices]
                size = size[valid_indices]

                data['RSQ'] += list(rsq)
                data['Size'] += list(size)

            # Create a DataFrame for easy plotting
            df = pd.DataFrame(data)

            # Plotting
            sns.scatterplot(x='Size', y='RSQ', data=df, color=bright_palette[i], ax=axes[i], s= 1)
            axes[i].set_title(f'Region of interest: {roi[:2]}')
            
            # Regression line for the entire set of dots
            x = df['Size'].values.reshape(-1, 1)
            y = df['RSQ'].values
            
            axes[i].set_xticks(np.arange(0, 8.5, 0.5))
            axes[i].set_xlabel('pRF size in degrees of visual angle')
            axes[i].set_ylabel(f'R-squared % explained variance of {rsq_type} signal')

        plt.suptitle('Comparison of explained variance (R2) per pRF size for visual regions of interest')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout
        plt.show()

        return df
    
    def compare_NSD_R2s(self, roi_masks, prf_dict, subject:str, slice:int, cmap:str='afmhot'):

        R2_dict_hrf = self.nsp.cortex.nsd_R2_dict(roi_masks, glm_type = 'hrf')
        R2_dict_onoff = self.nsp.cortex.nsd_R2_dict(roi_masks, glm_type = 'onoff')

        _,axes = plt.subplots(3,1, figsize = (6,15))
        axes[0].imshow(R2_dict_hrf[subject]['full_R2']['R2_ar'][:,:,slice], cmap = cmap)
        axes[0].set_title('Haemodynamic Response Function signal fit R2 \nGLMdenoise Ridge Regression R-squared in percentages')
        axes[0].axis('off')
        axes[1].imshow(R2_dict_onoff[subject]['full_R2']['R2_ar'][:,:,slice], cmap = cmap)
        axes[1].set_title('Voxel-wise variance explained for simple ON-OFF GLM model\nR-squared in percentages')
        axes[1].axis('off')
        axes[2].imshow(self.nsp.utils.cap_values(prf_dict[subject]['nsd_dat']['R2']['prf_ar'], 0, 100)[:,:,slice], cmap = cmap)
        axes[2].set_title('Voxel-wise variance explained by pRF model\nR-squared in percentages')
        axes[2].axis('off')
        
    # Function to compare the different ways of reaching a pRF filter. Nonlinear (CSS) and linear
    def compare_masks(self, mask_dict:dict=None, prf_dict:dict=None, subject:str='subj01', roi:str='V1', sigma_min=0.1, 
                    sigma_max=4.2, ecc_min=0, ecc_max=4.2, angle_min=0, angle_max=360, 
                    peripheral_center=None, patch_radius=1, cmap='afmhot', peri_info:bool=False,
                    peri_angle_ecc=None):
    
        def plot_mask(ax, mask, title):
            ax.imshow(mask, cmap = cmap)
            ax.set_title(title)
            ax.axis('off')

        # Assuming get_mask returns the mask you want to plot
        dobbel = random.randint(1, 1000)

        circle_dict = self.nsp.cortex.get_mask(dim=425, subject=subject, binary_masks = mask_dict, 
                                        prf_proc_dict=prf_dict, type='circle', roi=roi,
                                        plot='n', excl_reason='n', sigma_min=sigma_min, sigma_max=sigma_max,
                                        ecc_min = ecc_min, ecc_max = ecc_max, angle_min = angle_min, angle_max = angle_max,
                                        peripheral_center = peripheral_center, peri_angle_ecc = peri_angle_ecc, rand_seed=dobbel, patch_radius = patch_radius, 
                                        peri_info = peri_info)

        gaus = self.nsp.utils.make_gaussian_2d(425, circle_dict['x'], circle_dict['y'], circle_dict['pix_radius'])
        full_gaus = self.nsp.utils.make_gaussian_2d(425, circle_dict['x'], circle_dict['y'], (circle_dict['size'] * (425 / 8.4)))
        cut_gaus = self.nsp.utils.css_gaussian_cut(425, circle_dict['x'], circle_dict['y'], (circle_dict['size'] * (425 / 8.4)))
        
        # Create a figure with 1 row and 3 columns
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        # Plot each mask on a separate axis
        plot_mask(axs[0], circle_dict['mask'], 'Boolean mask')
        plot_mask(axs[1], gaus, 'Simple Gaussian')
        plot_mask(axs[2], full_gaus, 'Full CSS input responsive Gaussian \n with prf_size as sigma')
        plot_mask(axs[3], cut_gaus, '1 SD cut CSS Gaussian')

        # Add a main title for the entire figure
        fig.suptitle(f'{roi} of {subject}', fontsize=14)

        plt.show()

class Cortex():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    def visrois_dict(self, verbose:bool=False) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], np.ndarray]:
        """
        This function loads the ROI masks for each subject, prints the number of non-zero voxels in each ROI if verbose is True,
        and returns a list of ROIs, a dictionary of binary masks for each subject, and a sum of all ROI masks for the first subject.

        Parameters:
        verbose (bool): If True, the function prints the number of non-zero voxels in each ROI.

        Returns:
        rois (list): A list of ROIs.
        binary_masks (dict): A dictionary of binary masks for each subject.
        viscortex_mask (numpy array): A sum of all ROI masks for the first subject.
        """
        binary_masks = {}

        for subj_no in range(1, len(self.nsp.subjects) + 1):
            if verbose:
                print(f'Fetching roi masks for subject {Fore.LIGHTBLUE_EX}{subj_no}{Style.RESET_ALL}')
            mask_dir = f'{self.nsp.datapath}/natural-scenes-dataset/nsddata/ppdata/subj0{subj_no}/func1mm/roi'

            # read in and sort all the filenames in the mapped masks folder for each subject
            non_binary_masks = sorted(file for file in os.listdir(mask_dir) if '_mask.nii' in file)
            subj_binary_masks = {mask[:-7]: (nib.load(os.path.join(mask_dir, mask)).get_fdata()).astype(int) for mask in non_binary_masks}

            if verbose:
                # Print the amount of non-zero voxels in the roi
                for key, subj_binary_mask in subj_binary_masks.items():
                    print(f" - {Fore.BLUE}{key[:2]}{Style.RESET_ALL}: {np.sum(subj_binary_mask)} voxels")
                    
            binary_masks[f'subj0{subj_no}'] = subj_binary_masks

        rois = [roi[:2] for roi in binary_masks['subj01'].keys()]
        viscortex_mask = sum(binary_masks['subj01'][f'{roi}_mask'] for roi in rois)
        
        return rois, binary_masks, viscortex_mask
    
    # # This function provides a dictionary with all the pRF data for all subjects and rois
    # def prf_dict(self, rois:list, roi_masks:dict):
    #     prf_dict = {}

    #     # Make a loop to go over all the subjects
    #     for subject in self.nsp.subjects:
    #         prf_dict[subject] = {}
    #         prf_dict[subject]['nsd_dat'] = {}
            
    #         # Initialize dictionaries if they don't exist
    #         prf_dict[subject]['proc'] = {}

    #         # Get the overall prf results, save them in a dict
    #         prf_types = ['angle', 'eccentricity', 'exponent', 'gain', 'meanvol', 'R2', 'size']

    #         for prf_type in prf_types:
    #             prf_path = f'{self.nsp.datapath}/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/prf_{prf_type}.nii.gz'
    #             prf_dat, prf_ar, prf_dim, prf_range = self.nsp.datafetch.get_dat(prf_path)
    #             prf_dict[subject]['nsd_dat'][prf_type] = {
    #                 'prf_dat': prf_dat,
    #                 'prf_ar': prf_ar,
    #                 'prf_dim': prf_dim,
    #                 'prf_range': prf_range
    #             }
    #         roi_list =  [f'{roistr}_mask' for roistr in rois]
    #         for roi in roi_list:
    #             prf_dict[subject]['proc'][roi] = {
    #                 prf_type : None for prf_type in prf_types
    #             } 
    #             for prf_type in prf_types:
    #                 prf_dict[subject]['proc'][roi][prf_type] = self.nsp.utils.roi_filter(roi_masks[subject][roi], prf_dict[subject]['nsd_dat'][prf_type]['prf_ar'])

    #         # Calculate the linear pRF sigma values, these tend to be smaller and don't take
    #         # into account the nonlinear relationship between input and neural respons
    #             lin_sigmas = prf_dict[subject]['proc'][roi]['size'][:,3] * np.sqrt(prf_dict[subject]['proc'][roi]['exponent'][:,3])
    #             prf_dict[subject]['proc'][roi]['lin_sigma'] = np.column_stack([prf_dict[subject]['proc'][roi]['size'][:,0:3], lin_sigmas])

    #     return prf_dict
    
    # This function should be reduced to just storing the subject specific np.arrays,, much more efficient.
    # I'll keep this version, but I'll comment out the unneccesary shit
    def prf_dict(self, rois:list, roi_masks:dict):
        """
        This function provides a dictionary with all the pRF data for all subjects and rois.

        Parameters:
        rois (list): A list of ROIs.
        roi_masks (dict): A dictionary of ROI masks for each subject.

        Returns:
        prf_dict (dict): A dictionary with all the pRF data for all subjects and rois.
        """
        prf_types = ['angle', 'eccentricity', 'exponent', 'gain', 'meanvol', 'R2', 'size']
        roi_list =  [f'{roistr}_mask' for roistr in rois]

        # First, create the 'nsd_dat' part of prf_dict
        prf_dict = {
            subject: {
                'nsd_dat': {
                    prf_type: {
                        # 'prf_dat': prf_dat, 
                        'prf_ar': prf_ar, # This is all you should need
                        # 'prf_dim': prf_dim,
                        # 'prf_range': prf_range
                    } for prf_type in prf_types for prf_dat, prf_ar, prf_dim, prf_range in [self.nsp.datafetch.get_dat(f'{self.nsp.datapath}/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/prf_{prf_type}.nii.gz')]
                }
            } for subject in self.nsp.subjects
        }

        # Then, add the 'proc' part to prf_dict #### THis is redundant. I have functions to get this type of map., just multiply by the roi mask
        for subject in prf_dict:
            prf_dict[subject]['proc'] = {
                roi: {
                    prf_type: self.nsp.utils.roi_filter(roi_masks[subject][roi], prf_dict[subject]['nsd_dat'][prf_type]['prf_ar']) for prf_type in prf_types
                } for roi in roi_list
            }


        # Calculate the linear pRF sigma values, these tend to be smaller and don't take
        # into account the nonlinear relationship between input and neural respons
        for subject in prf_dict:
            for roi in roi_list:
                lin_sigmas = prf_dict[subject]['proc'][roi]['size'][:,3] * np.sqrt(prf_dict[subject]['proc'][roi]['exponent'][:,3])
                prf_dict[subject]['proc'][roi]['lin_sigma'] = np.column_stack([prf_dict[subject]['proc'][roi]['size'][:,0:3], lin_sigmas])

        return prf_dict

    def calculate_pRF_location(self, prf_size:np.ndarray, prf_ecc:np.ndarray, prf_angle:np.ndarray, image_size:Tuple[int, int]=(425, 425), visual_angle_extent:float=8.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate pRF location (x, y, z) given the size, eccentricity, angle, image dimensions, and degrees of visual angle the image spans.

        Parameters:
        prf_size (np.ndarray): The size of the pRF in degrees of visual angle.
        prf_ecc (np.ndarray): The eccentricity of the pRF in degrees of visual angle.
        prf_angle (np.ndarray): The angle of the pRF in degrees.
        image_size (tuple): The dimensions of the image. Default is (425, 425).
        visual_angle_extent (float): The visual angle extent of the image. Default is 8.4.

        Returns:
        tuple: The sigma parameter in pixel units, the pRF y-position in row pixel units, and the pRF x-position in column pixel units.
        """
        # Calculate sigma parameter in degrees visual angle
        sigma = prf_size * np.sqrt(2)
        
        # Calculate sigma parameter in pixel units
        sigma_px = sigma * (image_size[0] / visual_angle_extent)
        
        # Calculate pRF y-position in row pixel units
        r_index = (image_size[0] + 1) / 2 + (prf_ecc * np.sin(np.radians(prf_angle)) * (image_size[0] / visual_angle_extent))
        
        # Calculate pRF x-position in column pixel units
        c_index = (image_size[1] + 1) / 2 + (prf_ecc * np.cos(np.radians(prf_angle)) * (image_size[0] / visual_angle_extent))

        return sigma_px, r_index, c_index
    

    def anat_templates(self, prf_dict: Dict) -> Dict[str, nib.Nifti1Image]:
        """
        Load subject-specific T1 anatomical maps to use as base for later overlays.

        Parameters:
        prf_dict (dict): A dictionary containing subject data.

        Returns:
        dict: A dictionary where each key is a subject ID and each value is a Nifti1Image object representing the subject's T1 anatomical map.
        """
        anat_temps = {}
        for subject in prf_dict.keys():
            anat_temps[subject] = nib.load(f'{self.nsp.datapath}/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/T1_to_func1mm.nii.gz')
        return anat_temps
    
    # Function to create a dictionary containing all the R2 explained variance data of the NSD experiment, could also be turned into a general dict-making func
    def nsd_R2_dict(self, roi_masks:dict, glm_type:str='hrf'):
        
        """
        Function to get voxel specific R squared values of the NSD.
        The binary masks argument takes the binary masks of the visual rois as input.
        The glm_type argument specifies the type of glm used, either 'hrf' or 'onoff'.
        
        """
        
        # n_subjects = len(os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata'))
        # subject_list = [f'subj{i:02d}' for i in range(1, n_subjects + 1)] 

        nsd_R2_dict = {}

        # Make a loop to go over all the subjects
        for subject in self.nsp.subjects:
            nsd_R2_dict[subject] = {}
            nsd_R2_dict[subject]['full_R2'] = {}
            nsd_R2_dict[subject]['R2_roi'] = {}
            
            # Create list for all visual rois
            roi_list = list(roi_masks[subject].keys())
            if glm_type == 'onoff':
                nsd_R2_path = f'/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata/{subject}/func1mm/R2.nii.gz'
            elif glm_type == 'hrf':
                nsd_R2_path = f'/home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/R2.nii.gz'
            nsd_R2_dat, nsd_R2_ar, nsd_R2_dim, nsd_R2_range = self.nsp.datafetch.get_dat(nsd_R2_path)
            nsd_R2_dict[subject]['full_R2'] = {
                    'R2_dat': nsd_R2_dat,
                    'R2_ar': nsd_R2_ar,
                    'R2_dim': nsd_R2_dim,
                    'R2_range': nsd_R2_range
                }
            
            for roi in roi_list:
                nsd_R2_dict[subject]['R2_roi'][roi] = self.nsp.utils.roi_filter(roi_masks[subject][roi], nsd_R2_dict[subject]['full_R2']['R2_ar'])

        return nsd_R2_dict
    
    def get_voxname_for_xyz(self, xyz_to_voxname, x, y, z):
        # Create a boolean mask that is True for rows where the first three columns match val1, val2, val3
        mask = (xyz_to_voxname[:, 0] == x) & (xyz_to_voxname[:, 1] == y) & (xyz_to_voxname[:, 2] == z)

        # Use the mask to select the matching row(s) and the fourth column
        voxname = xyz_to_voxname[mask, 3]

        # If there is only one matching row, voxname will be a one-element array
        # You can get the element itself with:
        if voxname.size == 1:
            voxname = voxname[0]

        return voxname
    
    # This is basically what is also present inside the get_mask function, but now it just gives you a list of voxels       
    def find_top_voxels(self, subject='subj01', roi='V1', n_voxels=None, prf_dict=None, vismask_dict=None, 
                    min_size=0, max_size=4.2, min_prf_R2=0, min_hrf_R2=0, min_ecc=0, max_ecc=1):
        
        roi_voxels = self.nsp.utils.numpy2coords(vismask_dict[subject][f'{roi}_mask'], keep_vals = False)
        prf_pars = (list(prf_dict['subj01']['proc']['V1_mask'].keys())[:-1])
        R2_dict = self.nsd_R2_dict(vismask_dict, glm_type = 'hrf')
        brain_shape = vismask_dict[subject][f'{roi}_mask'].shape

        if n_voxels == None or n_voxels == 'all':
            max_voxels = np.sum(vismask_dict[subject][f'{roi}_mask'])
        else: max_voxels = n_voxels

        top_vox_dict = {}
        voxel_count = 0
        for n_vox, these_coords in enumerate(roi_voxels):
            these_coords = list(these_coords)
            voxel_pars = {}
            voxel_pars['xyz'] = these_coords
            voxel_pars['roi'] = roi
            for prf_par in prf_pars:
                prf_par_vals = prf_dict[subject]['nsd_dat'][prf_par]['prf_ar'][these_coords[0], these_coords[1], these_coords[2]]
                voxel_pars[prf_par] = prf_par_vals
            # Also add the NSD Rsquared value to the dict.
            voxel_pars['nsdR2'] = R2_dict[subject]['full_R2']['R2_ar'][these_coords[0], these_coords[1], these_coords[2]]
            
            # Check if the voxel parameters meet the conditions
            if (min_ecc <= voxel_pars['eccentricity']+(voxel_pars['size']/2) <= max_ecc and
                min_size <= voxel_pars['size'] <= max_size and
                min_prf_R2 <= voxel_pars['R2'] and
                min_hrf_R2 <= voxel_pars['nsdR2']):
                top_vox_dict[f'voxel{voxel_count}'] = voxel_pars
                voxel_count += 1
                if voxel_count >= max_voxels:  # Stop after finding 10 voxels
                    break

        print(f'Found {voxel_count} voxels in {roi}')
        return top_vox_dict    
    
    # Get some good voxels that tick all the boxes of the selection procedure (location, size, R2)
    def get_good_voxel(self, subject=None, roi:str=None, hrf_dict=None, xyz_to_voxname=None,
                    pick_manually=None, plot:bool=True, prf_dict=None, vismask_dict=None,
                    selection_basis:str='R2'):
        """
        Description:
            This function picks out a good voxel based on the R2/prfsize/meanbeta score of the HRF fits in the NSD.
        Arguments:
            subject = the subject to use
            roi = the region of interest for which to find a good voxel
            hrf_dict = the corresponding hrf dictionary created using get_hrf_dict()
            xyz_to_voxname = the corresponding xyz to voxname array given by get_hrf_dict()
            pick_manually = optional argument to manually select one of the top voxels regarding R2 value, takes None or integer value
                            which then selects a specific voxel from high to low (so 0 selects the optimal R2/prfsize/meanbeta voxel)
            plot = option to plot
            prf_dict = dictionary with population receptive field information
            vismask_dict = dictionary with visual cortex masks for the subjects, rois
            selection_basis = either 'R2', 'prfsize', or 'meanbeta'. Determines on what values the selection is based.
        """
        
        if selection_basis == 'R2':
            select_values = 'R2_vals'
        elif selection_basis == 'meanbeta':
            select_values = 'mean_betas'
        elif selection_basis == 'prfsize':
            select_values = 'roi_sizes'
        
        if pick_manually is not None:
            which = pick_manually
        else: which = random.randint(1, hrf_dict[subject][f'{roi}_mask'][select_values].shape[0])
        indices = tuple(self.nsp.utils.sort_by_column(hrf_dict[subject][f'{roi}_mask'][select_values], 3, top_n = 100000)[which,:3].astype('int')) 
        
        voxelname = self.nsp.cortex.get_voxname_for_xyz(xyz_to_voxname, indices[0], indices[1], indices[2])
        
        if plot:
            self.plot_top_vox(dim = 425, vox_dict_item = None, type = 'cut_gaussian', 
                        add_central_patch = True, outline_rad = 1, xyz_only = indices, 
                        subject = subject, prf_dict = prf_dict, vismask_dict = vismask_dict)
            
        return indices, voxelname
        
    
    # Simple function to plot a specific voxel from the top_vox_dict, vox_dict_item ought to be the same
    # type of dict object returned by the find_top_voxels() function.
    def plot_top_vox(self, dim=425, vox_dict_item=None, type:str=None, add_central_patch:bool=False, 
                    outline_rad=1, xyz_only=None, subject=None, prf_dict=None, vismask_dict=None):
        
        if xyz_only is not None:
            xyz_only = tuple(xyz_only)
            R2_dict = self.nsd_R2_dict(vismask_dict, glm_type = 'hrf')

            x_vox, y_vox, z_vox = [xyz_only[i] for i in range(3)]
            prf_size = prf_dict[subject]['nsd_dat']['size']['prf_ar'][xyz_only]
            prf_angle = prf_dict[subject]['nsd_dat']['angle']['prf_ar'][xyz_only]
            prf_ecc = prf_dict[subject]['nsd_dat']['eccentricity']['prf_ar'][xyz_only]
            prf_expt = prf_dict[subject]['nsd_dat']['exponent']['prf_ar'][xyz_only]
            prf_gain = prf_dict[subject]['nsd_dat']['gain']['prf_ar'][xyz_only]
            prf_rsq = prf_dict[subject]['nsd_dat']['R2']['prf_ar'][xyz_only]
            nsdR2 = R2_dict[subject]['full_R2']['R2_ar'][xyz_only]
            prf_meanvol = prf_dict[subject]['nsd_dat']['meanvol']['prf_ar'][xyz_only]
            roi = self.find_roi(vismask_dict, subject, xyz_only)
        else:
                    
            # Get the coordinate indices of the voxel
            x_vox, y_vox, z_vox = [vox_dict_item['xyz'][i] for i in range(3)]
            prf_size = vox_dict_item['size']
            prf_angle = vox_dict_item['angle']
            prf_ecc = vox_dict_item['eccentricity']
            prf_expt = vox_dict_item['exponent']
            prf_gain = vox_dict_item['gain']
            prf_rsq = vox_dict_item['R2']
            nsdR2 = vox_dict_item['nsdR2']
            prf_meanvol = vox_dict_item['meanvol']
            roi = vox_dict_item['roi']
        
        # Calculate the radius
        sigma = prf_size * np.sqrt(prf_expt)
        sigma_pure = sigma * (dim / 8.4)
        
        # Get the 2d coordinates of the pRF
        y = ((1 + dim) / 2) - (prf_ecc * np.sin(np.radians(prf_angle)) * (dim / 8.4)) #y in pix (c_index)
        x = ((1 + dim) / 2) + (prf_ecc * np.cos(np.radians(prf_angle)) * (dim / 8.4)) #x in pix (r_index)

        if type == 'circle' or type == 'gaussian':
            deg_radius = sigma
            pix_radius = sigma_pure
        elif type == 'cut_gaussian' or type == 'full_gaussian' or type == 'outline':
            deg_radius = prf_size
            pix_radius = prf_size * (dim / 8.4)

        # Note: all the masks are made using pixel values for x, y, and sigma
        # Check whether the same is done later on, in the heatmaps and get_img_prf.
        if type == 'gaussian':
            prf_mask = self.nsp.utils.make_gaussian_2d(dim, x, y, sigma_pure)
        elif type == 'circle':
            prf_mask = self.nsp.utils.make_circle_mask(dim, x, y, sigma_pure)
        elif type == 'full_gaussian':
            prf_mask = self.nsp.utils.make_gaussian_2d(dim, x, y, prf_size * (dim / 8.4))
        elif type == 'cut_gaussian':
            prf_mask = self.nsp.utils.css_gaussian_cut(dim, x, y, prf_size * (dim / 8.4))
        else:
            raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian'.")
        
        central_patch = 0
        if add_central_patch:
            central_patch = self.nsp.utils.make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), outline_rad * (dim / 8.4), fill = 'n')

        # Convert pixel indices to degrees of visual angle
        degrees_per_pixel = 8.4 / dim
        x_deg = (x - ((dim + 2) / 2)) * degrees_per_pixel
        y_deg = (((dim + 2) / 2) - y) * degrees_per_pixel
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow((prf_mask + central_patch), cmap='bone', origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
        ax.set_title(f'Region Of Interest: {roi}\n'
                    f'Voxel: [{x_vox}, {y_vox}, {z_vox}]\n'
                    f'pRF x,y,σ: {round(x_deg, 1), round(y_deg, 1), round(deg_radius, 1)}\n'
                    f'Angle: {round(prf_angle, 2)}°\nEccentricity: {round(prf_ecc, 2)}°\n'
                    f'Exponent: {round(prf_expt, 2)}\nSize: {round(prf_size, 2)}°\n'
                    f'Explained pRF variance (R2): {round(prf_rsq, 2)}%\n'
                    f'NSD R-squared: {round(nsdR2, 2)}%\n'
                    f'pRF R-squared: {round(prf_rsq, 2)}%\n'
                    f'pRF Gain: {round((prf_gain / prf_meanvol) *100, 2)}% BOLD\n')
        ax.set_xlabel('Horizontal Degrees of Visual Angle')
        ax.set_ylabel('Vertical Degrees of Visual Angle')

        # Set ticks at every 0.1 step
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
    
    # Simple function to get the roi of a specific voxel, given its coordinates, a subject, and the binary vismask_dict
    def find_roi(self, vismask_dict, subject, voxel_coords):
        for roi in vismask_dict[subject].keys():
            if vismask_dict[subject][roi][voxel_coords] != 0:
                return roi
        return None
    
# IMPROVE, IMPLEMENT
    # Create a dictionary for the top n R2 prf/nsd values, the amount of explained variance
    # it does so for every visual roi and subject separately. dataset can be 'nsd' or 'prf'
    # and input_dict should be given accordingly.
    def rsquare_selection(self, input_dict=None, top_n = 1000, n_subjects = None, dataset = 'nsd'):
        rsq_dict = {}
        
        if dataset == 'prf':
            rois = input_dict['subj01']['proc'].keys()
        elif dataset == 'nsd':
            rois = input_dict['subj01']['R2_roi'].keys()

        for subj_no in range (1, n_subjects + 1):
            subj_rsq = {}
            for roi in rois:
                if dataset == 'prf':
                    rsq_ar = input_dict[f'subj0{subj_no}']['proc'][roi]['R2']
                elif dataset == 'nsd':
                    rsq_ar = input_dict[f'subj0{subj_no}']['R2_roi'][roi]
                rsq_sort = np.argsort(rsq_ar[:, -1])
                rsq_top = rsq_ar[rsq_sort[-top_n:]]
                subj_rsq[roi] = rsq_top
            rsq_dict[f'subj0{subj_no}'] = subj_rsq
        return rsq_dict
    
    # This function is capable of figuring out what the best top R2 selection is for a specific roi   
    def optimize_rsquare(self, R2_dict_hrf, subject, dataset, this_roi, R2_threshold, verbose:int, stepsize):
        top_n = 1
        while True:
            top_n += stepsize
            if verbose:
                print(f'The top{top_n} R2 values are now included')
            highR2 = self.rsquare_selection(R2_dict_hrf, top_n, n_subjects=8, dataset=dataset)
            lowest_val = highR2[subject][this_roi][0,3]
            if verbose:
                print(lowest_val)
            if lowest_val < R2_threshold:
                break
        # Return the optimal top_n value, which is one less than the value that caused lowest_val to fall below R2_threshold
        return top_n - 1
    
    
    class AllPRFConsidered(Exception):
        pass
    def get_mask(self, dim=200, subject:str='subj01', binary_masks=None, 
                prf_proc_dict=None, type='full_gaussian', roi='V1', 
                plot='y', heatmap='n', prf_vec=None, iter=None, excl_reason='n', peri_info:bool=False,
                sigma_min=0, sigma_max=4.2, ecc_min=0, ecc_max=4.2, rand_seed=None, filter_dict=None, 
                ecc_strict=None, grid='n', fill_outline='n', min_overlap=0, add_central_patch:bool=False,
                peripheral_center=None, peri_angle_ecc=None, angle_min=0, angle_max=360, patch_radius=1,
                prfR2_min=0, nsdR2_min=0):

        if rand_seed == None: random.seed(random.randint(1, 1000000))
        else: random.seed(rand_seed)
        
        # Get the nsd R-squared values, in addition to the prf R-squareds
        R2_dict = self.nsd_R2_dict(binary_masks)

        degrees_per_pixel = 8.4 / dim
        bin_prf = region_patch = None
        
        # Determine the center of the peripheral patch, if center is not given, but angle and eccentricity are
        if peripheral_center == None and peri_angle_ecc != None:
            patchloc_triangle_s = peri_angle_ecc[1]
            peri_y = round(peri_angle_ecc[1] * np.sin(np.radians(peri_angle_ecc[0])), 2)
            peri_x = round(peri_angle_ecc[1] * np.cos(np.radians(peri_angle_ecc[0])), 2)
            peripheral_center = (peri_x, peri_y)
            if peri_info:
                print(f'Peripheral center at {peripheral_center}')
        
        if isinstance(peripheral_center, tuple):
            # Determine the eccentricity of the patch using Pythagoras' theorem
            patchloc_triangle_o = np.abs(peripheral_center[1])
            patchloc_triangle_a = np.abs(peripheral_center[0])
            patchloc_triangle_s = np.sqrt(patchloc_triangle_o**2 + patchloc_triangle_a**2) # Pythagoras triangle side s, patch center eccentricity
            if peri_info:
                print(f'Patch localisation triangle with side lengths o: {round(patchloc_triangle_o, 2)}, a: {round(patchloc_triangle_a,2)}, s: {round(patchloc_triangle_s,2)}')
            
            # Determine the angle boundaries for the patch, also using Pythagoras
            bound_triangle_a = patchloc_triangle_s
            bound_triangle_o = patch_radius
        
            patch_bound_angle = np.degrees(np.arctan(bound_triangle_o / bound_triangle_a))
            patch_center_angle = np.degrees(np.arctan(np.abs(peripheral_center[1] / peripheral_center[0])))
            
            if peripheral_center[0] >= 0 and peripheral_center[1] > 0: # top right
                patch_center_angle = patch_center_angle
            elif peripheral_center[0] < 0 and peripheral_center[1] > 0: # top left
                patch_center_angle = 180 - patch_center_angle
            elif peripheral_center[0] >= 0 and peripheral_center[1] < 0: # bottom right
                patch_center_angle = 360 - patch_center_angle
            elif peripheral_center[0] < 0 and peripheral_center[1] < 0: # bottom left
                patch_center_angle = 180 + patch_center_angle
                
            angle_min = patch_center_angle - patch_bound_angle
            angle_max = patch_center_angle + patch_bound_angle
            ecc_min = patchloc_triangle_s - bound_triangle_o
            ecc_max = patchloc_triangle_s + bound_triangle_o
            
            if peri_info:
                print(f'ecc_min: {round(ecc_min,2)}, ecc_max: {round(ecc_max,2)}')
                print(f'Peripheral patch at angle {round(patch_center_angle,2)} with boundary angles at min: {round(angle_min,2)}, max: {round(angle_max,2)}')
        
        # Create objects for all the required pRF data
        roi_mask_data = prf_proc_dict[subject]['proc'][f'{roi}_mask']
        angle_roi, ecc_roi, expt_roi, size_roi, rsq_roi, gain_roi, meanvol_roi = roi_mask_data['angle'], roi_mask_data['eccentricity'], roi_mask_data['exponent'], roi_mask_data['size'], roi_mask_data['R2'], roi_mask_data['gain'], roi_mask_data['meanvol']

        # Define a mask to filter away data rows based on the filter_dict, which is supposed to be
        # a dictionary that includes a subset of filtered values for every subject, roi, based on
        # another parameter, such as explained mean variance, R2.
        if filter_dict != None:
            smaller_xyz = filter_dict[subject][f'{roi}_mask'][:, :3]
            mask = np.any(np.all(angle_roi[:, None, :3] == smaller_xyz, axis=-1), axis=1)
        else:
            mask = range(0, angle_roi.shape[0])
        
        # Condition for when the function is used to plot a heatmap, set to 'y', or any other value to do so
        if heatmap == 'n':
            prf_vec = random.sample(range(angle_roi[mask].shape[0]), angle_roi[mask].shape[0])
            iter = 0

        max_prf_vec = max(prf_vec)  # Maximum value of prf_vec

        while True:
            if iter >= max_prf_vec:
                raise self.AllPRFConsidered("All potential pRFs have been considered")

            n = prf_vec[iter]
            iter += 1

            prf_angle, prf_ecc, prf_expt, prf_size, prf_rsq, prf_gain, prf_meanvol = angle_roi[mask][n][3], ecc_roi[mask][n][3], expt_roi[mask][n][3], size_roi[mask][n][3], rsq_roi[mask][n][3], gain_roi[mask][n][3], meanvol_roi[mask][n][3]
            x_vox, y_vox, z_vox = int(angle_roi[mask][n][0]), int(angle_roi[mask][n][1]), int(angle_roi[mask][n][2])
            nsdR2 = R2_dict[subject]['full_R2']['R2_ar'][x_vox,y_vox,z_vox]
            sigma = prf_size * np.sqrt(prf_expt)
            sigma_pure = sigma * (dim / 8.4)
            outer_bound = inner_bound = prf_ecc
            prop_in_patch = 'irrelevant'
            
            # Condition to regulate the strictness of maximum eccentricity values
            # If ecc_strict is 'n', then any overlap suffices, as long as the centre is inside the patch
            if ecc_strict == 'y' and min_overlap == 100:
                outer_bound = prf_ecc + prf_size
                inner_bound = prf_ecc - prf_size
                
            # Sinus is used to calculate height, cosinus width
            # so c_index is the y coordinate and r_index is the x coordinate. 
            # the * (dim / 8.4) is the factor to translate it into raw pixel values
            y = ((1 + dim) / 2) - (prf_ecc * np.sin(np.radians(prf_angle)) * (dim / 8.4)) #y in pix (c_index)
            x = ((1 + dim) / 2) + (prf_ecc * np.cos(np.radians(prf_angle)) * (dim / 8.4)) #x in pix (r_index)
            
            if type == 'circle' or type == 'gaussian':
                deg_radius = sigma
                pix_radius = sigma_pure
            elif type == 'cut_gaussian' or type == 'full_gaussian' or type == 'outline':
                deg_radius = prf_size
                pix_radius = prf_size * (dim / 8.4)

            valid_conditions = (
                0 < x < dim,
                0 < y < dim,
                sigma_min < deg_radius,
                deg_radius < sigma_max,
                outer_bound < ecc_max,
                ecc_min < inner_bound,
                prf_ecc > ecc_min,
                angle_min < prf_angle < angle_max,
                nsdR2 > nsdR2_min,
                prf_rsq > prfR2_min            
                )

            middle_xy = (((dim + 1) / 2), ((dim + 1) / 2))
            if peripheral_center == None:
                center_x = center_y = middle_xy[0]
            elif isinstance(peripheral_center, tuple):
                center_x = middle_xy[0] + (peripheral_center[0] * dim/8.4)
                center_y = middle_xy[1] - (peripheral_center[1] * dim/8.4) # This is reversed because that's how they do it
                # in the NSD documentation. It has to do with where the y-axis starts, which is ('upper') in this case.
                # To verify this I checked the angle coordinates
                
            if all(valid_conditions):
                if ecc_strict == 'y' and min_overlap < 100: # Fix this condiional
                    
                    region_patch = self.nsp.utils.make_circle_mask(dim, center_x, center_y, patch_radius * dim/8.4, fill='y')
                    bin_prf = self.nsp.utils.make_circle_mask(dim, x, y, pix_radius)
                    prop_in_patch = (np.sum(bin_prf * region_patch) / np.sum(bin_prf)) * 100
                    # print(f'Proportion inside the patch: {prop_in_patch}')
                else: 
                    min_overlap = 0
                    prop_in_patch = 0
                    
                prop_condition = (
                    prop_in_patch >= min_overlap
                )
                if prop_condition:
                    break

          # Check for argument option to print reason for excluding voxels
            elif excl_reason == 'y':
                print(f"Discarding pRF mask for voxel [{x_vox}, {y_vox}, {z_vox}] due to:")
                conditions_messages = [
                    ("x out of bounds", valid_conditions[0]),
                    ("y out of bounds", valid_conditions[1]),
                    ("sigma_pure too small", valid_conditions[2]),
                    ("sigma_pure too large", valid_conditions[3]),
                    (f"pRF outside of central {2 * ecc_max}° visual degrees", valid_conditions[4]),
                    (f"pRF does not overlap enough with central patch: {prop_in_patch}% of required {min_overlap}%", valid_conditions[5]),
                    (f"pRF angle not within predetermined range of {angle_min}° to {angle_max}°", valid_conditions[6]),
                    (f"This voxel's NSD R-squared does not explain more than {nsdR2_min}% of the fMRI signal variance", valid_conditions[7]),
                    (f"This voxel's pRF R-squared does not explain more than {nsdR2_min}% of the fMRI signal variance", valid_conditions[7])
                ]
                for message, condition in conditions_messages:
                    if not condition:
                        print(f"   - {message}")
                        
        # Note: all the masks are made using pixel values for x, y, and sigma
        # Check whether the same is done later on, in the heatmaps and get_img_prf.
        if type == 'gaussian':
            prf_mask = self.nsp.utils.make_gaussian_2d(dim, x, y, sigma_pure)
        elif type == 'circle':
            prf_mask = self.nsp.utils.make_circle_mask(dim, x, y, sigma_pure)
        elif type == 'full_gaussian':
            prf_mask = self.nsp.utils.make_gaussian_2d(dim, x, y, prf_size * (dim / 8.4))
        elif type == 'cut_gaussian':
            prf_mask = self.nsp.utils.css_gaussian_cut(dim, x, y, prf_size * (dim / 8.4))
        elif type == 'outline':
            x = y = ((dim + 2)/2)
            x_deg = y_deg = prf_angle = prf_ecc = prf_expt = 0
            deg_radius = prf_size = ecc_max
            prf_mask = (self.nsp.utils.make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), ecc_max * (dim / 8.4), fill = fill_outline))
        else:
            raise ValueError(f"Invalid type: {type}. Available mask types are 'gaussian','circle','full_gaussian','cut_gaussian', and 'outline'.")
        
        # Convert pixel indices to degrees of visual angle
        x_deg = (x - ((dim + 2) / 2)) * degrees_per_pixel
        y_deg = (((dim + 2) / 2) - y) * degrees_per_pixel
        
        central_patch = 0
        if add_central_patch:
            central_patch = self.nsp.utils.make_circle_mask(dim, ((dim+2)/2), ((dim+2)/2), patch_radius * (dim / 8.4), fill = 'n')

        if plot == 'y':
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow((prf_mask + central_patch), cmap='bone', origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
            ax.set_title(f'Region Of Interest: {roi}\n'
                        f'Voxel: [{x_vox}, {y_vox}, {z_vox}]\n'
                        f'pRF x,y,σ: {round(x_deg, 1), round(y_deg, 1), round(deg_radius, 1)}\n'
                        f'Angle: {round(prf_angle, 2)}°\nEccentricity: {round(prf_ecc, 2)}°\n'
                        f'Exponent: {round(prf_expt, 2)}\nSize: {round(prf_size, 2)}°\n'
                        f'Explained pRF variance (R2): {round(prf_rsq, 2)}%\n'
                        f'pRF proportion inside central {2 * ecc_max}° patch: {round(prop_in_patch, 2)}%\n'
                        f'NSD R-squared: {round(nsdR2, 2)}%\n'
                        f'pRF R-squared: {round(prf_rsq, 2)}%\n'
                        f'pRF Gain: {round((prf_gain / prf_meanvol) *100, 2)}% BOLD\n')
            ax.set_xlabel('Horizontal Degrees of Visual Angle')
            ax.set_ylabel('Vertical Degrees of Visual Angle')

            # Set ticks at every 0.1 step
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            
            if grid == 'y':
                ax.grid(which='both', linestyle='--', linewidth=0.5, color='black')

        # Create a dictionary to store the output values
        prf_output_dict = {
            'mask': prf_mask,
            'x': x,
            'y': y,
            'pix_radius': pix_radius,
            'deg_radius': deg_radius, 
            'iterations': iter,
            'x_vox': x_vox,
            'y_vox': y_vox,
            'z_vox': z_vox,
            'x_deg': x_deg,
            'y_deg': y_deg,
            'angle': prf_angle,
            'eccentricity': prf_ecc,
            'exponent': prf_expt,
            'size': prf_size,
            'R2': prf_rsq,
            'central_overlap': prop_in_patch,
            'peri_center': peripheral_center,
            'bin_prf': bin_prf,
            'region_patch': region_patch
        }
        return prf_output_dict
    
    # # This class is to make sure that the heatmap can still be plotted if all pRF
    # # options have been considered.
    class AllPRFConsidered(Exception):
        pass
    def prf_heatmap(self, n_prfs, binary_masks, prf_proc_dict, dim=425, mask_type='gaussian', cmap='gist_heat', 
                    roi='V2', sigma_min=1, sigma_max=25, ecc_min = 0, ecc_max = 4.2, angle_min = 0, angle_max = 360, 
                    peripheral_center = None, print_prog='n', excl_reason = 'n', subjects='all',
                    outline_degs = None, filter_dict = None, fill_outline = 'n', plot_heat = 'y', 
                    ecc_strict = None, grid = 'n', min_overlap = 100, patch_radius = 1, peri_info:bool = False, peri_angle_ecc = None):
        
        # Create new dictionary to store the filtered voxels that pass the pRF requirements imposed
        prfmask_dict = copy.deepcopy(binary_masks)
        
        if outline_degs == None:
            outline_degs = 0
            
        outline_surface = np.pi * outline_degs**2
        prf_sumstack = []
        prf_sizes = []
        prf_overlaps = []
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
                roi_flt = np.sum(binary_masks[subject][f'{roi}_mask']).astype('int') # This is the total number of voxels for subj, roi
                prf_vec = random.sample(range(roi_flt), roi_flt) # Create random vector to shuffle order voxels to consider
                # prf_vec = random.sample(range(np.sum(roi_flt)), np.sum(roi_flt)) # Idem dito as in the 'if' part
                
            # FIX THIS STILL!!! ??
            if n_prfs == 'all':
                n_prfs_subject = np.sum(binary_masks[subject][f'{roi}_mask']) # This does not work. I think it does now
                # n_prfs_subject = random.randint(10,20)
            else:
                n_prfs_subject = n_prfs

            # Create an empty array to fill with the masks
            prf_single = np.zeros([dim, dim, n_prfs_subject])
            
            # Set the filtered dictionary values to zero
            prfmask_dict[subject][f'{roi}_mask'] = np.zeros(binary_masks[subject][f'{roi}_mask'].shape)
            
            iter = 0
            end_premat = False
            for prf in range(n_prfs_subject):
                try:
                    # prf_single[:, :, prf], _, _, _, new_iter = get_mask(dim=dim,
                    prf_dict = self.get_mask(dim=dim,
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
                                        ecc_min = ecc_min,
                                        ecc_max = ecc_max,
                                        angle_min = angle_min,
                                        angle_max = angle_max,
                                        excl_reason=excl_reason,
                                        filter_dict = filter_dict,
                                        ecc_strict = ecc_strict,
                                        grid = grid,
                                        min_overlap = min_overlap,
                                        peripheral_center = peripheral_center,
                                        patch_radius = patch_radius,
                                        peri_info = peri_info,
                                        peri_angle_ecc = peri_angle_ecc)
                    
                    prf_single[:, :, prf] = prf_dict['mask']
                    iter = prf_dict['iterations']
                    prf_size = prf_dict['size']
                    prf_sizes.append(prf_size)
                    prf_overlap = prf_dict['central_overlap']
                    prf_overlaps.append(prf_overlap)
                    
                    prfmask_dict[subject][f'{roi}_mask'][prf_dict['x_vox']][prf_dict['y_vox']][prf_dict['z_vox']] = 1
                    
                    if print_prog == 'y':
                        print(f"Subject: {subject}, Voxel {prf+1} out of {n_prfs_subject} found")
                        if (prf+1) == n_prfs_subject:
                            print('\n')
                except self.AllPRFConsidered:
                    if prf >= n_prfs_subject:
                        print(f'All potential pRFs have been considered at least once.\n'
                            f'Total amount of pRFs found: {len(prf_sizes)}')
                        end_premat = True
                        
                    break  # Exit the loop immediately
            
            prf_sumstack.append(np.mean(prf_single, axis=2))
            total_prfs_found = len(prf_sizes)
            print(f'Currently {total_prfs_found} prfs found')
            
        avg_prf_surface = np.pi * np.mean(prf_sizes)**2
        relative_surface = round(((avg_prf_surface / outline_surface) * 100), 2)
        # Combine heatmaps of all subjects
        prf_sum_all_subjects = np.mean(np.array(prf_sumstack), axis=0)
        
        dim = prf_dict['mask'].shape[0]
        
        middle_xy = (((dim + 1) / 2), ((dim + 1) / 2))
        
        if peripheral_center == None and peri_angle_ecc == None:
            center_x = center_y = middle_xy[0]
        else:
            peripheral_center = prf_dict['peri_center']
            center_x = middle_xy[0] + (peripheral_center[0] * dim/8.4)
            center_y = middle_xy[1] - (peripheral_center[1] * dim/8.4) # This is reversed because that's how they do it
            # in the NSD documentation. It has to do with where the y-axis starts, which is ('upper') in this case.
            # To verify this I checked the angle coordinates
            
        outline = self.nsp.utils.make_circle_mask(dim, center_x, center_y, outline_degs * dim/8.4, fill=fill_outline)
        
        # Create a circle outline if an array is provide in the outline argument (should be same dimensions, binary)
        prf_sum_all_subjects += (np.max(prf_sum_all_subjects) * outline) if outline_degs is not None else 1

        # Display the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(prf_sum_all_subjects, cmap=cmap, origin='upper', extent=[-4.2, 4.2, -4.2, 4.2])
        ax.set_title(f'Region Of Interest: {roi}\n'
                    f'Spatial restriction of central {2 * patch_radius}° visual angle\n'
                    f'Average pRF radius: {round(np.mean(prf_sizes), 2)}°, {relative_surface}% of outline surface\n'
                    f'Total amount of pRFs found: {total_prfs_found}\n'
                    f'Average pRF overlap with central patch: {round(np.mean(prf_overlaps), 2)}%')
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

        return prf_sum_all_subjects, iter, end_premat, roi, prf_sizes, relative_surface, total_prfs_found, prfmask_dict, prf_overlaps

    def compare_heatmaps(self, n_prfs, binary_masks=None, prf_proc_dict=None, filter_dict=None, basis='roi',
                        mask_type='cut_gaussian', cmap='CMRmap', roi='V1', excl_reason='n', sigma_min=0,
                        sigma_max=4.2, ecc_min=0, ecc_max=2, angle_min=0, angle_max=360, peripheral_center=None,
                        print_prog='n', outline_degs=None, fill_outline='n', ecc_strict=None, grid='n', min_overlap=100,
                        patch_radius=1, plotname='prf_heatmaps.png', peri_info:bool=False, peri_angle_ecc=None):
        if basis == 'roi':
            rois = sorted(prf_proc_dict['subj01']['proc'].keys())

        prfmask_dict_all = copy.deepcopy(binary_masks)

        def plot_mask(ax, mask, title, subtitle='aars', last=None, extent=[-4.2, 4.2, -4.2, 4.2]):
            img = ax.imshow(mask, cmap=cmap, extent=extent)
            ax.set_title(title, fontsize = 13, weight = 'semibold')
            
            if subtitle:
                ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=12)  # Adjust fontsize as needed
            
            ax.set_xlabel('Horizontal Degrees of Visual Angle')
            ax.set_ylabel('Vertical Degrees of Visual Angle')
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_major_locator(MultipleLocator(2))

        fig, axs = plt.subplots(1, 4, figsize=(18, 5))
        heatmaps = []
        for n, roi in enumerate(rois):
                    
            heatmap, _, _, roi, prf_sizes, rel_surf, _, prfmask_dict_all, prf_overlaps = self.prf_heatmap(
                                        n_prfs, binary_masks=prfmask_dict_all, prf_proc_dict=prf_proc_dict,
                                        mask_type=mask_type, cmap=cmap, roi=roi[:2], excl_reason=excl_reason,
                                        sigma_min=sigma_min, sigma_max=sigma_max, ecc_min=ecc_min, ecc_max=ecc_max,
                                        angle_min=angle_min, angle_max=angle_max, peripheral_center=peripheral_center,
                                        print_prog=print_prog, subjects='all', outline_degs=outline_degs,
                                        filter_dict=filter_dict, fill_outline=fill_outline, plot_heat='n',
                                        ecc_strict=ecc_strict, grid=grid, min_overlap=min_overlap, patch_radius=patch_radius, 
                                        peri_info=peri_info, peri_angle_ecc=peri_angle_ecc)

            heatmaps.append(heatmap)

            last_plot = 'y' if n == (len(rois) - 1) else 'n'
            plot_mask(axs[n], heatmap, f'{roi}\n\n\n\n', subtitle = (f'Average pRF radius: {round(np.mean(prf_sizes), 2)}°,\n'
                    f'{rel_surf}% of outline surface\n total pRFs found: {len(prf_sizes)}\n'
                    f'Average overlap with central patch: {round(np.mean(prf_overlaps), 2)}%'), last=last_plot)
        
        plt.tight_layout()
        plt.show()
        plt.savefig(plotname)
        
        return prfmask_dict_all, heatmaps
            
    # Function that does the same but it plots them differently and removes headers so it can be used in documents.
    def compare_heatmaps_clean(self, n_prfs, binary_masks=None, prf_proc_dict=None, filter_dict=None, basis='roi',
                        mask_type='cut_gaussian', cmap='CMRmap', roi='V1', excl_reason='n', sigma_min=0,
                        sigma_max=4.2, ecc_min =0, ecc_max=2, angle_min=0, angle_max = 360, peripheral_center = None,
                        print_prog='n', outline_degs=None, fill_outline='n', ecc_strict=None, grid='n', min_overlap = 100, 
                        patch_radius = 1, peri_info:bool = False, peri_angle_ecc = None, plotname = 'prf_heatmaps.png', 
                        save:bool=False):
        if basis == 'roi':
            rois = sorted(prf_proc_dict['subj01']['proc'].keys())

        # Create a new dictionary to store the masks for all the pRFs that pass the requirements
        prfmask_dict_all = copy.deepcopy(binary_masks)

        def plot_mask(ax, mask, title, subtitle='', last=None, extent=[-4.2, 4.2, -4.2, 4.2]):
            img = ax.imshow(mask, cmap=cmap, extent=extent)
            ax.set_title(title, fontsize = 13, weight = 'semibold')
            
            if subtitle:
                ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=12)  # Adjust fontsize as needed
            
            ax.set_xlabel('Horizontal Degrees of Visual Angle')
            ax.set_ylabel('Vertical Degrees of Visual Angle')
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_major_locator(MultipleLocator(2))

        _, axs = plt.subplots(2, 2, figsize=(11, 11))
        heatmaps = []
        for n, roi in enumerate(rois):
            heatmap, _, _, roi, prf_sizes, rel_surf, _, prfmask_dict_all, _ = self.prf_heatmap(
                                        n_prfs, binary_masks=prfmask_dict_all, prf_proc_dict=prf_proc_dict,
                                        mask_type=mask_type, cmap=cmap, roi=roi[:2], excl_reason=excl_reason,
                                        sigma_min=sigma_min, sigma_max=sigma_max, ecc_min = ecc_min, ecc_max=ecc_max,
                                        angle_min = 0, angle_max = 360, peripheral_center = peripheral_center,
                                        print_prog=print_prog, subjects='all', outline_degs=outline_degs,
                                        filter_dict=filter_dict, fill_outline=fill_outline, plot_heat='n',
                                        ecc_strict=ecc_strict, grid=grid, min_overlap = min_overlap, 
                                        patch_radius = patch_radius, peri_info = peri_info, peri_angle_ecc = peri_angle_ecc)

            heatmaps.append(heatmap)
            
            last_plot = 'y' if n == (len(rois) - 1) else 'n'
            plot_mask(axs[n//2, n%2], heatmap, f'{roi}\n\n\n\n', f'Average pRF radius: {round(np.mean(prf_sizes), 2)}°,\n {rel_surf}% of outline surface\n total pRFs found: {len(prf_sizes)}\n', last=last_plot)

        plt.tight_layout()
        plt.show()
        if save: plt.savefig(plotname)
        
        return prfmask_dict_all, heatmaps
    

    def viscortex_plot(self, prf_dict, vismask_dict, plot_param, subject, distinct_roi_colours:bool = True, inv_colour:bool = False, cmap = 'hot',
                    lowcap = None, upcap = None):

        mask_viscortex = np.zeros((vismask_dict[subject]['V1_mask'].shape))
        
        # Loop over all rois to create a mask of them conjoined
        for roi_factor, roi in enumerate(vismask_dict[subject].keys()):
            if distinct_roi_colours:
                roi_facor = 1
            mask_viscortex += (self.nsp.utils.cap_values(vismask_dict[subject][roi], lower_threshold = lowcap, upper_threshold = upcap) * ((roi_factor + 1)))

        mask_flat = self.nsp.utils.numpy2coords(mask_viscortex, keep_vals = True)

        if plot_param == 'nsdR2':
            R2_dict = self.nsd_R2_dict(vismask_dict, glm_type = 'hrf')
            brain = self.nsp.utils.cap_values(np.nan_to_num(R2_dict[subject]['full_R2']['R2_ar']), lower_threshold = lowcap, upper_threshold = upcap)
        else:
            brain = self.nsp.utils.cap_values(np.nan_to_num(prf_dict[subject]['nsd_dat'][plot_param]['prf_ar']), lower_threshold = lowcap, upper_threshold = upcap)
        brain_flat = self.nsp.utils.numpy2coords(brain, keep_vals = True)

        comrows = self.nsp.utils.find_common_rows(brain_flat, mask_flat, keep_vals = True)

        # slice_flt = cap_values(coords2numpy(coordinates = comrows, shape = brain.shape, keep_vals = True), threshold = 4)
        slice_flt = self.nsp.utils.coords2numpy(coordinates = comrows, shape = brain.shape, keep_vals = True)

        # Create sliders for each dimension
        z_slider = widgets.IntSlider(min=0, max=slice_flt.shape[2]-1, description='saggital')
        x_slider = widgets.IntSlider(min=0, max=slice_flt.shape[0]-1, description='horizontal:')
        # y_slider = widgets.IntSlider(min=0, max=slice_flt.shape[1]-1, description='coronal')
        y_slider = widgets.IntSlider(min=6, max=40, description='coronal')

        if inv_colour:
            rev = '_r'
        else:
            rev = ''

        # Function to update the plot
        def _update_plot(x, y, z):
            # Create a new figure with 2 subplots
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # Adjust the size as needed

            # Plot the first image on the left subplot
            img1 = np.rot90(mask_viscortex[:x, y, :z])
            im1 = axs[0].imshow(img1, cmap=f'{cmap}{rev}', vmin=np.min(mask_viscortex), vmax=np.max(mask_viscortex))
            axs[0].axis('off')
            cbar1 = fig.colorbar(im1, ax=axs[0])
            cbar1.set_ticks([1,2,3,4])  # Set tick positions
            cbar1.set_ticklabels(['V1', 'V2', 'V3', 'V4'])  # Set tick labels

            # Plot the second image on the right subplot
            img2 = np.rot90(slice_flt[:x, y, :z])
            im2 = axs[1].imshow(img2, cmap=f'{cmap}{rev}', vmin=np.min(brain), vmax=np.max(brain))
            axs[1].set_title(f'{plot_param} across visual cortex')
            axs[1].axis('off')
            fig.colorbar(im2, ax=axs[1])

            plt.show()
            
        widgets.interact(_update_plot, x=slice_flt.shape[0]-1, y=y_slider, z=slice_flt.shape[2]-1)
        
        
    def plot_prfs(self, voxelsieve: VoxelSieve, cmap='bone') -> None:
        print(f'There are {np.sum(voxelsieve.vox_pick)} voxels left')

        dims = np.repeat(np.array(425), np.sum(voxelsieve.vox_pick))

        prfs = np.sum(self.nsp.utils.css_gaussian_cut(dims, voxelsieve.xcoor.reshape(-1,1)[voxelsieve.vox_pick], voxelsieve.ycoor.reshape(-1,1)[voxelsieve.vox_pick], voxelsieve.size.reshape(-1,1)[voxelsieve.vox_pick] * (425 / 8.4)),axis=0)
        central_patch = np.max(prfs) * self.nsp.utils.make_circle_mask(dims[0], ((dims[0]+2)/2), ((dims[0]+2)/2), voxelsieve.patchbound * (dims[0] / 8.4), fill = 'n')

        _, ax = plt.subplots(figsize=(8,8))
        
        ax.imshow(prfs+central_patch, cmap=cmap, extent=[-4.2, 4.2, -4.2, 4.2])
        
        # Set major ticks every 2 and minor ticks every 0.1
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

        # Hide minor tick labels
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        plt.show()
        
class Stimuli():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    # Function to show a randomly selected image of the nsd dataset
    def show_stim(self, img_no='random', small:bool=False, hide:bool=False, crop:bool=False):
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
            if crop: test_image = img_brick_dataset[image_no][163:263,163:263]
            else: test_image = img_brick_dataset[image_no]
        hor = ver = 10
        if small:
            hor = ver = 5        
        if hide is not True:
            plt.figure(figsize=(hor, ver))
            plt.imshow(test_image)
            plt.title(f'Image number {image_no}')
            plt.axis('off')
            plt.show()
            
        return test_image, image_no

    def get_rms_contrast_lab(self, rgb_image:np.ndarray, mask_w_in:np.ndarray, rf_mask_in:np.ndarray, 
                            normalise:bool=True, plot:bool=False, cmap:str='gist_gray', 
                            crop_post:bool=False) -> float:
        """"
        Function that calculates Root Mean Square (RMS) contrast after converting RGB to LAB, 
        which follows the CIELAB colour space. This aligns better with how visual input is
        processed in human visual cortex.

        Arguments:
            rgb_image (np.ndarray): Input RGB image
            mask_w_in (np.ndarray): Weighted mask
            rf_mask_in (np.ndarray): RF mask
            normalise (bool): If True, normalise the input array
            plot (bool): If True, plot the square contrast and weighted square contrast
            cmap (str): Matplotlib colourmap for the plot
            crop_post (bool): If True, crop the image after calculation (to enable comparison of
                RMS values to images cropped prior to calculation)

        Returns:
            float: Root Mean Square visual contrast of input img
        """
        # Convert RGB image to LAB colour space
        lab_image = color.rgb2lab(rgb_image)
        
        # First channel [0] is Luminance, second [1] is green-red, third [2] is blue-yellow
        ar_in = lab_image[:, :, 0] # Extract the L channel for luminance values, assign to input array
            
        if normalise:
            ar_in /= ar_in.max()
        
        square_contrast = np.square(ar_in - ar_in[rf_mask_in].mean())
        msquare_contrast = (mask_w_in * square_contrast).sum()
        
        if crop_post:     
            x_min, x_max, y_min, y_max = self.nsp.utils.get_bounding_box(rf_mask_in)
            square_contrast = square_contrast[x_min:x_max, y_min:y_max]
            mask_w_in = mask_w_in[x_min:x_max, y_min:y_max]
        
        if plot:
            _, axs = plt.subplots(1, 2, figsize=(10, 5))
            plt.subplots_adjust(wspace=0.01)
            axs[1].set_title(f'RMS = {np.sqrt(msquare_contrast):.2f}')
            axs[0].imshow(square_contrast, cmap=cmap)
            axs[0].axis('off') 
            axs[1].imshow(mask_w_in * square_contrast, cmap=cmap)
            axs[1].axis('off') 
            
        return np.sqrt(msquare_contrast)
    
    # Function to get the visual contrast features and predictability estimates
    # IMPROVE: make sure that it also works for all subjects later on. Take subject arg, clean up paths.
    def features(self):
        feature_paths = [
            './data/custom_files/all_visfeats_rms.pkl',
            './data/custom_files/all_visfeats_rms_crop_prior.pkl',
            '/home/rfpred/data/custom_files/all_visfeats_scce.pkl',
            '/home/rfpred/data/custom_files/all_visfeats_scce_large.pkl',
            '/home/rfpred/data/custom_files/subj01/pred/all_predestims.h5'
        ]
        return {os.path.basename(file): self.nsp.datafetch.fetch_file(file) for file in feature_paths}
    
    def baseline_feats(self, feat_type:str):
        """
        Input options:
        - 'rms' for Root Mean Square
        - 'ce' for Contrast Energy (ce_l) for larger pooling region (5 instead of 1 degree radius)
        - 'sc' for Spatial Coherence (sc_l) for larger pooling region (5 instead of 1 degree radius)
        """
        if feat_type == 'rms':
            file_name = 'all_visfeats_rms_crop_prior.pkl'
            category = feat_type
            key = 'rms_z'
        elif feat_type == 'sc':
            file_name = 'all_visfeats_scce.pkl'
            category = 'scce'
            key = 'sc_z'
        elif feat_type == 'ce':
            file_name = 'all_visfeats_scce.pkl'
            category = 'scce'
            key = 'ce_z'
        elif feat_type[-1:] == 'l':
            file_name = 'all_visfeats_scce_large.pkl'
            category = 'scce'
            key = 'ce_z' if 'ce' in feat_type else 'sc_z'
        else:
            raise ValueError(f"Unknown feature type: {feat_type}")

        X = np.array(self.nsp.stimuli.features()[file_name]['subj01'][category][key]).reshape(-1,1)
        return X
        
    def unpred_feats(self, content:bool, style:bool, ssim:bool, pixel_loss:bool, L1:bool, MSE:bool, verbose:bool):
        """
        Function to create an X matrix based on the exclusion criteria defined in the arguments.
        Input:
        - content: boolean, whether to include content loss features
        - style: boolean, whether to include style loss features
        - ssim: boolean, whether to include structural similarity features
        - pixel_loss: boolean, whether to include pixel loss features
        - L1: boolean, whether to include L1 features
        - MSE: boolean, whether to include MSE or L2 features
        Output:
        - X: np.array, the X matrix based on the exclusion criteria
        """
        predfeatnames = [name for name in list(self.nsp.stimuli.features()['all_predestims.h5'].keys()) if name != 'img_ices']
        
        if not content:
            predfeatnames = [name for name in predfeatnames if 'content' not in name]
        if not style:
            predfeatnames = [name for name in predfeatnames if 'style' not in name]
        if not ssim:
            predfeatnames = [name for name in predfeatnames if 'ssim' not in name]
        if not pixel_loss:
            predfeatnames = [name for name in predfeatnames if 'pixel_loss' not in name]
        if not L1:
            predfeatnames = [name for name in predfeatnames if 'L1' not in name]
        if not MSE:
            predfeatnames = [name for name in predfeatnames if 'MSE' not in name]
        
        data = {name: self.nsp.stimuli.features()['all_predestims.h5'][name] for name in predfeatnames}
        
        # Convert the dictionary values to a list of lists
        data_list = list(data.values())
        
        # Convert the list of lists to a 2D array
        X = np.array(data_list)

        # Transpose the array so that each row corresponds to a sample and each column corresponds to a feature
        X = X.T[:,:]
        
        if verbose:
            print(predfeatnames)
        
        return X
    
        
    def unet_featmaps(self, list_layers:list):
        """
        Load in the UNet extracted feature maps
        Input:
        - list_layers: list with values between 1 and 4 to indicate which layers to include
        """
        # Load all the matrices and store them in a list
        matrices = [np.load(f'/home/rfpred/data/custom_files/subj01/pred/featmaps/Aunet_gt_feats_{layer}.npy') for layer in list_layers]

        # Horizontally stack the matrices
        Xcnn_stack = np.hstack(matrices)

        return Xcnn_stack
    
    def plot_unet_feats(self, layer:int, batch:int, cmap:str='bone', subject:str='subj01'):
        """
        Function to plot a selection of feature maps extracted from the U-Net class.
        Input:
        - layer: integer to select layer
        - batch: integer to select batch
        - cmap: string to define the matplotlib colour map used to plot the feature maps
        - subject: string to select the subject
        """
        with open(f'/home/rfpred/data/custom_files/{subject}/pred/featmaps/feats_gt_np_{batch}.pkl', 'rb') as f:
            feats_gt_np = pickle.load(f)
            
        # Get the number of feature maps
        num_feature_maps = feats_gt_np[0].shape[1]

        # Calculate the number of rows and columns for the subplots
        num_cols = int(np.ceil(np.sqrt(num_feature_maps)))
        num_rows = int(np.ceil(num_feature_maps / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

        # Flatten the axes array to make it easier to loop over
        axes = axes.flatten()

        # Loop over the feature maps and plot each one
        for i in range(num_feature_maps):
            # first index is the CNN layer, then, the [image, featmap no, dimx, dimy]
            axes[i].imshow(feats_gt_np[layer][random.randint(0, 500), i, :, :], cmap=cmap)
            axes[i].axis('off')  # Hide axes

        # If there are more subplots than feature maps, hide the extra subplots
        for j in range(num_feature_maps, len(axes)):
            axes[j].axis('off')

        plt.show()
        
    def alex_featmaps(self, layers:list, pcs_per_layer:Union[int, str]='all', subject:str='subj01'):
        """
        Load in the feature maps from the AlexNet model for a specific layer and subject
        Input:
        layers: list of integers representing the layers of the AlexNet model to include in the X-matrix
        pcs_per_layer: integer value indicating the top amount of principal components to which the feature map should be reduced to, or 
            'all' if all components should be included.
        - subject: string value representing the subject for which the feature maps should be loaded in
        """
        # Load in the feature maps extracted by the AlexNet model
        X_all = []
        
        if isinstance(pcs_per_layer, int):
            cut_off = pcs_per_layer
        
        for n_layer, layer in enumerate(layers):
            this_X = np.load(f'/home/rfpred/data/custom_files/subj01/center_strict/alex_lay{layer}.npy')
            if n_layer == 0:
                if pcs_per_layer == 'all':
                    cut_off = this_X.shape[0]
                X_all = this_X[:, :cut_off]
            else: X_all = np.hstack((X_all, this_X[:, :cut_off]))
        
        return X_all    
    
    # Create design matrix containing ordered indices of stimulus presentation per subject
    def imgs_designmx(self):
        
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
    
    # Get random design matrix to test other fuctions
    def random_designmx(idx_min = 0, idx_max = 40, n_img = 20):
        
        subjects = os.listdir('/home/rfpred/data/natural-scenes-dataset/nsddata/ppdata')
        
        # Create design matrix for the subject-specific stimulus presentation order
        stims_design_mx = {}
        for subject in sorted(subjects):
            # Generate 20 random integer values between 0 and 40
            stim_list = np.random.randint(idx_min, idx_max, n_img)
            stims_design_mx[subject] = stim_list
        
        return stims_design_mx
    
    # Plot a correlation matrix for specific loss value estimations of unpredictability estimates
    def unpred_corrmatrix(self, subject='subj01', type:str='content', loss_calc:str='MSE', cmap:str='copper_r'):
        """
        Plot a correlation matrix for specific loss value estimations of unpredictability estimates.

        Parameters:
        subject (str): The subject for which to plot the correlation matrix. Default is 'subj01'.
        type (str): The type of loss value estimations to include in the correlation matrix. Default is 'content'.
        loss_calc (str): The type of loss calculation to use. Default is 'MSE'.
        cmap (str): The colormap to use for the heatmap. Default is 'copper_r'.
        """
        predfeatnames = [name for name in list(self.features()['all_predestims.h5'].keys()) if name.endswith(loss_calc) and name.startswith(type)]

        # Build dataframe
        data = {name: self.features()['all_predestims.h5'][name] for name in predfeatnames}
        df = pd.DataFrame(data)

        # Compute correlation matrix
        corr_matrix = df.corr()
        ticks = [f'Layer {name.split("_")[2]}' for name in predfeatnames]
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks)
        plt.title(f'U-Net unpredictability estimates\n{type} loss {loss_calc} correlation matrix')
        plt.show()
        
    def plot_correlation_matrix(self, include_rms:bool=True, include_ce:bool=True, include_ce_l:bool=True, include_sc:bool=True, include_sc_l:bool=True, cmap:str='copper_r'): 
        """
        Plot a correlation matrix for the MSE content loss values per layer, and the baseline features.

        Parameters:
        include_rms (bool): If True, include the 'rms' column in the correlation matrix.
        include_ce (bool): If True, include the 'ce' column in the correlation matrix.
        include_ce_l (bool): If True, include the 'ce_l' column in the correlation matrix.
        include_sc (bool): If True, include the 'sc' column in the correlation matrix.
        include_sc_l (bool): If True, include the 'sc_l' column in the correlation matrix.
        """
        predfeatnames = [name for name in list(self.features()['all_predestims.h5'].keys()) if name.endswith('MSE') and name.startswith('content')]

        # Build dataframe
        data = {name: self.features()['all_predestims.h5'][name] for name in predfeatnames}
        if include_rms:
            data['rms'] = self.baseline_feats('rms').flatten()
        if include_ce:
            data['ce'] = self.baseline_feats('ce').flatten()
        if include_ce_l:
            data['ce_l'] = self.baseline_feats('ce_l').flatten()
        if include_sc:
            data['sc'] = self.baseline_feats('sc').flatten()
        if include_sc_l:
            data['sc_l'] = self.baseline_feats('sc_l').flatten()

        df = pd.DataFrame(data)

        # Compute correlation matrix
        corr_matrix = df.corr()
        ticks = [f'Pred {int(name.split("_")[2])+1}' for name in predfeatnames]
        if include_rms:
            ticks.append('RMS 1°')
        if include_ce:
            ticks.append('CE 1°')
        if include_ce_l:
            ticks.append('CE 5°')
        if include_sc:
            ticks.append('SC 1°')
        if include_sc_l:
            ticks.append('SC 5°')
        plt.figure(figsize=(9,7))
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, xticklabels=ticks, yticklabels=ticks)
        plt.title(f'Correlation matrix for the MSE content loss values per\nlayer, and the baseline features')
        plt.show()
            
        
    def extract_features(self, subject:str='subj01', layer:int=4, start_img:int=0, n_imgs:int=1,
                         batch_size:int=10, pca_components=10, verbose:bool=False, img_crop:bool=True):
        # Load the pretrained AlexNet model
        model = models.alexnet(pretrained=True)
        model.eval() # Set model to evaluation mode, as it's pretrained and we'll use it for feature extraction
        
        class ImageDataset(Dataset):
            def __init__(self, supclass, image_ids, transform=None):
                self.supclass = supclass
                self.image_ids = image_ids
                self.transform = transform

            def __len__(self):
                return len(self.image_ids)

            def __getitem__(self, idx):
                img_id = self.image_ids[idx]
                if img_crop: imgnp = (self.supclass.show_stim(img_no=img_id, hide=True, small=True)[0][163:263,163:263])
                else: imgnp = self.supclass.show_stim(img_no=img_id, hide=True, small=True)[0]
                imgPIL = Image.fromarray(imgnp) # Convert into PIL from np

                if self.transform:
                    imgPIL = self.transform(imgPIL)

                return imgPIL
                
        preprocess = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x24 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])
        
        layer_names = self.nsp.utils.get_layer_names(model) # Get the layer names
        train_nodes, _ = get_graph_node_names(model) # Get the node names
        if verbose:
            print(layer_names)
            print(train_nodes)
        this_layer = train_nodes[layer]
        this_layer_name = layer_names[layer]
        
        feature_extractor = create_feature_extractor(model, return_nodes=[this_layer])

        image_ids = self.imgs_designmx()[subject][start_img:start_img+n_imgs]
        dataset = ImageDataset(self, image_ids, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        def _fit_pca(feature_extractor, dataloader):
            # Define PCA parameters
            pca = IncrementalPCA(n_components=pca_components, batch_size=batch_size)

            while True:  # Keep trying until successful
                try:
                    # Fit PCA to batch
                    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                        # Extract features
                        ft = feature_extractor(d)
                        # Flatten the features
                        ft_flat = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
                        # Fit PCA to batch
                        pca.partial_fit(ft_flat.detach().cpu().numpy())
                    return pca, ft  # Return the PCA object
                except Exception as e:
                    print(f"Error occurred: {e}")
                    print("Restarting PCA fitting...")
                    
        pca, feature = _fit_pca(feature_extractor, dataloader)
        
        return image_ids, dataset, pca, feature, this_layer, this_layer_name
        
    def plot_features(self, which_img:int, features, layer:str, layer_type:str, img_ids:list, num_cols=10, random_cmap:bool=False):
        
        feature_maps = features[layer][which_img].detach().numpy()
        
        # Number of feature maps
        num_maps = feature_maps.shape[0]

        # Number of rows in the subplot grid
        num_rows = num_maps // num_cols
        if num_maps % num_cols:
            num_rows += 1
            
        cmaps = list(colormaps)
        this_cmap = cmaps[random.randint(0, len(cmaps))] if random_cmap else 'bone'
        if random_cmap:
            print (f'The Lord has decided for you to peek into feature space through the lens of {this_cmap}')
        
        # Create a figure for the subplots
        if layer_type == 'input':
            figsize = (10, 3)
        else: figsize = (num_cols, num_rows)
        plt.figure(figsize=figsize)

        # Plot each feature map
        for i in range(num_maps):
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(feature_maps[i], cmap=this_cmap)
            plt.axis('off')

        # Show the plot
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.tight_layout()
        plt.show()
    
    # Function to load in a list of images and masks given a list of indices. Used to provide the right input
    # to the U-Net model. Option to give the mask location, the eccentricity of the mask, and the output format.
    # The alternative mask_loc is 'irrelevant_patch', which places the mask at a fixed location in the image.
    # However, this is not yet working, because the final evaluation is done based on a 'eval_mask' object.
    # Perhaps also add this to the function.
    # Could also add the option to select a subject so it automatically gets a specified amount of their images.

    ##### Give this a better name, and change a bit so it works for different subjects. It is not really random, but 
    # it CAN be random, because it mainly just helps provide the lists
    def rand_img_list(self, n_imgs, asPIL:bool = True, add_masks:bool = True, mask_loc = 'center', ecc_max = 1, select_ices = None):
        imgs = []
        img_nos = []
        for i in range(n_imgs):
            img_no = random.randint(0, 27999)
            if select_ices is not None: img_no = select_ices[i]
            img = self.show_stim(img_no=img_no, hide=True)[0]

            if i == 0:
                dim = img.shape[0]
                radius = ecc_max * (dim / 8.4)
                
                if mask_loc == 'center': x = y = (dim + 1)/2
                elif mask_loc == 'irrelevant_patch': x = y = radius + 10

            if asPIL: img = Image.fromarray(img)

            imgs.append(img)
            img_nos.append(img_no)
        mask = (self.nsp.utils.make_circle_mask(dim, x, y, radius, fill = 'y', margin_width = 0) == 0)

        if asPIL: mask = Image.fromarray(mask)
        masks = [mask] * n_imgs

        return imgs, masks, img_nos
        
    # Allround function to run the U-Net and create intuitive plots of the resulting predictability estimates.
    def predplot(self, subject:str = None, start_img:int = 0, n_imgs:int = 5, mask_loc:str = 'center', ecc_max:float = 1, select_ices = 'subject_based', 
                cnn_type:str = 'alex', pretrain_version:str = 'places20k', eval_mask_factor:float = 1.2, log_y_MSE:str = 'y'):
        
        # Load in the U-Net
        if pretrain_version == 'places20k':
            pretrain = 'pconv_circ-places20k.pth'
        elif pretrain_version == 'places60k':
            pretrain = 'pconv_circ-places60k-fine.pth'
        elif pretrain_version == 'original':
            pretrain = 'pretrained_pconv.pth'
        else:
            raise TypeError('Please select a valid pretrain version: places20k, places60k or original')
            
        unet=UNet(checkpoint_name = pretrain,feature_model = cnn_type)

        # What images will be processed:
        if select_ices == 'random': # A random set of images
            specific_imgs = [random.randint(0,72999) for _ in range(n_imgs)]
        # If it is a list, set specific_imgs to that list
        elif type(select_ices) == list:
            specific_imgs = select_ices
        elif select_ices == 'subject_based':
            dmx = self.imgs_designmx() # A range of images based on the subject-specific design matrix
            subj_imgs = list(dmx[subject])
            specific_imgs = subj_imgs[start_img:start_img + n_imgs]
        else: 
            raise TypeError('Please select a valid image selection method: random, subject_based or a list of specific image indices')
            
        # Get the images, masks and image numbers based on the specific image selection
        imgs, masks, img_nos = self.rand_img_list(n_imgs, asPIL = True, add_masks = True, mask_loc = mask_loc, ecc_max = ecc_max, select_ices = specific_imgs)
            
        # Get the evaluation mask based on the evaluation mask size factor argument.
        eval_mask = self.nsp.utils.scale_square_mask(~np.array(masks[0]), min_size=((eval_mask_factor/1.5)*100), scale_fact= eval_mask_factor)


        # Run the images through the U-Net and time how long it takes.
        start_time = time.time()
    
        # Run them through the U-Net
        payload_full = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = None)
        payload_crop = unet.analyse_images(imgs, masks, return_recons=True, eval_mask = eval_mask)

        end_time = time.time()

        total_time = end_time - start_time
        average_time_per_image = (total_time / n_imgs) / 2

        print(f"Average time per image: {average_time_per_image} seconds")
            
        plt.style.use('dark_background')  # Apply dark background theme

        for img_idx in range(len(imgs)):
            scene_no = img_nos[img_idx]
            titles = ['Ground Truth', 'Input Masked', 'Output Composite', '', 'Content loss values', '', 'Style loss values']
            # fig, axes = plt.subplots(2, 5, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1, 1, 2, 2]})  # Create 4 subplots
            fig, axes = plt.subplots(2, 7, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 2, 2, .4, 2.5, 1.3, 2.5]})  # Create 4 subplots
            for ax in axes[:,3]:
                ax.axis('off')
            for ax in axes[:,5]:
                ax.axis('off')

            for eval_size in range(2):
                this_payload = payload_full if eval_size == 0 else payload_crop

                for loss_type in ['content', 'style']:
                    ntype = 6 if loss_type == 'style' else 4
                    yrange = [0, 5] if loss_type == 'content' else [0, .05]
                    ylogrange = [0.1, 100] if loss_type == 'content' else [0.00001, .1]
                    n_layers = 5
                    losses = {}
                    MSE = []
                    L1 = []

                    for i in range(n_layers):
                        MSE.append(round(this_payload[f"{loss_type}_loss_{i}_MSE"][img_idx], 3))  # Get the loss for each layer
                        L1.append(round(this_payload[f"{loss_type}_loss_{i}_L1"][img_idx], 3))  # Get the loss for each layer
                        losses['MSE'] = MSE
                        losses['L1'] = L1
                    
                    # Plot the loss values
                    axes[eval_size, ntype].plot(range(1, n_layers + 1), L1, marker='o', color='crimson', linewidth=3)  # L1 loss
                    if eval_size == 0:
                        axes[eval_size, ntype].set_title(titles[ntype])
                    if eval_size == 1:
                        axes[eval_size, ntype].set_xlabel('Feature space (Alexnet layer)')
                                
                    if loss_type == 'content':
                        # Create a secondary y-axis for MSE
                        ax_mse = axes[eval_size, ntype].twinx()
                    else: 
                        ax_mse = axes[eval_size, ntype]
                    ax_mse.plot(range(1, n_layers + 1), MSE, marker='o', color='cornflowerblue', linewidth=3)  # MSE loss
                    if loss_type == 'content':
                        ax_mse.tick_params(axis='y', labelcolor='cornflowerblue', labelsize = 12)
                    
                    
                    if log_y_MSE == 'y' and loss_type == 'content':
                        ax_mse.set_yscale('log')  # Set y-axis to logarithmic scale for MSE
                        ax_mse.set_ylabel('MSE Loss (log)', color='cornflowerblue', fontsize = 14)
                        ax_mse.set_ylim(ylogrange[0], ylogrange[1])
                        ax_mse.grid(False)
                        axes[eval_size, ntype].set_ylabel('L1 Loss (linear)', color='crimson', fontsize = 14)
                        axes[eval_size, ntype].set_ylim([yrange[0], yrange[1]])  # Set the range of the y-axis for L1
                    else:
                        axes[eval_size, ntype].set_ylabel('Loss value', color='white', fontsize = 14)
                        axes[eval_size, ntype].set_ylim([yrange[0], yrange[1]])
                        
                    axes[eval_size, ntype].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
                    if loss_type == 'content':
                        axes[eval_size, ntype].tick_params(axis='y', labelcolor='crimson', labelsize = 12)
                    axes[eval_size, ntype].grid(False)
                    
                    if loss_type == 'style':
                        ax_mse.legend(['L1 (MAE)', 'MSE'], loc='upper right')
                    
                fig.suptitle(f"Image Number: {scene_no}\n\n"
                            f"Cropped eval dissimilarity stats:\n"
                            f"Structural Similarity: {round(this_payload['ssim'][img_idx],4)}\n"
                            f"Pixel loss L1 (MAE): {round((this_payload['pixel_loss_L1'][img_idx]).astype('float'), 4)}\n"
                            f"Pixel loss MSE: {round((this_payload['pixel_loss_MSE'][img_idx]).astype('float'), 4)}", fontsize=14)

                # Loop through each key in the recon_dict to plot
                for i, key in enumerate(['input_gt', 'input_masked', 'out_composite']):
                    img = this_payload['recon_dict'][key][img_idx].permute(1, 2, 0)
                    axes[eval_size, i].imshow(img)
                    if eval_size == 0:
                        axes[eval_size, i].set_title(titles[i], fontsize = 12)
                    axes[eval_size, i].axis('off')  # To keep images square and hide the axes
            
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05)  # Adjust the spacing between subplots
            plt.show()
            
        return imgs, masks, img_nos, payload_full, payload_crop
        
class Analysis():
    
    def __init__(self, NSPobj):
        self.nsp = NSPobj
        pass
    
    # A terrible function
    # Okay this one is the actual good function. The other should be deleted and never be used again. 
    def get_hrf_dict(self, subjects, voxels, prf_region='center_strict', min_size=0.1, max_size=1,
                    prf_proc_dict=None, max_voxels=None, plot_sizes='n', verbose:bool=False,
                    vismask_dict=None, minimumR2:int=100, in_perc_signal_change:bool=False):
        hrf_dict = {}
        R2_dict_hrf = self.nsp.cortex.nsd_R2_dict(vismask_dict, glm_type = 'hrf')
        
        
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
                optimal_top_n_R2 = self.nsp.cortex.optimize_rsquare(R2_dict_hrf, 'subj01','nsd', roi, minimumR2, False, 250)
                print(f'Voxels in {roi[:2]} with a minimum R2 of {minimumR2} is approximately {optimal_top_n_R2}')
                # Fetch this specific number of selected top R2 values for this roi
                highR2 = self.nsp.cortex.rsquare_selection(R2_dict_hrf, optimal_top_n_R2, n_subjects = 8, dataset = 'nsd')[subject][roi]
                # print(f'The average R2 value for {roi}') # This does not make sense, because not filtered yet.
                voxel_mask = voxels[subject][roi] # So this is not the binary mask, but the prf-selection made with the heatmap function
                
                # if max_voxels is None or n_roi > 0:
                    # vox_n_cutoff = numpy2coords(voxel_mask).shape[0]
                    
                # This if statement is to allow for a size-based selection of voxels
                if min_size is not None and max_size is not None:
                    preselect_voxels = self.nsp.utils.numpy2coords(voxel_mask, keep_vals = True) # Get the voxel coordinates based on the prf selection
                    # This is another array with coordinates on the first 3 columns and then a selected size on the 4th column
                    size_selected_voxels = self.nsp.utils.filter_array_by_size(prf_proc_dict[subject]['proc'][roi]['size'], min_size, max_size)
                    
                    joint_ar_prf = self.nsp.utils.find_common_rows(size_selected_voxels, preselect_voxels, keep_vals = True) # Keep_vals keeps the values of the first array
                    joint_ar_R2 = self.nsp.utils.find_common_rows(joint_ar_prf, highR2, keep_vals = True) # Select based on the top R2 values
                    if verbose:
                        print(f'This is joint_ar_R2 {joint_ar_R2[10:15,:]}')
                    available_voxels = joint_ar_R2.shape[0] # Check how many voxels we end up with
                    print(f'Found {available_voxels} voxels in {roi[:2]} with pRF sizes between {min_size} and {max_size}')
                    
                    selected_R2_vals = self.nsp.utils.find_common_rows(highR2, joint_ar_R2, keep_vals = True)#[:,3] # Get a list of the R2 values for the selected voxels
                    if verbose:
                        print(f'This is the final r2 vals {selected_R2_vals[10:15,:]}')

                    # Check whether the amount of voxels available is more than a potential predetermined limit
                    if max_voxels is not None and available_voxels > max_voxels:
                        
                        top_n_R2_voxels = self.nsp.utils.sort_by_column(selected_R2_vals, 3, top_n = 1000)[:max_voxels, :] # Sort the R2 values and select the top n
                        size_selected_voxels_cut = self.nsp.utils.find_common_rows(joint_ar_R2, top_n_R2_voxels, keep_vals = True) # Get the pRF sizes of these voxels
                        print(f'The amount of voxels are manually restricted to {max_voxels} out of {available_voxels}')
                    else: size_selected_voxels_cut = joint_ar_R2                
                    
                    final_R2_vals = self.nsp.utils.find_common_rows(highR2, size_selected_voxels_cut, keep_vals = True) # Get a list of the R2 values for the selected voxels
                    
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
                
                xyz_to_name_roi = np.hstack((find_vox_ar[:,:3].astype('int'), find_vox_ar[:,4].reshape(-1,1)))
                if n_roi == 0:
                    xyz_to_name = xyz_to_name_roi
                else: xyz_to_name = np.vstack((xyz_to_name, xyz_to_name_roi))
                
                # Check whether the entire fourth column is now non-zero:
                if verbose:
                    print(f'\tChecking if all selected voxels are present in beta session file: {np.all(find_vox_ar[:, 4] != 0)}\n')
                for vox_no in range(n_voxels):
                    # Get the xyz coordinates of the voxel
                    vox_xyz = find_vox_ar[vox_no, :3]
                    vox_name = find_vox_ar[vox_no, 4]
                    
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

                    if in_perc_signal_change:
                        # Calculate the z-scores for each batch
                        for i in range(betas_reshaped.shape[0]):
                            betas_mean = np.mean(betas_reshaped[i])
                            betas_normalised[i] = self.nsp.utils.get_zscore(((betas_reshaped[i] / betas_mean) * 100), print_ars='n')
                    else: 
                        betas_normalised = betas_reshaped * 300
                        for i in range(betas_reshaped.shape[0]):
                            betas_normalised[i] = self.nsp.utils.get_zscore(betas_reshaped[i], print_ars='n')
                        
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
                mean_betas[:, 3] = self.nsp.utils.get_zscore(unscaled_betas, print_ars='n').flatten()
                
                hrf_dict[subject][roi]['mean_betas'] = mean_betas # Store the mean_beta values for each voxel in the roi


                n_betas = len(hrf_dict[subject][roi][vox_name]['hrf_betas'])
                if verbose:
                    print(f'\tProcessed images: {n_betas}')
                
        plt.style.use('default')

        if plot_sizes == 'y':
            _, axs = plt.subplots(2, 2, figsize=(10, 8))  # Create a figure with 2x2 subplots
            axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
            cmap = plt.get_cmap('gist_heat')  # Get the 'viridis' color map
            for i, roi in enumerate(rois):
                sizes = hrf_dict[subject][roi]['roi_sizes'][:,3]
                color = cmap(i / len(rois))  # Get a color from the color map
                sns.histplot(sizes, kde=True, ax=axs[i], color=color, bins = 10)  # Plot on the i-th subplot
                axs[i].set_title(f'RF sizes for {roi[:2]} (n={sizes.shape[0]})')  # Include the number of voxels in the title
                axs[i].set_xlim([min_size-.1, max_size+.1])  # Set the x-axis limit from 0 to 2
              
        return hrf_dict, xyz_to_name
    

# class DataProcessor:

    # def get_hrf_dict(self, subjects, voxels, prf_region='center_strict', min_size=0.1, max_size=1,
    #                 prf_proc_dict=None, max_voxels=None, plot_sizes='n', verbose:bool=False,
    #                 vismask_dict=None, minimumR2:int=100, in_perc_signal_change:bool=False):

    #     hrf_dict = {}
    #     R2_dict_hrf = self.nsp.cortex.nsd_R2_dict(vismask_dict, glm_type='hrf')

    #     for subject in subjects:
    #         hrf_dict[subject] = {}
    #         beta_sessions = self.load_beta_sessions(subject, prf_region)
    #         rois = list(beta_sessions[0].keys())

    #         for n_roi, roi in enumerate(rois):
    #             self.process_roi(subject, roi, n_roi, beta_sessions, hrf_dict, R2_dict_hrf,
    #                              voxels, prf_proc_dict, min_size, max_size, max_voxels, minimumR2, 
    #                              verbose, in_perc_signal_change)

    #     if plot_sizes == 'y':
    #         self.plot_sizes(hrf_dict, min_size, max_size)
        
    #     return hrf_dict

    # def load_beta_sessions(self, subject, prf_region):
    #     beta_sessions = []
    #     base_path = f'/home/rfpred/data/custom_files/{subject}/{prf_region}/'
    #     for file_name in sorted(os.listdir(base_path)):
    #         if file_name.startswith("beta_dict") and file_name.endswith(".pkl"):
    #             with open(os.path.join(base_path, file_name), 'rb') as fp:
    #                 beta_sessions.append(pickle.load(fp)[subject])
    #     return beta_sessions

    # def process_roi(self, subject, roi, n_roi, beta_sessions, hrf_dict, R2_dict_hrf, voxels, 
    #                 prf_proc_dict, min_size, max_size, max_voxels, minimumR2, verbose, in_perc_signal_change):
    #     hrf_dict[subject][roi] = {}
    #     optimal_top_n_R2 = self.nsp.cortex.optimize_rsquare(R2_dict_hrf, subject, 'nsd', roi, minimumR2, False, 250)
        
    #     highR2 = self.nsp.cortex.rsquare_selection(R2_dict_hrf, optimal_top_n_R2, n_subjects=8, dataset='nsd')[subject][roi]
    #     voxel_mask = voxels[subject][roi]
        
    #     preselect_voxels = self.nsp.utils.numpy2coords(voxel_mask, keep_vals=True)
    #     size_selected_voxels = self.nsp.utils.filter_array_by_size(prf_proc_dict[subject]['proc'][roi]['size'], min_size, max_size)
    #     joint_ar_prf = self.nsp.utils.find_common_rows(size_selected_voxels, preselect_voxels, keep_vals=True)
    #     joint_ar_R2 = self.nsp.utils.find_common_rows(joint_ar_prf, highR2, keep_vals=True)
        
    #     available_voxels = joint_ar_R2.shape[0]
    #     if verbose:
    #         print(f'Found {available_voxels} voxels in {roi[:2]} with pRF sizes between {min_size} and {max_size}')
        
    #     selected_R2_vals = joint_ar_R2
    #     if max_voxels is not None and available_voxels > max_voxels:
    #         selected_R2_vals = self.nsp.utils.sort_by_column(selected_R2_vals, 3, top_n=max_voxels)
        
    #     hrf_dict[subject][roi]['roi_sizes'] = selected_R2_vals
    #     hrf_dict[subject][roi]['R2_vals'] = selected_R2_vals

    # def plot_sizes(self, hrf_dict, min_size, max_size):
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     plt.style.use('default')
    #     _, axs = plt.subplots(2, 2, figsize=(10, 8))
    #     axs = axs.flatten()
    #     cmap = plt.get_cmap('gist_heat')
    #     for i, (subject, rois) in enumerate(hrf_dict.items()):
    #         for j, (roi, data) in enumerate(rois.items()):
    #             sizes = data['roi_sizes'][:, 3]
    #             color = cmap(j / len(rois))
    #             sns.histplot(sizes, kde=True, ax=axs[j], color=color, bins=10)
    #             axs[j].set_title(f'RF sizes for {roi[:2]} (n={sizes.shape[0]})')
    #             axs[j].set_xlim([min_size - .1, max_size + .1])

    
    def load_y(self, subject:str, roi:str, hrf_dict:dict, 
           xyz_to_name:np.array, roi_masks:dict, prf_dict:dict, 
           n_voxels, start_img:int, n_imgs:int, verbose:bool=True, across_rois:bool=False):
        
        if across_rois: # Optional looping over the four different regions of interest
            rois = self.nsp.cortex.visrois_dict()[0]
            ys = []
            xyzs_stack = []
        else: rois = [roi]
        for roi in rois:
            # Check the maximum amount of voxels for this subject, roi
            max_voxels = len(hrf_dict[subject][f'{roi}_mask']['R2_vals'])
            if n_voxels == 'all': 
                n_voxels = max_voxels

            selection_xyz = np.zeros((min(max_voxels, n_voxels), 2),dtype='object')
            y_matrix = np.zeros((start_img+n_imgs-start_img, n_voxels))

            for voxel in range(n_voxels):
                if voxel < max_voxels:
                    vox_xyz, voxname = self.nsp.cortex.get_good_voxel(subject=subject, roi=roi, hrf_dict=hrf_dict, xyz_to_voxname=xyz_to_name, 
                                            pick_manually=voxel, plot=False, prf_dict=prf_dict, vismask_dict=roi_masks,selection_basis='R2')
                    selection_xyz[voxel,0] = vox_xyz
                    selection_xyz[voxel,1] = voxname
                    y_matrix[:,voxel] = hrf_dict[subject][f'{roi}_mask'][voxname]['hrf_betas_z'][start_img:start_img+n_imgs]
                else: 
                    print(f'Voxel {voxel+1} not found in {roi}, only {max_voxels} available for {roi}')
                    voxdif = n_voxels - max_voxels
                    y_matrix = y_matrix[:,:-voxdif]
                    break
            if across_rois:
                ys.append(y_matrix)
                xyzs_stack.append(selection_xyz)
                
        if across_rois:
            y_matrix = np.hstack(ys)
            selection_xyz = np.vstack(xyzs_stack)
        if verbose:
            print(f'Loaded y-matrix with {selection_xyz.shape[0]} voxels from {rois}')
        return y_matrix, selection_xyz
                                
    def run_ridge_regression(self, X:np.array, y:np.array, alpha=1.0):
        """Function to run a ridge regression model on the data.

        Args:
            X (np.array): The independent variables with shape (n_trials, n_features)
            y (np.array): The dependent variable with shape (n_trials, n_outputs)
            alpha (float, optional): Regularisation parameter of Ridge regression, larger values penalise stronger. Defaults to 1.0.

        Returns:
            sk.linear_model._ridge.Ridge: The model object
        """        
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        return model

    # Not really necessary
    def _get_coefs(self, model:sk.linear_model._ridge.Ridge):
        return model.coef_

    def _get_r(self, y:np.ndarray, y_hat:np.ndarray):
        """Function to get the correlation between the predicted and actual HRF signal betas.

        Args:
            y (np.ndarray): The original HRF signal betas from the NSD
            y_hat (np.ndarray): The predicted HRF signal betas

        Returns:
            float: The correlation between the two sets of betas as a measure of fit
        """        
        return np.mean(y * y_hat, axis=0)

    def score_model(self, X:np.ndarray, y:np.ndarray, model:sk.linear_model._ridge.Ridge, cv:int=5):
        """This function evaluates the performance of the model using cross-validation.

        Args:
            X (np.ndarray): X-matrix, independent variables with shape (n_trials, n_features)
            y (np.ndarray): y-matrix, dependent variable with shape (n_trials, n_outputs)
            model (sk.linear_model._ridge.Ridge): The ridge model to score
            cv (int, optional): The number of cross validation folds. Defaults to 5.

        Returns:
            tuple: A tuple containing:
                - y_hat (np.ndarray): The predicted values for y, with shape (n_trials, n_outputs)
                - scores (np.ndarray): The R^2 scores for each output, with shape (n_outputs,)
        """        
        # Initialize the KFold object
        kf = KFold(n_splits=cv)
        
        # Initialize lists to store the predicted values and scores for each fold
        y_hat = []
        cor_scores = []
        
        # For each fold...
        for train_index, test_index in kf.split(X):
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Fit the model on the training data
            model.fit(X_train, y_train)
            
            # Predict the values for the testing data
            y_hat_fold = model.predict(X_test)
            
            # Calculate the R^2 score for each column, no multi output
            # scores_fold = [r2_score(y_test[:, i], y_hat_fold[:, i]) for i in range(y_test.shape[1])]
            r_fold = self._get_r(y_test, y_hat_fold)
            
            # Append the predicted values and scores for this fold to the lists
            y_hat.append(y_hat_fold)
            cor_scores.append(r_fold)
        
        # Concatenate the predicted values from each fold into a single array
        y_hat = np.concatenate(y_hat)
        
        # Calculate the average R^2 score for each column
        # scores = np.mean(cor_scores, axis=0)
        
        return y_hat, cor_scores
    
    def plot_brain(self, prf_dict:dict, roi_masks:dict, subject:str, brain_numpy:np.ndarray, glass_brain=False):
        """Function to plot a 3D np.ndarray with voxel-specific values on an anatomical brain template of that subject.

        Args:
            prf_dict (dict): The pRF dictionary
            roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
            subject (str): The subject ID
            brain_numpy (np.ndarray): The 3D np.ndarray with voxel-specific values
            glass_brain (bool, optional): Optional argument to plot a glass brain instead of a static map. Defaults to False.
        """        
        brain_nii = nib.Nifti1Image(brain_numpy, self.nsp.cortex.anat_templates(prf_dict)[subject].affine)
        if glass_brain:
            plotting.plot_glass_brain(brain_nii, display_mode='ortho', colorbar=True)
        else:
            plotting.plot_stat_map(brain_nii, bg_img=self.nsp.cortex.anat_templates(prf_dict)[subject], display_mode='ortho', colorbar=True)
            
    def stat_on_brain(self, prf_dict:dict, roi_masks:dict, subject:str, stat:np.ndarray, xyzs:np.ndarray, glass_brain=False):
        """Function to create a brain plot based on a specific statistic and the corresponding voxel coordinates.

        Args:
            prf_dict (dict): The pRF dictionary
            roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
            subject (str): The subject ID
            stat (np.ndarray): The statistic to plot on the brain
            xyzs (np.ndarray): The voxel coordinates
            glass_brain (bool, optional): Optional argument to plot a glass brain instead of a static map. Defaults to False.
        """        
        n_voxels = len(xyzs)
        statmap = np.zeros((n_voxels, 4))
        for vox in range(n_voxels):
            # statmap[vox, :3] = (xyzs[vox][0][0], xyzs[vox][0][1], xyzs[vox][0][2]) # this is for the old xyzs
            statmap[vox, :3] = xyzs[vox]
            statmap[vox, 3] = stat[vox]

        brainp = self.nsp.utils.coords2numpy(statmap, roi_masks[subject]['V1_mask'].shape, keep_vals=True)
        
        self.plot_brain(prf_dict, roi_masks, subject, brainp, glass_brain)
        
    def evaluate_model(self, X, y, alpha=1.0, cv=5, extra_stats:bool=True):
        # Create and fit the model
        model = self.run_ridge_regression(X, y, alpha)

        # Get the coefficients
        coefs = self._get_coefs(model)

        # Score the model with cross-validation
        y_hat, scores = self.score_model(X, y, model, cv)

        if extra_stats:
            # Calculate the MAE, MSE, RMSE, and MAPE
            mae = mean_absolute_error(y, y_hat)
            mse = mean_squared_error(y, y_hat)
            rmse = sqrt(mse)
            mape = np.mean(np.abs((y - y_hat) / y)) * 100 # Mean abs percentage error
            
            # Return all the results
            return {
                'model': model,
                'coefficients': coefs,
                'predicted_values': y_hat,
                'cross_validation_scores': scores,
                'mean_absolute_error': mae,
                'mean_squared_error': mse,
                'root_mean_squared_error': rmse,
                'mean_absolute_percentage_error': mape
            }
        else:
            return {
                'model': model,
                'coefficients': coefs,
                'predicted_values': y_hat,
                'cross_validation_scores': scores
            }

    def plot_learning_curve(self, X, y, model=None, alpha=1.0, cv=5):
        if model is None:
            # Create and fit the model
            model = self.run_ridge_regression(X, y, alpha)

        # Initialize the KFold object
        kf = KFold(n_splits=cv)

        # Initialize a list to store the scores for each fold
        scores = []

        # For each fold...
        for train_index, test_index in kf.split(X):
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Predict the values for the testing data
            y_hat = model.predict(X_test)

            # Calculate the correlation for each column
            scores_fold = [self._get_r(y_test[:, i], y_hat[:, i]) for i in range(y_test.shape[1])]

            # Append the average correlation for this fold to the list
            scores.append(scores_fold)

        # Plot the scores
        for i, scores_fold in enumerate(scores, start=1):
            # Scatter plot of individual scores
            plt.scatter([i]*len(scores_fold), scores_fold, color='blue', alpha=0.5)

            # Line plot of mean score
            plt.plot(i, np.mean(scores_fold), color='red', marker='o')

        plt.xlabel('Fold')
        plt.ylabel('Correlation Score')
        plt.title('Learning Curve')

        # Set x-axis to only show integer values
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()
                
    def plot_residuals(self, X, y, model=None, alpha=1.0):
        """Plot the residuals of the model, which is the difference between the actual y and the predicted y (y_hat)

        Args:
            X (_type_): _description_
            y (_type_): _description_
            model (_type_, optional): _description_. Defaults to None.
            alpha (float, optional): _description_. Defaults to 1.0.
        """        
        if model is None:
            # Create and fit the model
            model = self.run_ridge_regression(X, y, alpha)

        # Get the predicted values
        y_hat = model.predict(X)

        # Calculate the residuals
        residuals = y - y_hat

        # Create a scatter plot of the predicted values and residuals
        plt.scatter(y_hat, residuals, alpha=0.3)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()    

class NatSpatPred():
    
    def __init__(self, datapath:str='/home/rfpred/data'):
        # Define the subclasses
        self.utils = None
        self.cortex = None
        self.stimuli = None
        self.datafetch = None
        self.explore = None
        self.analyse = None
        
        self.datapath = datapath
        self.subjects = sorted(os.listdir(f'{datapath}/natural-scenes-dataset/nsddata/ppdata'), key=lambda s: int(s.split('subj')[-1]))
        self.attributes = None
        self.hidden_methods = None

    # TODO: Expand this initialise in such way that it creates all the globally relevant attributes by calling on methods from the
    # nested classes
    def initialise(self):
        self.utils = Utilities()
        self.cortex = Cortex(self)
        self.stimuli = Stimuli(self)
        self.datafetch = DataFetch(self)
        self.explore = Explorations(self)
        self.analyse = Analysis(self)
        
        self.attributes = [attr for attr in dir(self) if not attr.startswith('_')] # Filter out both the 'dunder' and hidden methods
        self.attributes_unfiltered = [attr for attr in dir(self) if not attr.startswith('__')] # Filter out only the 'dunder' methods
        
        print(f'Naturalistic Spatial Prediction class: {Fore.LIGHTWHITE_EX}Initialised{Style.RESET_ALL}')
        print('\nClass contains the following attributes:')
        for attr in self.attributes:
            print(f"{Fore.BLUE} .{attr}{Style.RESET_ALL}")