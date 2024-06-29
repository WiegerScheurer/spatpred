import copy
import fnmatch
import os
import pickle
import random
import re
import sys
import time
from importlib import reload
from math import e, sqrt
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import ipywidgets as widgets
import joblib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats.mstats as mstats
import seaborn as sns
import sklearn as sk
import torch
import torchvision.models as models
import yaml
from arrow import get
from colorama import Fore, Style
from IPython.display import display
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter, MaxNLocator,
                               MultipleLocator, NullFormatter)
from nilearn import plotting
from PIL import Image
from scipy import stats
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from scipy.special import softmax
from scipy.stats import zscore as zs
from skimage import color
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)
from tqdm.notebook import tqdm

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

import lgnpy.CEandSC.lgn_statistics
from lgnpy.CEandSC.lgn_statistics import LGN, lgn_statistics, loadmat

from unet_recon.inpainting import UNet
from classes.analysis import Analysis
from classes.cortex import Cortex
from classes.datafetch import DataFetch
from classes.explorations import Explorations
from classes.stimuli import Stimuli
from classes.utilities import Utilities
from classes.voxelsieve import VoxelSieve

class NatSpatPred:

    def __init__(
        self,
        nsd_datapath: str = "/home/rfpred/data/natural-scenes-dataset",
        own_datapath: str = "/home/rfpred/data/custom_files",
    ):
        # Define the subclasses
        self.utils = None
        self.cortex = None
        self.stimuli = None
        self.datafetch = None
        self.explore = None
        self.analyse = None

        self.nsd_datapath = nsd_datapath
        self.own_datapath = own_datapath
        self.subjects = sorted(
            os.listdir(f"{nsd_datapath}/nsddata/ppdata"),
            key=lambda s: int(s.split("subj")[-1]),
        )
        self.attributes = None
        self.hidden_methods = None

    # TODO: Expand this initialise in such way that it creates all the globally relevant attributes by calling on methods from the
    # nested classes
    def initialise(self, verbose: bool = True):
        self.utils = Utilities(self)
        self.cortex = Cortex(self)
        self.stimuli = Stimuli(self)
        self.datafetch = DataFetch(self)
        self.explore = Explorations(self)
        self.analyse = Analysis(self)

        self.attributes = [
            attr for attr in dir(self) if not attr.startswith("_")
        ]  # Filter out both the 'dunder' and hidden methods
        self.attributes_unfiltered = [
            attr for attr in dir(self) if not attr.startswith("__")
        ]  # Filter out only the 'dunder' methods
        if verbose:
            print(
                f"Naturalistic Spatial Prediction class: {Fore.LIGHTWHITE_EX}Initialised{Style.RESET_ALL}"
            )
            print("\nClass contains the following attributes:")
            for attr in self.attributes:
                print(f"{Fore.BLUE} .{attr}{Style.RESET_ALL}")
