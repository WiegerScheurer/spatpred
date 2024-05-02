#!/usr/bin/env python3

import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from matplotlib.ticker import MultipleLocator
import nibabel as nib
import pickle
from typing import Dict, Union, List
import argparse
import importlib
from importlib import reload

os.environ["OMP_NUM_THREADS"] = "10" # Limit the number of processors use to 10

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

import funcs.natspatpred
importlib.reload(funcs.natspatpred)
from funcs.natspatpred import NatSpatPred

NSP = NatSpatPred()
NSP.initialise()

rois, roi_masks, viscortex_mask = NSP.cortex.visrois_dict(verbose=True)
# prf_dict = NSP.cortex.prf_dict(rois, roi_masks)

predparser = argparse.ArgumentParser(description='Get the predictability estimates for a range of images of a subject')

predparser.add_argument('start', type=int, help='The starting index of the images to get the predictability estimates for')
predparser.add_argument('n_sessions', type=int, help='The ending index of the images to get the predictability estimates for')
predparser.add_argument('subject', type=str, help='The subject to get the predictability estimates for')

args = predparser.parse_args()

def get_betas(subject:str, 
              viscortex_mask:np.ndarray, 
              roi_masks:Dict[str, np.ndarray], 
              start_session:int, 
              n_sessions:int):
    betapath = f'/home/rfpred/data/natural-scenes-dataset/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/'

    # Initialize a dictionary to hold the 2D arrays for each ROI
    # beta_dict = {roi: None for roi in roi_masks[subject].keys()}

    for session in range(start_session, start_session + n_sessions): # If start = 1 and n = 10 it goes 1 2 3 4 5 6 7 8 9 10
        print(f'Working on session: {session}')
        session_str = f'{session:02d}'
        session_data = nib.load(f"{betapath}betas_session{session_str}.nii.gz").get_fdata(caching='unchanged')

        for roi in roi_masks[subject].keys():
            print(f'Working on roi: {roi}')
            roi_mask = roi_masks[subject][roi]
            filtbet = session_data[roi_mask.astype(bool)]

            # Get the indices of the True values in the mask
            if session == 1:  # only get indices for the first session
                x, y, z = np.where(roi_mask)
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                z = z.reshape(-1, 1)
                voxbetas = np.concatenate((x, y, z, filtbet), axis=1)
            else:
                voxbetas = filtbet
            print(f'Current size of voxbetas: {voxbetas.shape}')        
                
            np.save(f'/home/rfpred/data/custom_files/subj01/betas/{roi[:2]}/beta_stack_session{session_str}.npy', voxbetas)
            print(f'Saved beta_stack_session{session_str}.npy')
        
        del session_data

get_betas(args.subject, viscortex_mask, roi_masks, args.start, args.n_sessions)