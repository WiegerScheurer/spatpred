#!/usr/bin/env python3

import os
# Limit the number of CPUs used to 2
os.environ["OMP_NUM_THREADS"] = "5"


import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
import argparse
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torch.nn import Module
from matplotlib import colormaps
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
import scipy.stats.mstats as mstats
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from tqdm import tqdm
import traceback

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


os.chdir('/home/rfpred')
sys.path.append('/home/rfpred')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

print(sys.path)

from funcs.imgproc import rand_img_list, show_stim, get_imgs_designmx


predparser = argparse.ArgumentParser(description='Get the AlexNet feature space neural representations for a range of images of a subject')

predparser.add_argument('start', type=int, help='The starting index of the images to get the cnn layer specific neural represenations for')
predparser.add_argument('end', type=int, help='The ending index of the images to get the cnn layer specific neural represenations  for')
predparser.add_argument('subject', type=str, help='The subject to get the cnn layer specific neural represenations for')
predparser.add_argument('cnn_layer', type=int, help='The layer to extract neural representations of')

args = predparser.parse_args()

print(args,'\n')

# Load the pretrained AlexNet model
model = models.alexnet(pretrained=True)
model.eval()  # Set the model to evaluation mode

class ImageDataset(Dataset):
    def __init__(self, image_ids, transform=None):
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        imgnp = (show_stim(img_no=img_id, hide='y', small = 'y')[0][163:263,163:263])
        
        imgPIL = Image.fromarray(imgnp) # Convert into PIL from np

        if self.transform:
            imgPIL = self.transform(imgPIL)

        return imgPIL
    
preprocess = transforms.Compose([
    transforms.Resize((224,224)), # resize the images to 224x24 pixels
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
])

train_nodes, _ = get_graph_node_names(model)
print(train_nodes)
this_layer = train_nodes[args.cnn_layer + 1]
# Which layer to extract the features from # Also add this as argparse thing.
# model_layer = "features.2" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}

feature_extractor = create_feature_extractor(model, return_nodes=[this_layer])

start = args.start
end = args.end
batch_size = 600
image_ids = get_imgs_designmx()[args.subject][start:end]
dataset = ImageDataset(image_ids, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def fit_pca(feature_extractor, dataloader):
    # Define PCA parameters
    pca = IncrementalPCA(n_components=600, batch_size=batch_size)

    while True:  # Keep trying until successful
        try:
            # Fit PCA to batch
            for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Extract features
                ft = feature_extractor(d)
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
                # Fit PCA to batch
                pca.partial_fit(ft.detach().cpu().numpy())
            return pca  # Return the PCA object
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Restarting PCA fitting...")

pca = fit_pca(feature_extractor, dataloader)

# Implement the code belowin the NSP.cortex class as well, to check whether it makes sense

def extract_features(feature_extractor, dataloader, pca):
    while True:  # Keep trying until successful
        try:
            features = []
            for i, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Extract features
                ft = feature_extractor(d)
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])

                # Check if the features contain NaN values
                if np.isnan(ft.detach().numpy()).any():
                    raise ValueError("NaN value detected")
                
                # Apply PCA transform
                ft = pca.transform(ft.cpu().detach().numpy())
                features.append(ft)
            return np.vstack(features)  # Return the features
        except ValueError as e:
            print(f"Error occurred: {e}")
            print("Restarting feature extraction...")

features_algo = extract_features(feature_extractor, dataloader, pca)

np.savez(f'/home/rfpred/data/custom_files/{args.subject}/center_strict/cnn_pcs_layer{this_layer}_{start}-{end}.npz', *features_algo)

print('gelukt hoor')

del model, pca

print('Deleted model and pca object to save memory')

if args.end == 30000:
    prf_region = 'center_strict'
    
    # Calculate total number of arrays across all files
    total_arrays = 0
    for file_name in sorted(os.listdir(f'/home/rfpred/data/custom_files/{args.subject}/{prf_region}/')):
        if file_name.startswith(f"cnn_pcs_layerfeatures.{args.cnn_layer}") and file_name.endswith(".npz"):
            data = np.load(f'/home/rfpred/data/custom_files/{args.subject}/{prf_region}/{file_name}')
            total_arrays += len(data.files)

    # Create layer7_feats with the correct size
    layer_feats = np.zeros((total_arrays, 600))

    n_file = -1
    current_array = 0
    for file_name in sorted(os.listdir(f'/home/rfpred/data/custom_files/{args.subject}/{prf_region}/')):
        n_file += 1
        if file_name.startswith(f"cnn_pcs_layerfeatures{args.cnn_layer}4") and file_name.endswith(".npz"):
            data = np.load(f'/home/rfpred/data/custom_files/{args.subject}/{prf_region}/{file_name}')
            n_imgs = len(data.files)
            
            if n_imgs > 0:
                session_ar = np.zeros((n_imgs, 600))
                for image in range(n_imgs):
                    session_ar[image, :] = data[f'arr_{image}']

                layer_feats[current_array:current_array+n_imgs, :] = session_ar
                current_array += n_imgs
                
                
    # if layer_feats.shape == (30000, 600):
    np.save(f'/home/rfpred/data/custom_files/subj01/center_strict/alex_lay{args.cnn_layer}.npy', layer_feats)
    print(f'Saved yet another collection of extracted features, this time from layer {args.cnn_layer}')
