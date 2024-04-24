#!/usr/bin/env python3

import os
# Limit the number of CPUs used to 2
os.environ["OMP_NUM_THREADS"] = "5"


import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
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
batch_size = 200
image_ids = get_imgs_designmx()[args.subject][start:end]
dataset = ImageDataset(image_ids, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

def fit_pca(feature_extractor, dataloader):

    # Define PCA parameters
    pca = IncrementalPCA(n_components=200, batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca

# pca = fit_pca(feature_extractor, train_imgs_dataloader)
pca = fit_pca(feature_extractor, dataloader)

def extract_features(feature_extractor, dataloader, pca):

    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features.append(ft)
    return np.vstack(features)

features_algo = extract_features(feature_extractor, dataloader, pca)

np.savez(f'/home/rfpred/data/custom_files/{args.subject}/center_strict/cnn_pcs_layer{this_layer}_{start}-{end}.npz', *features_algo)

print('gelukt hoor')

del model, pca
print('Deleted model and pca object to save memory')




# class AlexNetFeatureExtractorReLU(Module):
#     def __init__(self, model):
#         super(AlexNetFeatureExtractorReLU, self).__init__()
#         self.features = model.features
#         self.layer1 = torch.nn.Sequential(*list(self.features.children())[:2])
#         self.layer2 = torch.nn.Sequential(*list(self.features.children())[2:5])
#         self.layer3 = torch.nn.Sequential(*list(self.features.children())[5:7])
#         self.layer4 = torch.nn.Sequential(*list(self.features.children())[7:9])
#         self.layer5 = torch.nn.Sequential(*list(self.features.children())[9:12])
        


#     def forward(self, x):
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#         x5 = self.layer5(x4)

#         return x1, x2, x3, x4, x5

# # Initialize your model
# pca_feats_relu = AlexNetFeatureExtractorReLU(model)

# # Collect the features of each layer for multiple samples
# features = []
# batch_count = 0

# # for batch in dataloader:
# #     batch_count += 1
# #     print(f"Processing batch {batch_count} of {len(dataloader)}")

# #     img_tensor = (batch)  

# #     x1, x2, x3, x4, x5 = pca_feats_relu(img_tensor)
        
# #     features.append((x1.view(x1.size(0), -1).detach().numpy(), 
# #                         x2.view(x2.size(0), -1).detach().numpy(), 
# #                         x3.view(x3.size(0), -1).detach().numpy(), 
# #                         x4.view(x4.size(0), -1).detach().numpy(), 
# #                         x5.view(x5.size(0), -1).detach().numpy()))


# for batch in dataloader:
#     batch_count += 1
#     print(f"Processing batch {batch_count} of {len(dataloader)}")

#     img_tensor = (batch)  

#     # Check for NaN values in the input data
#     if torch.isnan(img_tensor).any():
#         print(f"\tBatch {batch_count} contains NaN values")

#     x1, x2, x3, x4, x5 = pca_feats_relu(img_tensor)
        
#     features.append((x1.view(x1.size(0), -1).detach().numpy(), 
#                         x2.view(x2.size(0), -1).detach().numpy(), 
#                         x3.view(x3.size(0), -1).detach().numpy(), 
#                         x4.view(x4.size(0), -1).detach().numpy(), 
#                         x5.view(x5.size(0), -1).detach().numpy()))


# print("Finished processing all batches")

# for batch in range((end - start) // batch_size):
#     for layer in range(5):
#         max_val = np.round(np.max(features[batch][layer]), 3)
#         min_val = np.round(np.min(features[batch][layer]), 3)
#         zeros = round(np.count_nonzero(features[batch][layer] == 0), 3)
#         non_zeros = round(np.count_nonzero(features[batch][layer] != 0), 3)
#         percentage_zeros = round(np.count_nonzero(features[batch][layer] == 0) / features[batch][layer].size * 100, 3)
        
#         print(f'Max value of batch {batch} layer {layer}: {max_val} min val: {min_val}')
#         print(f'Amount of zeros: {zeros}, amount of non-zeros: {non_zeros}')
#         print(f'Percentage of zeros: {round(percentage_zeros, 4)}%')
#         print(f'There are {np.isnan(features[batch][layer]).sum()} NaNs in this layer\n')


# ############OLDCODE
# pca_list = []  # List to store PCA objects for each layer

# for i in range(5):
#     print(f"Performing PCA on AlexNet layer {i+1}")
#     layer_features = np.concatenate([feat[i] for feat in features], axis=0)
#     # Scale the features
#     # Check if there are any NaN values in the data
#     if np.isnan(layer_features).any():
#         # print("NaN values found in layer_features. Imputing with mean.")
#         print("Still some NaNs in this shithole")
#     layer_features = np.nan_to_num(layer_features)  # Replace NaN values with 0
#     layer_features_win = winsorize_data(layer_features, lower_percentile=5, upper_percentile=95)
#     # Initialize and fit PCA for this layer
#     pca = PCA(n_components=600, svd_solver='full')  # Adjust as needed
#     # pca = PCA(n_components=.95, svd_solver='full')  # Adjust as needed
#     layer_features_pca = pca.fit_transform(layer_features_win)

#     pca_list.append(pca)  # Store the PCA object for later use


# print ("Finished performing PCA on all layers")

# # Initialize a new StandardScaler for each layer
# # scaler_list = [StandardScaler() for _ in range(5)]

# # Get the PCs for each layer
# pcs_list = []
# for i in range(5):
#     print(f"Scaling and transforming features of layer {i+1}")
#     # Get the features of the i-th layer for all images
#     layer_features = np.concatenate([feat[i] for feat in features], axis=0)
#     # Fit and transform the StandardScaler on the features
#     # scaler = scaler_list[i]
#     print("Before scaling:", np.isnan(layer_features).any())
#     layer_features = np.nan_to_num(layer_features)  # Replace NaN values with 0
#     layer_features_win = winsorize_data(layer_features, lower_percentile=5, upper_percentile=95)

#     # layer_features_scaled = scaler.fit_transform(layer_features_win)
#     # print("After scaling:", np.isnan(layer_features_scaled).any())
#     # Get the PCA object for this layer
#     pca = pca_list[i]
#     # Apply PCA to the scaled features
#     # layer_features_pca = pca.transform(layer_features_scaled)
#     layer_features_pca = pca.transform(layer_features)
#     # Add the PCs to the list
#     pcs_list.append(layer_features_pca)
    
# print("Finished scaling and transforming features of all layers")

# np.savez(f'/home/rfpred/data/custom_files/{args.subject}/center_strict/cnn_pcs{start}-{end}.npz', *pcs_list)

# print('gelukt hoor')