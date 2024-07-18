#!/usr/bin/env python3

# This script pulls the feature maps from the specified layer of the CNN for each subject runs
# dimensionality reduction on them using incremental PCA. Can take a while and can be adapted

import os

# Limit the number of CPUs used to 2
os.environ["OMP_NUM_THREADS"] = "10"

import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import joblib
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from torchvision import models
from typing import Dict, Tuple, Union, Optional

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

from classes.natspatpred import NatSpatPred
NSP = NatSpatPred()
NSP.initialise()

# Define the run parameteres using argparse
predparser = argparse.ArgumentParser(
    description="Get the AlexNet feature space neural representations for a range of images of a subject"
)

predparser.add_argument(
    "pca_fit_batch", type=int, help="The size of the image batch to fit the PCA to"
)  # Standard is 1000
predparser.add_argument(
    "n_comps", type=int, help="The fixed number of principal components to extract"
)  # Standard is 1000
predparser.add_argument(
    "cnn_layer", help="The layer to extract neural representations of"
)

args = predparser.parse_args()
prf_region = "center_strict"

# Load the pretrained AlexNet model
# model = models.vgg16_bn(pretrained=True)
model = models.vgg16(pretrained=True)
model.eval()  # Set the model to evaluation mode
modeltype = model._get_name()

class ImageDataset(Dataset):
    def __init__(self, image_ids, transform=None, crop: bool = True):
        self.image_ids = image_ids
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        if self.crop:
            imgnp = NSP.stimuli.show_stim(img_no=img_id, hide=True, small=True, crop=False)[0][
                163:263, 163:263
            ]  # I CROP THEM, YOU SEE
        else:
            imgnp = NSP.stimuli.show_stim(img_no=img_id, hide=True, small=True, crop=False)[0]

        imgPIL = Image.fromarray(imgnp)  # Convert into PIL from np

        if self.transform:
            imgPIL = self.transform(imgPIL)

        return imgPIL


preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # resize the images to 224x24 pixels
        transforms.ToTensor(),  # convert the images to a PyTorch tensor
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # normalize the images color channels
    ]
)

train_nodes, _ = get_graph_node_names(model)
print(train_nodes)


this_layer = train_nodes[args.cnn_layer + 1] if args.cnn_layer != "norm" else "x"

# Which layer to extract the features from # Also add this as argparse thing.
# model_layer = "features.2" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}

# if args.cnn_layer != "norm":
feature_extractor = create_feature_extractor(model, return_nodes=[this_layer])

train_batch = args.pca_fit_batch
apply_batch = 1000  # The image batch over which the fitted PCA is applied later on.
fixed_n_comps = args.n_comps

# image_ids = get_imgs_designmx()[args.subject][start:end] # This was for subject-specific image indices. Current line (below) is for all images.
image_ids = list(range(0, train_batch))
dataset = ImageDataset(image_ids, transform=preprocess, crop=False) # CHECK THIS CROP ARG
dataloader = DataLoader(dataset, batch_size=train_batch, shuffle=False)

# Normalization Layer for VGG
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.mean = 
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, input):
        # normalize img
        if self.mean.type() != input.type():
            self.mean = self.mean.to(input)
            self.std = self.std.to(input)
        return (input - self.mean) / self.std

def extract_features(feature_extractor, dataloader, pca, cnn_layer: int|str):
    while True:  # Keep trying until successful
        try:
            features = []
            for i, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Calculate mean and std of the current batch
                # mean = d.mean([0, 2, 3])
                # std = d.std([0, 2, 3])
                
                MEAN = [0.485, 0.456, 0.406]
                STD = [0.229, 0.224, 0.225]

                # Extract features
                if cnn_layer == "norm":
                    # Create an instance of the Normalization class
                    normalizer = Normalization(MEAN, STD)
                    # Normalize the input tensor
                    # ft = [normalizer]
                    ft = normalizer(d)
                    # Flatten the normalised tensor
                    ft = torch.flatten(ft, start_dim=1)
                else:
                    ft = feature_extractor(d)
                    # Flatten the features
                    ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])

                # Print out some summary statistics of the features
                print(
                    f"AlexNet layer: {cnn_layer}, Mean: {ft.mean()}, Std: {ft.std()}, Min: {ft.min()}, Max: {ft.max()}"
                )

                # Check if the features contain NaN values
                if np.isnan(ft.detach().numpy()).any():
                    raise ValueError("NaN value detected")

                # Check for extreme outliers
                if (ft.detach().numpy() < -100000).any() or (
                    ft.detach().numpy() > 100000
                ).any():
                    raise ValueError("Extreme outlier detected before PCA fit")

                # Apply PCA transform
                ft = pca.transform(ft.cpu().detach().numpy())
                features.append(ft)
            return np.vstack(features)  # Return the features
        except ValueError as e:
            print(f"Error occurred: {e}")
            print("Restarting feature extraction...")



def extract_features_and_check(d, feature_extractor, cnn_layer):
    while True:  # Keep trying until successful
        try:
            
            # Calculate mean and std of the current batch
            mean = d.mean([0, 2, 3])
            std = d.std([0, 2, 3])

            # Extract features
            if cnn_layer == "norm":
                # Create an instance of the Normalization class
                normalizer = Normalization(mean, std)
                # Normalize the input tensor
                # ft = normalizer(d).to_sparse()
                # ft = [normalizer]
                ft = normalizer(d)
                # Flatten the normalised tensor
                ft = torch.flatten(ft, start_dim=1)
            else: 
                # Extract features
                ft = feature_extractor(d)
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])

            # Check for NaN values
            if np.isnan(ft.detach().numpy().any()):
                raise ValueError("NaN value detected before PCA fit")

            # Check for extreme outliers
            if (ft.detach().numpy() < -100000).any() or (ft.detach().numpy() > 100000).any():
                raise ValueError("Extreme outlier detected before PCA fit")

            return ft  # If everything is fine, return the features

        except ValueError as e:
            print(f"Error occurred: {e}")
            print("Restarting feature extraction...")


def fit_pca(
    feature_extractor,
    dataloader,
    pca_save_path=None,
    fixed_n_comps: Optional[int] = None,
    train_batch: int = None,
    cnn_layer: int|str = None,
):
    # Define PCA parameters
    pca = IncrementalPCA(n_components=None, batch_size=train_batch)

    try:
        if fixed_n_comps is None:
            # Fit PCA to batch to determine number of components
            print(
                "Determining the number of components to maintain 95% of the variance..."
            )
            for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                ft = extract_features_and_check(d, feature_extractor, cnn_layer)
                # Fit PCA to batch
                pca.partial_fit(ft.detach().cpu().numpy())

            # Calculate cumulative explained variance ratio
            cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)
            # Find the number of components to maintain 95% of the variance
            n_comps = np.argmax(cumulative_var_ratio >= 0.95) + 1
            print(f"Number of components to maintain 95% of the variance: {n_comps}")

        else:
            n_comps = fixed_n_comps
            print(f"Using fixed number of components: {n_comps}")

        # Set the number of components
        pca = IncrementalPCA(n_components=n_comps, batch_size=train_batch)

        # Fit PCA to the entire dataset
        print("Fitting PCA with determined number of PCs to batch...")
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            ft = extract_features_and_check(d, feature_extractor, cnn_layer)
            # Fit PCA to batch
            pca.partial_fit(ft.detach().cpu().numpy())

        # Save the fitted PCA object if specified
        if pca_save_path:
            print(f"Saving fitted PCA object to: {pca_save_path}")
            joblib.dump(pca, pca_save_path)

        # Return the fitted PCA object
        print("PCA fitting completed.")
        return pca

    except Exception as e:
        print(f"Error occurred: {e}")
        print("PCA fitting failed.")
        return None


os.makedirs(f"{NSP.own_datapath}/visfeats/cnn_featmaps/{modeltype}/", exist_ok=True)

# Fit PCA and get the fitted PCA object
pca = fit_pca(
    feature_extractor,
    dataloader,
    # pca_save_path=f"/home/rfpred/data/custom_files/visfeats/cnn_featmaps/pca_{args.cnn_layer}_{fixed_n_comps}pcs.joblib",
    pca_save_path=f"{NSP.own_datapath}/visfeats/cnn_featmaps/{modeltype}/pca_{args.cnn_layer}_{fixed_n_comps}pcs.joblib",
    fixed_n_comps=fixed_n_comps,
    train_batch=train_batch,
    cnn_layer=args.cnn_layer,
    )

del dataloader, dataset

# Redefine the dataset and dataloader with the entire image set to apply the fitted PCA to.
all_img_ids = list(range(0, 73000))  # All the NSD images
# all_img_ids = list(NSP.stimuli.imgs_designmx()["subj01"]) # If it still is too heavy
full_dataset = ImageDataset(all_img_ids, transform=preprocess, crop=False)
full_dataloader = DataLoader(full_dataset, batch_size=apply_batch, shuffle=False)

# Check if PCA fitting was successful
if pca is not None:
    # Apply the fitted PCA to the rest of the dataset
    features_algo = extract_features(
        feature_extractor, full_dataloader, pca, args.cnn_layer
    )
else:
    print("PCA fitting failed. Unable to apply PCA, fock.")

# np.savez(
#     # f"/home/rfpred/data/custom_files/visfeats/cnn_featmaps/featmaps/featmaps_lay{this_layer}.npz",
#     f"/home/rfpred/data/custom_files/visfeats/cnn_featmaps/featmaps/featmaps_smallpatch_lay{this_layer}.npz",
#     *features_algo,
# )

os.makedirs(f"{NSP.own_datapath}/visfeats/cnn_featmaps/{modeltype}/featmaps/", exist_ok=True)

np.savez(
    # f"/home/rfpred/data/custom_files/visfeats/cnn_featmaps/featmaps/featmaps_lay{this_layer}.npz",
    f"{NSP.own_datapath}/visfeats/cnn_featmaps/{modeltype}/featmaps/featmaps_lay{this_layer}.npz",
    *features_algo,
)

print("Ook deze VGG (Visual Geometry Group) tori is je gelukt, generaal")
