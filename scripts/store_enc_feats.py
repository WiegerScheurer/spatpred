#!/usr/bin/env python3

# This script pulls the feature maps from the specified layer of the CNN for each subject and makes them
# regression-ready. It saves the feature maps in the 'encoding' folder of the subject-specific location.
# All matrices are Z-scored per feature. This script should only be run once so that the feature maps are
# more easily accessible as numpy.ndarrays that can be loaded in in minimal time.

# I'VE ONLY RAN IT FOR SUBJECT 01 WITH THE SMALLPATCH, FIRST CHECK IF IT'S WORTH DOING.
# IF SO, ALSO RUN FOR THE OTHER SUBJECTS

import os

# Limit the number of CPUs used to 5
os.environ["OMP_NUM_THREADS"] = "5"

import sys
import numpy as np
from scipy.stats import zscore as zs

os.chdir("/home/rfpred")
sys.path.append("/home/rfpred")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

from classes.natspatpred import NatSpatPred

NSP = NatSpatPred()
NSP.initialise()


def _pull_featmaps(subject: str, cnn_layer: str, verbose: bool = False, modeltype:str = 'alexnet') -> np.array:
    """Function to pull the feature maps from the specified layer of the CNN, for the specified number of voxels.
        The raw files are dicts that result from the encoding_stack.sh script, with ['arr_{image_number}'] as keys.
        This function transforms these dicts into a 2d numpy that can be used as X matrix for the regression analysis.
        It is saved at the subject-specific location in the 'encoding' folder. Should only be used once per subject.

    Args:
    - subject (str): The subject for which the feature maps are pulled.
    - cnn_layer (str): The layer of the CNN from which the feature maps are pulled.
    - verbose (bool): Whether to print out the progress of the function.

    Out:
    - Xsubj (np.array): The feature maps for the specified subject and layer.
    """
    # Load in the feature maps for the specified layer of the CNN, dimensionality reduction has already been applied
    featmaps_pc = np.load(
        f"{NSP.own_datapath}/visfeats/cnn_featmaps/{modeltype}/featmaps/featmaps_layfeatures.{cnn_layer}.npz"
        # f"{NSP.own_datapath}/visfeats/cnn_featmaps/featmaps/featmaps_smallpatch_layfeatures.{cnn_layer}.npz"
    )

    # Convert the dictionary-like object to a numpy array
    Xpca = np.array([zs(featmaps_pc[key]) for key in featmaps_pc])
    if verbose:
        NSP.utils.inspect_dat(Xpca)

    # Subject specific indices, 30k images
    subj_ices = [NSP.stimuli.imgs_designmx()[subject]]
    subj_dmx = Xpca[subj_ices[0]]
    Xsubj = zs(subj_dmx, axis=0)  # Z-score the feature maps for every image

    os.makedirs(f"{NSP.own_datapath}/{subject}/encoding/{modeltype}", exist_ok=True)
    np.save(
        "%s/%s/encoding/%s/regprepped_featmaps_layer%s"
        # "%s/%s/encoding/regprepped_featmaps_smallpatch_layer%s"
        % (NSP.own_datapath, subject, modeltype, cnn_layer),
        Xsubj,
    )
    if verbose:
        print(f"Saved the feature maps for subject {subject} and layer {cnn_layer}.")

    return Xsubj


# The layers of AlexNet that are ReLU layers, others weren't extracted (yet)
# relu_lays = [1, 4, 7, 9, 11]
relu_lays = ["norm", 5, 10, 17, 24, 31]

# for subject in NSP.subjects:
for subject in ['subj01']:
    for cnn_layer in relu_lays:
        _ = _pull_featmaps(subject, cnn_layer, verbose=True, modeltype="vgg6")

print("Ook dit script heeft de klus wederom geklaard, chapeau!")
