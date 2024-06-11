#!/usr/bin/env python3

# This script does the automatic alignment for freesurfer surfaces to be used for pycortex

import os
import sys

os.environ["OMP_NUM_THREADS"] = "10"

import os
import sys
import cortex

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

def reload_nsp():
    import funcs.natspatpred
    importlib.reload(funcs.natspatpred)
    from funcs.natspatpred import NatSpatPred, VoxelSieve
    NSP = NatSpatPred()
    NSP.initialise()
    return NSP

import lgnpy.CEandSC.lgn_statistics
# from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN
import importlib
from importlib import reload
import funcs.natspatpred
import unet_recon.inpainting

importlib.reload(funcs.natspatpred)
importlib.reload(unet_recon.inpainting)
importlib.reload(lgnpy.CEandSC.lgn_statistics)

from unet_recon.inpainting import UNet
from funcs.natspatpred import NatSpatPred, VoxelSieve
from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN

NSP = NatSpatPred()
NSP.initialise()

print(cortex.options.config['basic']['filestore'])
pc_files = '/home/rfpred/envs/rfenv/share/pycortex/db'

# Reload the subjects and check whether they've successfully been reloaded in the pycortex database
cortex.db.reload_subjects()
print(cortex.db.subjects.keys())

# Manually set the subjects directory to the specific FreeSurfer directory
os.environ['SUBJECTS_DIR'] = f"{NSP.nsd_datapath}/nsddata/freesurfer/"
print(os.environ['SUBJECTS_DIR'])


for subject in NSP.subjects[1:]:
    subj_path = f'{pc_files}/{subject}/'
    cortex.align.automatic(subject, 'test_transform', f'{subj_path}anatomicals/raw.nii.gz')


print("All subjects' cortices have been aligned automatically, now proceed to the manual part.")