"""
==============================================================
Example: Replicating Ismail et al. 2025
==============================================================
"""
# sphinx_gallery_thumbnail_number = 1
#

# %%
# 0. Overview
# ---------------------------------------------------
#
# This example replicates modelling in the Ismail et al. 2025 paper.
# The code includes data fetching, model fitting, and result visualization based on the methods presented in the paper.

# %%
# Summary of paper
# In this study, we explore the mechanisms underlying language lateralization in childhood 
# using personalized whole-brain network models. Our findings reveal that interhemispheric inhibitory 
# circuits play a crucial role in shaping lateralized language function, with local inhibition decreasing 
# over development while interhemispheric inhibition increases. Using systematic model manipulations and virtual 
# transplant experiments, we show that the reduction in local inhibition allows pre-existing asymmetries in interhemispheric 
# inhibition to drive laterality. This work provides a developmental framework for understanding how inhibitory circuits shape language networks.



# %%
# 1. Setup
# --------------------------------------------------

# Importage:
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import gdown

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
import os
import numpy as np
import gdown
from scipy.io import loadmat
from whobpyt.depr.ismail2025.jansen_rit import ParamsModel, RNNJANSEN, Model_fitting, dataloader


# -------------------------------------------------------------------------
# Download data
# -------------------------------------------------------------------------
#The data is available on a public Google Drive folder and includes:
#- verb_evoked.npy
#- noise_evoked.npy
#- leadfield_3d.mat
#- weights.csv (structural connectivity)
#- distance.txt (Euclidean inter-region distances)

folder_id = "1F8XOPfKihcV5hk0p9N_UVciyC5-SMHsn"
output_dir = "eg__ismail2025_data"

if not os.path.exists(output_dir):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, quiet=True, use_cookies=False)
 
# -------------------------------------------------------------------------
# Load Data
# -------------------------------------------------------------------------
verb_eeg_raw = np.load(os.path.join('eg__ismail2025_data', 'verb_evoked.npy'))   # (time, channels)
noise_eeg_raw = np.load(os.path.join('eg__ismail2025_data', 'noise_evoked.npy')) # (time, channels)

# Normalize both signals
verb_eeg = verb_eeg_raw / np.abs(verb_eeg_raw).max() * 1
noise_eeg = noise_eeg_raw / np.abs(noise_eeg_raw).max() * 1

leadfield = loadmat(os.path.join('eg__ismail2025_data', 'leadfield_3d.mat'))  # shape (sources, sensors, 3)
lm_3d = leadfield['M']  # 3D leadfield matrix

# Convert 3D to 2D using SVD-based projection
lm = np.zeros_like(lm_3d[:, :, 0])
for sources in range(lm_3d.shape[0]):
    u, d, v = np.linalg.svd(lm_3d[sources])
    lm[sources] = u[:,:3].dot(np.diag(d)).dot(v[0])
# Scale the leadfield matrix
lm = lm.T / 1e-11 * 5  # Shape: (channels, sources)

sc_df = pd.read_csv(os.path.join('eg__ismail2025_data', 'weights.csv'), header=None).values
sc = np.log1p(sc_df)
sc = sc / np.linalg.norm(sc)
dist = np.loadtxt(os.path.join('eg__ismail2025_data', 'distance.txt'))

# -------------------------------------------------------------------------
# Define simulation and model parameters
# -------------------------------------------------------------------------
node_size = sc.shape[0]
output_size = verb_eeg.shape[0]
batch_size = 250
step_size = 0.0001
input_size = 3
num_epoches = 2 #used 250 in paper using 2 for example
tr = 0.001
state_size = 6
base_batch_num = 20
time_dim = verb_eeg.shape[1]
hidden_size = int(tr/step_size)


# Format input data
data_verb = dataloader(verb_eeg.T, num_epoches, batch_size)
data_noise = dataloader(noise_eeg.T, num_epoches, batch_size)

ki0 = np.zeros((node_size, 1))
ki0[[2, 183, 5]] = 1

lm_n = 0.01 * np.random.randn(output_size, node_size)
lm_v = 0.01 * np.random.randn(output_size, node_size)

par = ParamsModel('JR', A = [3.25, 0.1], a= [100, 1], B = [22, 0.5], b = [50, 1], g=[400, 1], g_f=[10, 1], g_b=[10, 1],\
                    c1 = [135, 1], c2 = [135*0.8, 1], c3 = [135*0.25, 1], c4 = [135*0.25, 1],\
                    std_in=[0, 1], vmax= [5, 0], v0=[6,0], r=[0.56, 0], y0=[-0.5 , 0.05],\
                    mu = [1., 0.1], k = [5, 0.2], kE = [0, 0], kI = [0, 0],
                    cy0 = [5, 0], ki=[ki0, 0], lm=[lm+lm_n, .1 * np.ones((output_size, node_size))+lm_v])

# -------------------------------------------------------------------------
# Run verb model fitting
# -------------------------------------------------------------------------
verb_model = RNNJANSEN(node_size, batch_size, step_size, output_size, tr, sc, lm, dist, True, False, par)
verb_model.setModelParameters()

#Stimulate the auditory cortices defined by roi in ki0
stim_input = np.zeros((node_size, hidden_size, time_dim))
stim_input[:, :, 100:140] = 5000

verb_F = Model_fitting(verb_model, data_verb, num_epoches, 0)
verb_F.train(u=stim_input)
verb_F.test(base_batch_num, u=stim_input)

print("Finished fitting model to verb trials")

# -------------------------------------------------------------------------
# Run noise model fitting
# -------------------------------------------------------------------------
noise_model = RNNJANSEN(node_size, batch_size, step_size, output_size, tr, sc, lm, dist, True, False, par)
noise_model.setModelParameters()

noise_F = Model_fitting(noise_model, data_noise, num_epoches, 0)
noise_F.train(u=stim_input)
noise_F.test(base_batch_num, u=stim_input)

print("Finished fitting model to noise trials")





