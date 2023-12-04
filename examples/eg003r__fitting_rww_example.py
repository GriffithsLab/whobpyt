# -*- coding: utf-8 -*-
r"""
=================================
Fitting Wong_Wang model (HCP data)
=================================

"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Importage
# ---------------------------------------------------
#

# os stuff
import os
import sys
sys.path.append('..')

# whobpyt stuff
import whobpyt
from whobpyt.datatypes import par, Recording
from whobpyt.models.RWW import RNNRWW, ParamsRWW
from whobpyt.optimization.custom_cost_RWW import CostsRWW
from whobpyt.run import Model_fitting
from whobpyt.data import fetch_hcpl2k8

# array and pd stuff
import numpy as np
import pandas as pd



# viz stuff
import matplotlib.pyplot as plt

#gdown
import gdown
# %%
# define destination path and download data


"""
des_dir = '../'
if not os.path.exists(des_dir):
    os.makedirs(des_dir)  # create folder if it does not exist
url = "https://drive.google.com/drive/folders/18smy3ElTd4VksoL4Z15dhwT5l3yjk6xS"
os.chdir(des_dir)
gdown.download_folder(url, remaining_ok = True)
"""

# Go to examples folder
#os.chdir('examples')


# Download the data for this example to default location ~/.whobpyt/data
data_dir = fetch_hcpl2k8()

base_dir = os.path.join(data_dir, 'HCP')

# %%
#get subject list
#base_dir = '../HCP/'

#subs =sorted([sc_file[-10:-4] for sc_file in os.listdir(base_dir) if sc_file[:8] == 'weights_'])

sub = '100307'



# %%
# define options for wong-wang model
node_size = 83
mask = np.tril_indices(node_size, -1)
num_epochs = 5
TPperWindow = 20
step_size = 0.05
input_size = 2
tr = 0.75
repeat_size = 5


# %%
# load raw data and get SC empirical BOLD and FC
sc_file = os.path.join(base_dir, 'weights_' + sub + '.txt')
ts_file = os.path.join(base_dir, sub + '_rfMRI_REST1_LR_hpc200_clean__l2k8_sc33_ts.pkl')  # out_dir+'sub_'+sub+'simBOLD_idt.txt'#


sc = np.loadtxt(sc_file)
SC = (sc + sc.T) * 0.5
sc = np.log1p(SC) / np.linalg.norm(np.log1p(SC))

ts_pd = pd.read_pickle(ts_file)
ts = ts_pd.values
ts = ts / np.max(ts)
fc_emp = np.corrcoef(ts.T)


# %%
# prepare data structure of the model
print(ts.T.shape)
fMRIstep = tr
data_mean = Recording(ts.T, fMRIstep) #dataloader(ts, num_epoches, TPperWindow)

# %%
# get model parameters structure and define the fitted parameters by setting non-zero variance for the model
params = ParamsRWW(g=par(400, 400, 1/np.sqrt(10), True, True), g_EE=par(1.5, 1.5, 1/np.sqrt(50), True, True), g_EI =par(0.8, 0.8, 1/np.sqrt(50), True, True), \
                   g_IE=par(0.6, 0.6, 1/np.sqrt(50), True, True), I_0 =par(0.2), std_in=par(0.0), std_out=par(0.00))

# %%
# call model want to fit
model = RNNRWW(node_size, TPperWindow, step_size, repeat_size, tr, sc, True, params)

# %%
# create objective function
ObjFun = CostsRWW(model)

# %%
# call model fit
F = Model_fitting(model, ObjFun)

# %%
# Model Training
# ---------------------------------------------------
#
F.train(u = 0, empRecs = [data_mean], num_epochs = num_epochs, TPperWindow = TPperWindow, learningrate = 0.05)

# %%
# Plots of loss over Training
plt.plot(np.arange(1,len(F.trainingStats.loss)+1), F.trainingStats.loss)
plt.title("Total Loss over Training Epochs")

# %%
# Plots of parameters values over Training
plt.plot(F.trainingStats.fit_params['g_EE'], label = "g_EE")
plt.plot(F.trainingStats.fit_params['g_EI'], label = "g_EI")
plt.plot(F.trainingStats.fit_params['g_IE'], label = "g_IE")
plt.legend()
plt.title("Local Coupling Variables Changing Over Training Epochs")

# %%
# Model Evaluation (with 20 window for warmup)
# ---------------------------------------------------
#
F.evaluate(u = 0, empRec = data_mean, TPperWindow = TPperWindow, base_window_num = 20)

# %%
# Plot SC and fitted SC

fig, ax = plt.subplots(1, 2, figsize=(5, 4))
im0 = ax[0].imshow(sc, cmap='bwr', vmin = 0.0, vmax = 0.05)
ax[0].set_title('The empirical SC')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(F.model.sc_fitted.detach().numpy(), cmap='bwr', vmin = 0.0, vmax = 0.05)
ax[1].set_title('The fitted SC')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.show()

# %%
# Plot E I and simulated BOLD
fig, ax = plt.subplots(1, 3, figsize=(12, 8))
ax[0].plot(F.lastRec['E'].npTS().T)
ax[0].set_title('Test: E')
ax[1].plot(F.lastRec['I'].npTS().T)
ax[1].set_title('Test: I')
ax[2].plot(F.lastRec['bold'].npTS().T)
ax[2].set_title('Test: BOLD')
plt.show()

# %%
# Plot the FC and the test FC
fig, ax = plt.subplots(1, 2, figsize=(5, 4))
im0 = ax[0].imshow(fc_emp, cmap='bwr')
ax[0].set_title('The empirical FC')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(np.corrcoef(F.lastRec['bold'].npTS()), cmap='bwr')
ax[1].set_title('The simulated FC')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.show()
