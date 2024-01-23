# -*- coding: utf-8 -*-
r"""
=================================
Fitting Concurrent EEG and BOLD data using RWW
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
from whobpyt.data import dataloader
from whobpyt.models.RWW import ParamsRWW
from whobpyt.models.EEG import ParamsEEG
from whobpyt.models.BOLD import ParamsBOLD
from whobpyt.models.RWW.wong_wang_rt import RNNRWWMM
from whobpyt.models.RWWNEURO import ParamsRWWNEU
from whobpyt.models.RWWEI2 import RWW_EEG_BOLD
from whobpyt.optimization.custom_cost_RWW2 import CostsRWW2
from whobpyt.run import Model_fitting
from whobpyt.data import fetch_hcpl2k8

# array and pd stuff
import numpy as np
import pandas as pd

# viz stuff
import seaborn as sns
import matplotlib.pyplot as plt

#gdown
import gdown
# %%
# define destination path and download data


des_dir = '../'
if not os.path.exists(des_dir):
    os.makedirs(des_dir)  # create folder if it does not exist
url = "https://drive.google.com/drive/folders/1fCmBKkvbWKZZqx36jviIi7B2PWhq7lI8"
os.chdir(des_dir)
gdown.download_folder(url, remaining_ok = True)

# Go to examples folder
os.chdir('examples')

base_dir = '../EEG_BOLD/'

sub_ID ='32'
eeg_file = base_dir+'sub-'+ sub_ID + '_eeg.npy'
bold_file = base_dir+'sub-'+ sub_ID + '_fmri.npy'

eeg = np.load(eeg_file)
bold = np.load(bold_file)
eeg_data = eeg[1:,:,::10]*1e5

# %%
# get SC and distance template
sc_file = base_dir + 'Schaefer2018_200Parcels_7Networks_count.csv'
dist_file = base_dir + 'Schaefer2018_200Parcels_7Networks_distance.csv'
sc_df = pd.read_csv(sc_file, header=None, sep=' ')
sc = sc_df.values
dist_df = pd.read_csv(dist_file, header=None, sep=' ')
dist = dist_df.values
sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

# %%
# define options for RWW2 (multimodal)
step_size = 0.05
tr = 0.25*25
tr_eeg = 0.25
node_size = sc.shape[0]
pop_size = 1
output_size = eeg_data.shape[1]
TPperWindow = 37

num_epochs = 2

state_size = 6

bold_mean = dataloader(bold, num_epochs, TPperWindow)
eeg_mean =dataloader(np.concatenate(list(eeg_data), 1).T, num_epochs, 37*eeg_data.shape[2])
lm = np.zeros((output_size,200))
lm_v = np.zeros((output_size,200))
params = ParamsRWW(g=par(400, 400, 1/np.sqrt(10), True), g_EE=par(1.5, 1.5, 1/np.sqrt(50), True), \
                   g_EI =par(0.8, 0.8, 1/np.sqrt(50), True), \
                  g_IE=par(np.log(0.6), np.log(0.6), 0.1, True, True), I_0 =par(0.2), \
                   std_in=par(np.log(0.2), np.log(0.2), 0.1, True, True), std_out=par(0.00), \
                   lm=par(lm, lm, 0.1 * np.ones((output_size, node_size))+lm_v, True))

# %%
# call model want to fit
model = RNNRWWMM(params, node_size =node_size, output_size=output_size, TRs_per_window =TPperWindow, step_size=step_size, \
                   tr=tr, tr_eeg= tr_eeg, sc=sc, use_fit_gains=True)

# %%
# create objective function
ObjFun = CostsRWW2(model)

# %%
# call model fit
F = Model_fitting(model, ObjFun)

# %%
# Model Training
# ---------------------------------------------------
#
F.train(u = 0, empRec = bold_mean, empRecSec=eeg_mean, num_epochs = num_epochs, TPperWindow = TPperWindow, warmupWindow=5, learningrate = 0.05)

F.evaluate(u = 0, empRec = bold_mean, empRecSec=eeg_mean, TPperWindow = TPperWindow, base_window_num = 5)

# %%
# Plots of loss over Training
plt.plot(np.arange(1,len(F.trainingStats.loss)+1), F.trainingStats.loss)
plt.title("Main Loss over Training Epochs")

# Plots of BOLD FC
sim_FC = np.corrcoef(F.trainingStats.outputs['bold_testing'])
emp_FC = np.corrcoef(bold.T)
fig, ax = plt.subplots(1,2,figsize = (8, 8))
plt.suptitle("Simulated BOLD FC: After Training")
mask = np.eye(200)
sns.heatmap(sim_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0, ax=ax[0])
ax[0].set_title('simulated')
sns.heatmap(emp_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0, ax=ax[1])
ax[1].set_title('empirical')

#EEG FC
#EEG FC
sim_FC = np.corrcoef(F.trainingStats.outputs['eeg_testing'])
emp_FC = np.corrcoef(np.concatenate(list(eeg_data),1))
fig, ax = plt.subplots(1,2,figsize = (8, 8))
plt.suptitle("Simulated EEG FC: After Training")
mask = np.eye(output_size)
sns.heatmap(sim_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0, ax=ax[0])
ax[0].set_title('simulated')
sns.heatmap(emp_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0, ax=ax[1])
ax[1].set_title('empirical')


#################################################
#using EEG BOLD RWW saparate models

bold_mean = dataloader(bold, num_epochs, TPperWindow)
eeg_mean =dataloader(np.concatenate(list(eeg_data), 1).T, num_epochs, 37*eeg_data.shape[2])
lm = np.zeros((output_size,200))
lm_v = np.zeros((output_size,200))
params_rww = ParamsRWWNEU(g=par(400, 400, 1/np.sqrt(10), True), g_EE=par(1.5, 1.5, 1/np.sqrt(50), True), \
                   g_EI =par(0.8, 0.8, 1/np.sqrt(50), True), \
                  g_IE=par(np.log(0.6), np.log(0.6), 0.1, True, True), I_0 =par(0.2), \
                   std_in=par(np.log(0.2), np.log(0.2), 0.1, True, True), std_out=par(0.00))
params_bold = ParamsBOLD()
params_eeg = ParamsEEG(lm=par(lm, lm, 0.1 * np.ones((output_size, node_size))+lm_v, True))
# %%
# call model want to fit
model = RWW_EEG_BOLD(params_rww,params_eeg, params_bold, node_size =node_size, output_size=output_size, TRs_per_window =TPperWindow, step_size=step_size, \
                   tr=tr, tr_eeg= tr_eeg, sc=sc, use_fit_gains=True)

# %%
# create objective function
ObjFun = CostsRWW2(model)

# %%
# call model fit
F = Model_fitting(model, ObjFun)

# %%
# Model Training
# ---------------------------------------------------
#
F.train(u = 0, empRec = bold_mean, empRecSec=eeg_mean, num_epochs = num_epochs, TPperWindow = TPperWindow, warmupWindow=5, learningrate = 0.05)

F.evaluate(u = 0, empRec = bold_mean, empRecSec=eeg_mean, TPperWindow = TPperWindow, base_window_num = 5)

# %%
# Plots of loss over Training
plt.plot(np.arange(1,len(F.trainingStats.loss)+1), F.trainingStats.loss)
plt.title("Main Loss over Training Epochs")

# Plots of BOLD FC
sim_FC = np.corrcoef(F.trainingStats.outputs['bold_testing'])
emp_FC = np.corrcoef(bold.T)
fig, ax = plt.subplots(1,2,figsize = (8, 8))
plt.suptitle("Simulated BOLD FC: After Training")
mask = np.eye(200)
sns.heatmap(sim_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0, ax=ax[0])
ax[0].set_title('simulated')
sns.heatmap(emp_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0, ax=ax[1])
ax[1].set_title('empirical')

#EEG FC
#EEG FC
sim_FC = np.corrcoef(F.trainingStats.outputs['eeg_testing'])
emp_FC = np.corrcoef(np.concatenate(list(eeg_data),1))
fig, ax = plt.subplots(1,2,figsize = (8, 8))
plt.suptitle("Simulated EEG FC: After Training")
mask = np.eye(output_size)
sns.heatmap(sim_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0, ax=ax[0])
ax[0].set_title('simulated')
sns.heatmap(emp_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0, ax=ax[1])
ax[1].set_title('empirical')