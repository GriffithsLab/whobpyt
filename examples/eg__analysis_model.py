"""
.. _ex-modelanalysis:

========================================================
Modelling TMS-EEG evoked responses
========================================================

This example shows the analysis from model:

1. model parameters
2. networks
3. neural states

"""
# %%  
# First we must import the necessary packages required for the example:  

# System-based packages
import os
import sys
sys.path.append('..')


# Whobpyt modules taken from the whobpyt package
import whobpyt
from whobpyt.datatypes import Parameter as par, Timeseries
from whobpyt.models.jansen_rit import JansenRitModel,JansenRitParams
from whobpyt.run import ModelFitting
from whobpyt.optimization.custom_cost_JR import CostsJR
from whobpyt.datasets.fetchers import fetch_egtmseeg

# Python Packages used for processing and displaying given analytical data (supported for .mat and Google Drive files)
import numpy as np
import pandas as pd
import scipy.io
import gdown
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt # Plotting library (For Visualization)
import seaborn as sns

import mne # Neuroimaging package






# %%
# load in a previously completed model fitting results object
full_run_fname = os.path.join(data_dir, 'Subject_1_low_voltage_fittingresults_stim_exp.pkl')
F = pickle.load(open(full_run_fname, 'rb'))


### get labels for Yeo 200
url = 'https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
atlas = pd.read_csv(url)
labels = atlas['ROI Name']

# get networks 
nets = [label.split('_')[2] for label in labels]
net_names = np.unique(np.array(nets))

# %% 
# 1. model parameters
# -----------------------------------

# %%
# Plots of parameter values over Training (check if converges)
fig, axs = plt.subplots(2,2, figsize = (12,8))
paras = ['c1', 'c2', 'c3', 'c4']
for i in range(len(paras)):
    axs[i//2,i%2].plot(F.trainingStats.fit_params[paras[i]])
    axs[i//2, i%2].set_title(paras[i])
plt.title("Select Variables Changing Over Training Epochs")

# %%
# Plots of parameter values over Training (prior vs post)
fig, axs = plt.subplots(2,2, figsize = (12,8))
paras = ['c1', 'c2', 'c3', 'c4']
for i in range(len(paras)):
    axs[i//2,i%2].hist(F.trainingStats.fit_params[paras[i]][:500], label='prior')
    axs[i//2,i%2].hist(F.trainingStats.fit_params[paras[i]][-500:], label='post')
    axs[i//2, i%2].set_title(paras[i])
plt.title("Prior vs Post")

# %% 
# 2. Networks
# -----------------------------------
fig, axs = plt.subplots(1,3, figsize = (12,8))
networks_frommodels = ['p2p', 'p2e', 'p2i']
sns.heatmap(F.model.w_p2p.detach().numpy(), cmap = 'bwr', center=0, ax=axs[0])
axs[0].set_title(networks_frommodels[0])
sns.heatmap(F.model.w_p2p.detach().numpy(), cmap = 'bwr', center=0, ax=axs[1])
axs[1].set_title(networks_frommodels[1])
sns.heatmap(F.model.w_p2p.detach().numpy(), cmap = 'bwr', center=0, ax=axs[2])
axs[2].set_title(networks_frommodels[2])
    

# %% 
# 3. Neural states
# -----------------------------------

#### plot E response on each networks 
fig, ax = plt.subplots(2,4, figsize=(12,10), sharey= True)
t = np.linspace(-0.1,0.3, 400)

for i, net in enumerate(net_names):
    mask = np.array(nets) == net
    ax[i//4, i%4].plot(t, F.lastRec['E'].npTS()[mask,:].mean(0).T)
    ax[i//4, i%4].set_title(net)
plt.suptitle('Test: E')
plt.show()

### plot I response at each networks
fig, ax = plt.subplots(2,4, figsize=(12,10), sharey= True)
t = np.linspace(-0.1,0.3, 400)

for i, net in enumerate(net_names):
    mask = np.array(nets) == net
    ax[i//4, i%4].plot(t, F.lastRec['I'].npTS()[mask,:].mean(0).T)
    ax[i//4, i%4].set_title(net)
plt.suptitle('Test: I')
plt.show()

### plot P response at each networks
fig, ax = plt.subplots(2,4, figsize=(12,10), sharey= True)
t = np.linspace(-0.1,0.3, 400)

for i, net in enumerate(net_names):
    mask = np.array(nets) == net
    ax[i//4, i%4].plot(t, F.lastRec['P'].npTS()[mask,:].mean(0).T)
    ax[i//4, i%4].set_title(net)
plt.suptitle('Test: P')
plt.show()


### plot phase of E at each network
j = complex(0,1)
fig, ax = plt.subplots(2,4, figsize=(12,10), sharey= True)
t = np.linspace(-0.1,0.3, 400)

phase = np.angle(F.lastRec['E'].npTS()+j*F.lastRec['Ev'].npTS())
for i, net in enumerate(net_names):
    mask = np.array(nets) == net
    ax[i//4, i%4].plot(t, phase[mask,:].mean(0).T)
    ax[i//4, i%4].set_title(net)
plt.suptitle('Test: phase E')
plt.show()

### plot I phase at each network
j = complex(0,1)
fig, ax = plt.subplots(2,4, figsize=(12,10), sharey= True)
t = np.linspace(-0.1,0.3, 400)

phase = np.angle(F.lastRec['I'].npTS()+j*F.lastRec['Iv'].npTS())
for i, net in enumerate(net_names):
    mask = np.array(nets) == net
    ax[i//4, i%4].plot(t, phase[mask,:].mean(0).T)
    ax[i//4, i%4].set_title(net)
plt.suptitle('Test: phase I')
plt.show()

### plot P phase at each network

j = complex(0,1)
fig, ax = plt.subplots(2,4, figsize=(12,10), sharey= True)
t = np.linspace(-0.1,0.3, 400)

phase = np.angle(F.lastRec['P'].npTS()+j*F.lastRec['Pv'].npTS())
for i, net in enumerate(net_names):
    mask = np.array(nets) == net
    ax[i//4, i%4].plot(t, phase[mask,:].mean(0).T)
    ax[i//4, i%4].set_title(net)
plt.suptitle('Test: phase P')
plt.show()

