# -*- coding: utf-8 -*-
r"""
=================================
Fitting JR model (Tepit data)
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
from whobpyt.models.JansenRit import RNNJANSEN, ParamsJR
from whobpyt.optimization.custom_cost_JR import CostsJR
from whobpyt.run import Model_fitting

# array and pd stuff
import numpy as np
import pandas as pd
import scipy.io


# viz stuff
import matplotlib.pyplot as plt

#gdown
import gdown
# %%
# define destination path and download data
des_dir = '../'
if not os.path.exists(des_dir):
    os.makedirs(des_dir)  # create folder if it does not exist
url = 'https://drive.google.com/drive/folders/1uXrtehuMlLBvPCV8zDaUYxF-MoMaD0fk'
os.chdir(des_dir)
gdown.download_folder(url, quiet = True, use_cookies = False)
os.chdir('examples/')


# %%
# get  EEG data
base_dir = '../Tepit/'
eeg_file = base_dir + 'eeg_data.npy'
eeg_data_all = np.load(eeg_file)
eeg_data = eeg_data_all.mean(0) 

eeg_data = eeg_data[:,700:1100] / 12

# %%
# get stimulus weights on regions
ki0 =np.loadtxt(base_dir + 'stim_weights.txt')[:,np.newaxis]

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
# define options for JR model
node_size = sc.shape[0]

output_size = eeg_data.shape[0]
TPperWindow = 20
step_size = 0.0001
num_epochs = 20
tr = 0.001
state_size = 6
base_batch_num = 200
time_dim = 400
state_size = 6
base_batch_num = 20
hidden_size = int(tr/step_size)


# %%
# prepare data structure of the model
print(eeg_data.shape)
EEGstep = tr
data_mean = dataloader(eeg_sub.T, num_epoches, batch_size)
#data_mean = Recording(eeg_data, EEGstep) #dataloader(eeg_data.T, num_epochs, batch_size)

# %%
# get model parameters structure and define the fitted parameters by setting non-zero variance for the model
lm = np.zeros((output_size,200))
lm_v = np.zeros((output_size,200))
params = ParamsCBnet( g= par(400), g_f = par(10), g_b=par(10),\
                  sigma_V=par(np.log(50), np.log(50), 1, True, True),\
                   sigma_g=par(np.log(1), np.log(1), .1, True, True), \
                  y0=par(-0.2,-0.2, .01, True),  \
                   pi_sigma=par(np.log(32)*np.ones((3,1)), np.log(32)*np.ones((3,1)), 0.1*np.ones((3,1)), True, True),\
                   mu = par(np.log(1.5), np.log(1.5), 0.1, True, True), k = par(10, 10, 1, True), \
                  gamma_gE= par(.5*np.ones((3,3)), .5*np.ones((3,3)),  0.1*np.ones((3,3)), True, True), \
                  gamma_gI = par(.5*np.ones((3,3)), .5*np.ones((3,3)),  0.1*np.ones((3,3)), True, True), \
                   gamma_k = par(30),\
                   gamma_gNMDA = par(.5*np.ones((3,3)), .5*np.ones((3,3)), 0.1*np.ones((3,3)), True, True),\
                   cy0 = par(50, 50, 1, True), ki=par(ki0), \
                   lm=par(lm, lm, .1 * np.ones((output_size, node_size))+lm_v, True))

model = RNNCBNET(node_size, pop_size,batch_size, step_size, output_size, tr, sc, lm, dist, True, params)
# %%
# create objective function
ObjFun = CostsJR(model)

# %%
# call model fit
F = Model_fitting(model, ObjFun)

# %%
# Model Training
# ---------------------------------------------------
#
u = np.zeros((node_size,hidden_size,time_dim))
u[:,:,110:120]= 200
F.train(u = u, empRecs = [data_mean], num_epochs = num_epochs, TPperWindow = TPperWindow)

# %%
# Plots of loss over Training
plt.plot(np.arange(1,len(F.trainingStats.loss)+1), F.trainingStats.loss)
plt.title("Total Loss over Training Epochs")

# %%
# Plots of parameter values over Training
"""plt.plot(F.trainingStats.fit_params['a'], label = "a")
plt.plot(F.trainingStats.fit_params['b'], label = "b")
plt.plot(F.trainingStats.fit_params['c1'], label = "c1")
plt.plot(F.trainingStats.fit_params['c2'], label = "c2")
plt.plot(F.trainingStats.fit_params['c3'], label = "c3")
plt.plot(F.trainingStats.fit_params['c4'], label = "c4")
plt.legend()
plt.title("Select Variables Changing Over Training Epochs")"""

# %%
# Model Evaluation (with 20 window for warmup)
# ---------------------------------------------------
#
F.evaluate(u = u, empRec = data_mean, TPperWindow = TPperWindow, base_window_num = 100)

# %%
# Plot SC and fitted SC
fig, ax = plt.subplots(1, 2, figsize=(5, 4))
im0 = ax[0].imshow(sc, cmap='bwr', vmin = 0.0, vmax = 0.02)
ax[0].set_title('The empirical SC')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(F.model.sc_fitted.detach().numpy(), cmap='bwr', vmin = 0.0, vmax = 0.02)
ax[1].set_title('The fitted SC')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.show()

# %%
# Plot the EEG
fig, ax = plt.subplots(1,3, figsize=(12,8))
ax[0].plot(F.trainingStats.stats['testing'][:,2,0].T)
ax[0].set_title('Test: sourced EEG')
ax[1].plot(F.trainingStats.outputs['testing'].T)
ax[1].set_title('Test')
ax[2].plot(eeg_data.T)
ax[2].set_title('empirical')
plt.show()