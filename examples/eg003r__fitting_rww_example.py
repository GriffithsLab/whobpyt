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
from whobpyt.data.dataload import dataloader
from whobpyt.models.RWW.wong_wang import ParamsRWW
from whobpyt.models.RWW.wong_wang import RNNRWW
from whobpyt.optimization.modelfitting import Model_fitting
from whobpyt.optimization.custom_cost_RWW import CostsRWW

# array and pd stuff
import numpy as np
import pandas as pd



# viz stuff
import matplotlib.pyplot as plt

#gdown
import gdown
# %%
# define destination path and download data
des_dir = '../'
if not os.path.exists(des_dir):
    os.makedirs(des_dir)  # create folder if it does not exist
url = "https://drive.google.com/drive/folders/18smy3ElTd4VksoL4Z15dhwT5l3yjk6xS"
os.chdir(des_dir)
gdown.download_folder(url, remaining_ok = True)
os.chdir('examples/')


# %%
#get subject list
base_dir = '../HCP/'
#subs =sorted([sc_file[-10:-4] for sc_file in os.listdir(base_dir) if sc_file[:8] == 'weights_'])
sub = '100307'



# %%
# define options for wong-wang model
node_size = 83
mask = np.tril_indices(node_size, -1)
num_epoches = 15
batch_size = 20
step_size = 0.05
input_size = 2
tr = 0.75
repeat_size = 5


# %%
# load raw data and get SC empirical BOLD and FC
sc_file = base_dir + 'weights_' + sub + '.txt'
ts_file = base_dir + sub + '_rfMRI_REST1_LR_hpc200_clean__l2k8_sc33_ts.pkl'  # out_dir+'sub_'+sub+'simBOLD_idt.txt'#


sc = np.loadtxt(sc_file)
SC = (sc + sc.T) * 0.5
sc = np.log1p(SC) / np.linalg.norm(np.log1p(SC))

ts_pd = pd.read_pickle(ts_file)
ts = ts_pd.values
ts = ts / np.max(ts)
fc_emp = np.corrcoef(ts.T)

# %%
# prepare data structure of the model
data_mean = dataloader(ts, num_epoches, batch_size)

# %%
# get model parameters structure and define the fitted parameters by setting non-zero variance for the model
par = ParamsRWW(g=[400, 1/np.sqrt(10)], g_EE=[1.5, 1/np.sqrt(50)], g_EI =[0.8,1/np.sqrt(50)], \
                g_IE=[0.6,1/np.sqrt(50)], I_0 =[0.2, 0], std_in=[0.0,0], std_out=[0.00,0])

# %%
# call model want to fit
model = RNNRWW(node_size, batch_size, step_size, repeat_size, tr, sc, True, par)

# %%
# initial model parameters and set the fitted model parameter in Tensors
model.setModelParameters()

# %%
# create objective function
ObjFun = CostsRWW()

# %%
# call model fit
F = Model_fitting(model, data_mean, num_epoches, ObjFun)

# %%
# model training
F.train(learningrate= 0.05)

# %%
# model test with 20 window for warmup
F.test(20)

# %%
# Plot SC and fitted SC

fig, ax = plt.subplots(1, 2, figsize=(5, 4))
im0 = ax[0].imshow(sc, cmap='bwr')
ax[0].set_title('The empirical SC')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(F.model.sc_fitted.detach().numpy(), cmap='bwr')
ax[1].set_title('The fitted SC')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.show()

# %%
# Plot E I and simulated BOLD
fig, ax = plt.subplots(1, 3, figsize=(12, 8))
ax[0].plot(F.output_sim.E_test.T)
ax[0].set_title('Test: E')
ax[1].plot(F.output_sim.I_test.T)
ax[1].set_title('Test: I')
ax[2].plot(F.output_sim.bold_test.T)
ax[2].set_title('Test: BOLD')
plt.show()

# %%
# Plot the FC and the test FC
fig, ax = plt.subplots(1, 2, figsize=(5, 4))
im0 = ax[0].imshow(fc_emp, cmap='bwr')
ax[0].set_title('The empirical FC')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(np.corrcoef(F.output_sim.bold_test), cmap='bwr')
ax[1].set_title('The simulated FC')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.show()
