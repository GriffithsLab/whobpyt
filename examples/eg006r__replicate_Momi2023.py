# -*- coding: utf-8 -*-
r"""
=================================
Replicate Momi et al. (2023): TMS-evoked Responses
===========================================

This script replicates the findings of the paper:

Momi, D., Wang, Z., Griffiths, J.D. (2023).
"TMS-evoked responses are driven by recurrent large-scale network dynamics."
eLife, [doi: 10.7554/eLife.83232](https://elifesciences.org/articles/83232)

The code includes data fetching, model fitting, and result visualization based on the methods presented in the paper.

"""


# sphinx_gallery_thumbnail_number = 1
#
# %%
# Importage
# --------------------------------------------------

# whobpyt stuff
import whobpyt
from whobpyt.datatypes import par, Recording
from whobpyt.models.JansenRit import RNNJANSEN, ParamsJR
from whobpyt.run import Model_fitting
from whobpyt.optimization.custom_cost_JR import CostsJR

# python stuff
import numpy as np
import pandas as pd
import scipy.io
import gdown
import pickle
import warnings
warnings.filterwarnings('ignore')

#neuroimaging packages
import mne

# viz stuff
import matplotlib.pyplot as plt



# %%
# Download and load necessary data for the example
url='https://drive.google.com/drive/folders/1Qu-JyZc3-SL-Evsystg4D-DdpsGU4waB?usp=sharing'
gdown.download_folder(url, quiet=True)


# %%
# Load EEG data from a file
file_name = './data/Subject_1_low_voltage.fif'
epoched = mne.read_epochs(file_name, verbose=False);
evoked = epoched.average()

# %%
# Load Atlas
url = 'https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
atlas = pd.read_csv(url)
labels = atlas['ROI Name']
coords = np.array([atlas['R'], atlas['A'], atlas['S']]).T
conduction_velocity = 5 #in ms

# %%
# Compute the distance matrix
dist = np.zeros((coords.shape[0], coords.shape[0]))

for roi1 in range(coords.shape[0]):
  for roi2 in range(coords.shape[0]):
    dist[roi1, roi2] = np.sqrt(np.sum((coords[roi1,:] - coords[roi2,:])**2, axis=0))
    dist[roi1, roi2] = np.sqrt(np.sum((coords[roi1,:] - coords[roi2,:])**2, axis=0))


# %%
# Load the stim weights matrix which encode where to inject the external input
stim_weights = np.load('./data/stim_weights.npy')
stim_weights_thr = stim_weights.copy()
labels[np.where(stim_weights_thr>0)[0]]

# %%
# Load the structural connectivity matrix
sc_file =  './data/Schaefer2018_200Parcels_7Networks_count.csv'
sc_df = pd.read_csv(sc_file, header=None, sep=' ')
sc = sc_df.values
sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

# %%
# Load the leadfield matrix
lm = np.load('./data/Subject_1_low_voltage_lf.npy')
ki0 =stim_weights_thr[:,np.newaxis]
delays = dist/conduction_velocity

# %%
# define options for JR model
eeg_data = evoked.data.copy()
time_start = np.where(evoked.times==-0.1)[0][0]
time_end = np.where(evoked.times==0.3)[0][0]
eeg_data = eeg_data[:,time_start:time_end]/np.abs(eeg_data).max()*4
node_size = sc.shape[0]
output_size = eeg_data.shape[0]
batch_size = 20
step_size = 0.0001
num_epoches = 120
tr = 0.001
state_size = 6
base_batch_num = 20
time_dim = 400
state_size = 6
base_batch_num = 20
hidden_size = int(tr/step_size)


# %%
# prepare data structure of the model
data_mean = Recording(eeg_data, num_epoches, batch_size)

# %%
# get model parameters structure and define the fitted parameters by setting non-zero variance for the model
lm = np.zeros((output_size,200))
lm_v = np.zeros((output_size,200))
params = ParamsJR(A = par(3.25), a= par(100,100, 2, True, True), B = par(22), b = par(50, 50, 1, True, True),
               g=par(500,500,2, True, True), g_f=par(10,10,1, True, True), g_b=par(10,10,1, True, True),
               c1 = par(135, 135, 1, True, True), c2 = par(135*0.8, 135*0.8, 1, True, True), c3 = par(135*0.25, 135*0.25, 1, True, True),
               c4 = par(135*0.25, 135*0.25, 1, True, True), std_in= par(0,0, 1, True, True), vmax= par(5), v0=par(6), r=par(0.56),
               y0=par(-2, -2, 1/4, True, True),mu = par(1., 1., 0.4, True, True), k =par(5., 5., 0.2, True, True), k0=par(0),
               cy0 = par(50, 50, 1, True, True), ki=par(ki0), lm=par(lm, lm, 1 * np.ones((output_size, node_size))+lm_v, True, True))


# %%
# call model want to fit
model = RNNJANSEN(node_size, batch_size, step_size, output_size, tr, sc, lm, dist, True, False, params)


# create objective function
ObjFun = CostsJR(model)


# %%
# call model fit
F = Model_fitting(model, ObjFun)

# %%
# model training
u = np.zeros((node_size,hidden_size,time_dim))
u[:,:,80:120]= 1000
F.train(u=u, empRecs = [data_mean], num_epochs = num_epoches, TPperWindow = batch_size)

# %%
# model test with 20 window for warmup
F.evaluate(u = u, empRec = data_mean, TPperWindow = batch_size, base_window_num = 20)

# filename = 'Subject_1_low_voltage_fittingresults_stim_exp.pkl'
# with open(filename, 'wb') as f:
# 	pickle.dump(F, f)

# %%
# Plot the original and simulated EEG data
epoched = mne.read_epochs(file_name, verbose=False);
evoked = epoched.average()
ts_args = dict(xlim=[-0.1,0.3])
ch, peak_locs1 = evoked.get_peak(ch_type='eeg', tmin=-0.05, tmax=0.01)
ch, peak_locs2 = evoked.get_peak(ch_type='eeg', tmin=0.01, tmax=0.02)
ch, peak_locs3 = evoked.get_peak(ch_type='eeg', tmin=0.03, tmax=0.05)
ch, peak_locs4 = evoked.get_peak(ch_type='eeg', tmin=0.07, tmax=0.15)
ch, peak_locs5 = evoked.get_peak(ch_type='eeg', tmin=0.15, tmax=0.20)
times = [peak_locs1, peak_locs2, peak_locs3, peak_locs4, peak_locs5]
plot = evoked.plot_joint(ts_args=ts_args, times=times);


simulated_EEG_st = evoked.copy()
simulated_EEG_st.data[:,time_start:time_end] = F.lastRec['eeg'].npTS()
times = [peak_locs1, peak_locs2, peak_locs3, peak_locs4, peak_locs5]
simulated_joint_st = simulated_EEG_st.plot_joint(ts_args=ts_args, times=times)


# %%
# Results Description
# ---------------------------------------------------
#

# The plot above shows the original EEG data and the simulated EEG data using the fitted Jansen-Rit model.
# The simulated data closely resembles the original EEG data, indicating that the model fitting was successful.
# Peak locations extracted from different time intervals are marked on the plots, demonstrating the model's ability
# to capture key features of the EEG signal.

# %%
# Reference:
# Momi, D., Wang, Z., Griffiths, J.D. (2023). "TMS-evoked responses are driven by recurrent large-scale network dynamics."
# eLife, 10.7554/eLife.83232. https://doi.org/10.7554/eLife.83232
