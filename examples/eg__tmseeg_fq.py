"""
.. _ex-tmseeg:

========================================================
Modelling TMS-EEG evoked responses
========================================================

This example shows how to organize the empirical eeg data, set-up JR model with user-defined learnable model
parameters and train model. After train how to test model with new inputs (noises) to generate simulated EEG.
Furethermore, show some analysis based on uncovered neural states from the model.

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
from whobpyt.models.linear_fq import LINEAR_FQ, ParamsLinearFreqs
from whobpyt.run import Model_fitting_fq
from whobpyt.optimization.cost_Freq import CostsFreqs
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

import mne # Neuroimaging package



# %%
# Download and load example data
data_dir = fetch_egtmseeg()

# %%
# Load EEG data 
eeg_file_name = os.path.join(data_dir, 'Subject_1_low_voltage.fif')
epoched = mne.read_epochs(eeg_file_name, verbose=False);
evoked = epoched.get_data()
eeg = np.concatenate(list(evoked), axis=1)

# %%
# Load Atlas
atlas_file_name = os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.txt')
atlas = pd.read_csv(atlas_file_name)
labels = atlas['ROI Name']
coords = np.array([atlas['R'], atlas['A'], atlas['S']]).T
conduction_velocity = 5 #in ms

# %%
# Compute the distance matrix which is used to calculate delay between regions
dist = np.zeros((coords.shape[0], coords.shape[0]))

for roi1 in range(coords.shape[0]):
  for roi2 in range(coords.shape[0]):
    dist[roi1, roi2] = np.sqrt(np.sum((coords[roi1,:] - coords[roi2,:])**2, axis=0))
    dist[roi1, roi2] = np.sqrt(np.sum((coords[roi1,:] - coords[roi2,:])**2, axis=0))


# %%
# Load the stim weights matrix which encode where to inject the external input
stim_weights = np.load(os.path.join(data_dir, 'stim_weights.npy'))
stim_weights_thr = stim_weights.copy()
labels[np.where(stim_weights_thr>0)[0]]

# %%
# Load the structural connectivity matrix
sc_file =  os.path.join(data_dir, 'Schaefer2018_200Parcels_7Networks_count.csv')
sc_df = pd.read_csv(sc_file, header=None, sep=' ')
sc = sc_df.values
sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

u_l, s_l, v_l = np.linalg.svd(sc)

# %%
# Load the leadfield matrix
lm_file = os.path.join(data_dir, 'Subject_1_low_voltage_lf.npy')
lm = np.load(lm_file)
print(lm.shape)
ki0 =stim_weights_thr[:,np.newaxis]
delays = dist/conduction_velocity

# %%
# define options for JR model: batch size integration step and sampling rate of the empirical eeg
# the number of regions in the parcellation and the number of channels

node_size = sc.shape[0]
output_size = eeg.shape[0]

sim_psd_source, sim_freqs_source = mne.time_frequency.psd_array_welch(eeg, sfreq=1000,fmin=1,fmax=50,n_fft=1900, n_per_seg=2000)

psd_train={}
psd_train['fq'] =[]
psd_train['psd'] = []
epochs_size = 1500
for i in range(epochs_size):
    fq_test_low =np.random.uniform(1,50, 500)
    psd_train['fq'].append(np.sort(fq_test_low))
    sim_psd_test = []

    for w in psd_train['fq'][i]:

        ind = np.where(sim_freqs_source > w)[0][0]
        #print(ind)
        #print(sim_freqs_source[ind-1], w, sim_freqs_source[ind])
        per = np.abs(w-sim_freqs_source[ind-1])/(np.abs(w-sim_freqs_source[ind])+np.abs(w-sim_freqs_source[ind-1]))
        #print(sim_psd_source.T[ind-1][0], ((1-per)*per*sim_psd_source.T[ind-1] +per*sim_psd_source.T[ind])[0], sim_psd_source.T[ind][0])
        sim_psd_test.append(1*((1-per)*sim_psd_source.T[ind-1] +per*sim_psd_source.T[ind]))

    psd_train['psd'].append(np.array(sim_psd_test).T)

psd_train['fq'] = np.array(psd_train['fq'])
psd_train['psd'] = np.array(psd_train['psd'])
lm_v = 0.01*np.random.randn(output_size,200)
params = ParamsLinearFreqs(mu = par(5,5, 0.5, True), g = par(100,100,1,True), eigvals= par(s_l, s_l, .1 * np.ones((node_size,1)), True),a = par(50, 50, 1, True), \
                           b = par(20,20, 0.5,True), A = par(3, 3, 0.2, True), B = par(22), C1 = par(100, 100, 1, True), C2 = par(30, 30, 1, True),c = par(0.2, 0.2, 0.001, True),
                        lm=par(lm, lm, .1 * np.ones((output_size, node_size))+lm_v, True),std_in= par(1000000))


# %%
# call model want to fit
model = LINEAR_FQ(params, node_size =node_size, output_size=output_size, sc_eigvecs =u_l, dist =dist)
# create objective function
ObjFun = CostsFreqs( model)

# %%
# call model fit
F = Model_fitting_fq(psd_train, epochs_size, model, ObjFun)
# %%


F.train()
fq_test, psd_test = F.test(sim_freqs_source)