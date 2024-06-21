"""
.. _ex-momi2023-rep:

========================================================
Replicating Momi et al. (2023): TMS-evoked Responses
========================================================

This script replicates the findings of the paper:

Momi, D., Wang, Z., Griffiths, J.D. (2023).
 "TMS-evoked responses are driven by recurrent large-scale network dynamics."
  eLife, [doi: 10.7554/eLife.83232](https://elifesciences.org/articles/83232)

*The effect of two anatomical connectivity-based lesion 
strategies (random vs targeted) and time of damage (20ms: blue; 50ms: orange; 100ms: green) on TMS-EEG dynamics for one
representative subject. Overall, targeted attack (left column) significantly compromised the propagation of the TMS-evoked signal
compared to the random attack (right column) condition. Moreover, the EEG dynamics were significantly affected by early (20ms:
blue and 50ms: orange) compared to late (100ms: green) virtual lesions.*

The code includes data fetching, model fitting, and result visualization based on the methods presented in the paper.

**PLACEHOLDER -- What findings specifically are being replicated?**
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Setup
# --------------------------------------------------

# Importage:

# os stuff
import os
import sys
sys.path.append('..')


# whobpyt stuff
import whobpyt
from whobpyt.datatypes import Parameter as par, Timeseries
from whobpyt.models.jansen_rit import JansenRitModel,JansenRitParams
from whobpyt.run import ModelFitting
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

download_data = True 
url = 'https://drive.google.com/drive/folders/1dpyyfJl9wjTrWVo5lqOmB8HRhD3irjNj?usp=drive_link'
if download_data: gdown.download_folder(url, quiet=True)
data_dir = os.path.abspath('eg__replicate_Momi2023_data')

# %%
# Load EEG data from a file
file_name = os.path.join(data_dir, 'Subject_1_low_voltage.fif')
epoched = mne.read_epochs(file_name, verbose=False);
evoked = epoched.average()

# %%
# Load Atlas
url = 'https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
atlas = pd.read_csv(url)

labels = atlas['ROI Name']
coords = np.array([atlas['R'], atlas['A'], atlas['S']]).T
# why choice of value 5?
conduction_velocity = 5 #in ms

# %%
# Compute the Euclidean distance matrix
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

# %%
# Load the leadfield matrix
# **PLACEHOLDER: Commented out `delays = dist/conduction_velocity` as it is unused**
lm = os.path.join(data_dir, 'Subject_1_low_voltage_lf.npy')
ki0 =stim_weights_thr[:,np.newaxis]
#UNUSED
# delays = dist/conduction_velocity

# %%
# Define options for JR model \
# **PLACEHOLDER: Start and end times are chosen for X reasons?**
eeg_data = evoked.data.copy()
time_start = np.where(evoked.times==-0.1)[0][0]
time_end = np.where(evoked.times==0.3)[0][0]
eeg_data = eeg_data[:,time_start:time_end]/np.abs(eeg_data).max()*4
node_size = sc.shape[0]
output_size = eeg_data.shape[0]
batch_size = 20
step_size = 0.0001
num_epochs = 2 #2 # num_epochs = 20
tr = 0.001 #needs to be renamed to sampling_rate according to JansenRitModel documentation
state_size = 6
base_batch_num = 20
time_dim = 400
state_size = 6
base_batch_num = 20
hidden_size = int(tr/step_size)


# %%
# Prepare data structure of the model using `Timeseries <https://github.com/GriffithsLab/whobpyt/blob/dev/whobpyt/datatypes/timeseries.py>`_, 
# which is responsible for holding the empirical and simulated data. It is the format expected by the visualization function of whobpyt. 
data_mean = Timeseries(eeg_data, num_epochs, batch_size)

# %%
# Get model parameters structure and define the fitted parameters by setting non-zero variance for the model using
# `JansenRitParams <https://github.com/GriffithsLab/whobpyt/blob/main/whobpyt/models/jansen_rit/jansen_rit.py#L42>`_. \
# **PLACEHOLDER: What's g_f, g_b, k0, lm? No definition given in class**
lm = np.zeros((output_size,200))
lm_v = np.zeros((output_size,200))
params = JansenRitParams(A = par(3.25), 
                         a= par(100,100, 2, True), 
                         B = par(22), 
                         b = par(50, 50, 1, True),
                         g=par(500,500,2, True), 
                         g_f=par(10,10,1, True), 
                         g_b=par(10,10,1, True),
                         c1 = par(135, 135, 1, True), 
                         c2 = par(135*0.8, 135*0.8, 1, True), 
                         c3 = par(135*0.25, 135*0.25, 1, True),
                         c4 = par(135*0.25, 135*0.25, 1, True), 
                         std_in= par(np.log(10), np.log(10), .1, True, True), 
                         vmax= par(5), 
                         v0=par(6), 
                         r=par(0.56),
                         y0=par(-2, -2, 1/4, True),
                         mu = par(np.log(1.5),
                                  np.log(1.5), .1, True, True, lb=0.1), 
                         k =par(5., 5., 0.2, True, lb=1), 
                         k0=par(0),
                         cy0 = par(50, 50, 1, True), 
                         ki=par(ki0), 
                         lm=par(lm, lm, 1 * np.ones((output_size, node_size))+lm_v, True)
                         )


# %%
# Call model you want to fit, in this case `JansenRitModel <https://github.com/GriffithsLab/whobpyt/blob/main/whobpyt/models/jansen_rit/jansen_rit.py#L113>`_
model = JansenRitModel(params, 
                       node_size=node_size, 
                       TRs_per_window=batch_size, 
                       step_size=step_size, 
                       output_size=output_size, 
                       tr=tr, 
                       sc=sc, 
                       lm=lm, 
                       dist=dist, 
                       use_fit_gains=True, 
                       use_fit_lfm = False)


# %%
# Create objective function using `CostsJR <https://github.com/GriffithsLab/whobpyt/blob/main/whobpyt/optimization/custom_cost_JR.py#L14>`_ \
# **PLACEHOLDER: Does objective mean loss function? If so point or insert here explanation of theory behind it**
ObjFun = CostsJR(model)


# %%
# Call model fit using `ModelFitting <https://github.com/GriffithsLab/whobpyt/blob/main/whobpyt/run/model_fitting.py#L18>`_
F = ModelFitting(model, ObjFun)

# %%
# Model training \
# **PLACEHOLDER: what is the reason for the values being replaced below?**
u = np.zeros((node_size,hidden_size,time_dim))
u[:,:,80:120]= 1000
F.train(u=u, empRecs = [data_mean], num_epochs = num_epochs, TPperWindow = batch_size)

# %%
# Quick test run with 2 epochs
F.evaluate(u = u, empRec = data_mean, TPperWindow = batch_size, base_window_num = 20)


# %%
# Load in a previously completed ModelFitting results object run with 20 epochs as a comparison
full_run_fname = os.path.join(data_dir, 'Subject_1_low_voltage_fittingresults_stim_exp.pkl')
F = pickle.load(open(full_run_fname, 'rb'))
F.evaluate(u = u, empRec = data_mean, TPperWindow = batch_size, base_window_num = 20)


# %%
# Plot the original and simulated EEG data \
# **PLACEHOLDER: Commented out duplicate code** \
# ```
# epoched = mne.read_epochs(file_name, verbose=False); \
# evoked = epoched.average()
# ```
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
#
# The plot above shows the original EEG data and the simulated EEG data using the fitted Jansen-Rit model.
# The simulated data closely resembles the original EEG data, indicating that the model fitting was successful.
# Peak locations extracted from different time intervals are marked on the plots, demonstrating the model's ability
# to capture key features of the EEG signal.

# %%
# References
# ---------------------------------------------------
#
# Momi, D., Wang, Z., Griffiths, J.D. (2023). "TMS-evoked responses are driven by recurrent large-scale network dynamics." eLife, 10.7554/eLife.83232. https://doi.org/10.7554/eLife.83232



