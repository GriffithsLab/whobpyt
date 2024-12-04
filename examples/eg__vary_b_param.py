"""
.. _ex-momi2023-part2:

========================================================
Exploring the Effects of the Inhibitory Rate Constant Parameter (b)
========================================================

This script replicates the findings of the paper :footcite:`MomiEtAl2023`:

Momi, D., Wang, Z., Griffiths, J.D. (2023).
 "TMS-evoked responses are driven by recurrent large-scale network dynamics."
# eLife, [doi: 10.7554/eLife.83232](https://elifesciences.org/articles/83232)

This code loads up a previously-fit whobpyt model, varies a specific model parameter (the inhibitory rate constant; b), and simulates TEPs to visualize what effect this model parameter has on the output.


"""

# sphinx_gallery_thumbnail_number = 1
#
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
url = 'https://drive.google.com/drive/folders/1DTdF_xR78DxB6kzxqY3SVYBAcdU9IkAB?usp=drive_link'
if download_data: gdown.download_folder(url, quiet=True)
data_dir = os.path.abspath('eg__tmseeg_data')


# # %%
# # load in a previously completed model fitting results object
full_run_fname = os.path.join(data_dir, 'Subject_1_low_voltage_fittingresults_stim_exp.pkl')
F = pickle.load(open(full_run_fname, 'rb'))


# %%
# Define relevant variables for whobpyt fititng/simuations

# Load EEG data from a file
file_name = os.path.join(data_dir, 'Subject_1_low_voltage.fif')
epoched = mne.read_epochs(file_name, verbose=False);
evoked = epoched.average()


# %%
# define options for JR model
eeg_data = evoked.data.copy()
time_start = np.where(evoked.times==-0.1)[0][0]
time_end = np.where(evoked.times==0.3)[0][0]
eeg_data = eeg_data[:,time_start:time_end]/np.abs(eeg_data).max()*4
node_size = F.model.node_size
output_size = eeg_data.shape[0]
batch_size = 20
step_size = 0.0001
num_epochs = 20 #2 # num_epochs = 20
tr = 0.001
state_size = 6
base_batch_num = 20
time_dim = 400
state_size = 6
base_batch_num = 20
hidden_size = int(tr/step_size)


# %%
# prepare data structure of the model
data_mean = Timeseries(eeg_data, num_epochs, batch_size)

# stimulation info
u = np.zeros((node_size,hidden_size,time_dim))
u[:,:,80:120]= 1000


# %%

# Run simulation 
F.evaluate(u = u, empRec = data_mean, TPperWindow = batch_size, base_window_num = 20)

# Visualizng the original fit
ts_args = dict(xlim=[-0.1,0.3])
ch, peak_locs1 = evoked.get_peak(ch_type='eeg', tmin=-0.05, tmax=0.01)
ch, peak_locs2 = evoked.get_peak(ch_type='eeg', tmin=0.01, tmax=0.02)
ch, peak_locs3 = evoked.get_peak(ch_type='eeg', tmin=0.03, tmax=0.05)
ch, peak_locs4 = evoked.get_peak(ch_type='eeg', tmin=0.07, tmax=0.15)
ch, peak_locs5 = evoked.get_peak(ch_type='eeg', tmin=0.15, tmax=0.20)
times = [peak_locs1, peak_locs2, peak_locs3, peak_locs4, peak_locs5]

simulated_EEG_st = evoked.copy()
simulated_EEG_st.data[:,time_start:time_end] = F.lastRec['eeg'].npTS()
times = [peak_locs1, peak_locs2, peak_locs3, peak_locs4, peak_locs5]
simulated_joint_st = simulated_EEG_st.plot_joint(ts_args=ts_args, times=times)

# %%
# now we want to vary paramters and simulate again.

# looking at original value of b
print(F.model.params.b.val)

# defining a range of b to vary and simulate
b_range = np.linspace(45, 55, 4)


# %% 
# Defining a range for the b paramter to explore
b_range = np.linspace(45, 55, 5)
# testing all ranges:
sims_dict = {}
for n, new_b in enumerate(b_range):
    # changing parameter in model
    F.model.params.b = par(new_b, new_b, 1, True)

    # evaluate the model
    F.evaluate(u = u, empRec = data_mean, TPperWindow = batch_size, base_window_num = 20)

    # plotting and saving new simulations
    simulated_EEG_st.data[:,time_start:time_end] = F.lastRec['eeg'].npTS()
    simulated_joint_st = simulated_EEG_st.plot_joint(ts_args=ts_args, times=times, title=f'b param = {new_b}')
    sims_dict[str(new_b)] = simulated_EEG_st.copy().crop(tmin=0, tmax=0.2)

# %%
# comparing conditions
fig, ax = plt.subplots(figsize=(12,4))
mne.viz.plot_compare_evokeds(sims_dict, picks='eeg', combine='gfp', axes=ax, cmap='viridis',);
fig.show()

# %%
# Results Description
# ---------------------------------------------------
#
#
# Here we replicate the results of Momi et al. 2023 (Fig.  5D). As the inhibitory synaptic rate constant b increases (or equivalently, the time constant decreases), 
# we observe an increase in the amplitude of the first, early, and local TEP components; and a decrease of the second, 
# late, and global TEP components.