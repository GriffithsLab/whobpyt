# -*- coding: utf-8 -*-
r"""
=================================
Evaluating CPU vs. GPU Performance
=================================

GPU Support has been added to mutiple classes in WhoBPyT. This code is for evaluating the difference in speed between CPU and GPU. The relative performance will depend on the hardware being used. 

This code is set to run on CPU by default, and then GPU can be tested by updating the device (See Importage Section).

"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Importage
# ---------------------------------------------------
#

# whobpyt stuff
import whobpyt
from whobpyt.datatypes import par, Recording
from whobpyt.models.RWWEI2 import RWWEI2_EEG_BOLD, RWWEI2_EEG_BOLD_np, RWWEI2, RWWEI2_np, ParamsRWWEI2
from whobpyt.models.BOLD import BOLD_Layer, BOLD_np, BOLD_Params
from whobpyt.models.EEG import EEG_Layer, EEG_np, EEG_Params
from whobpyt.optimization import CostsFC, CostsPSD, CostsMean, CostsFixedFC, CostsFixedPSD
from whobpyt.optimization.custom_cost_mmRWW2 import CostsmmRWWEI2
from whobpyt.run import Model_fitting, Fitting_FNGFPG, Fitting_Batch
from whobpyt.data.generators import gen_cube

# general python stuff
import time
import torch
import numpy as np
import pandas as pd

# viz stuff
import seaborn as sns
import matplotlib.pyplot as plt

print("Is cuda avaliable?")
print(torch.cuda.is_available())

device = torch.device("cpu") #Options: "cpu" or "cuda"

# %%
# Defining the Data and Parameters
# ---------------------------------------------------- 
#

# %%
# Using the Synthetic Cube Data For Demo Purposes

syntheticCubeInfo = gen_cube(device)
num_regions = 8
num_channels = 6

Con_Mtx = syntheticCubeInfo["SC"]
dist_mtx = syntheticCubeInfo["dist"]
LF_Norm = syntheticCubeInfo["LF"]
sourceFC = syntheticCubeInfo["Source FC"]
channelFC = syntheticCubeInfo["Channel FC"]

print(max(abs(torch.linalg.eig(Con_Mtx).eigenvalues)))
mask = np.eye(num_regions)
sns.heatmap(Con_Mtx.to(torch.device("cpu")), mask = mask, center=0, cmap='RdBu_r', vmin=-0.1, vmax = 0.25)
plt.title("SC of Artificial Data")

# Create a RWW Params
paramsNode = ParamsRWWEI2(num_regions)

paramsNode.J = par((0.15 * np.ones(num_regions)), fit_par = True, asLog = True) #This is a parameter that will be updated during training
paramsNode.G = par(torch.tensor(1.0), None, None, True, False, False)
paramsNode.sig = par(torch.tensor(0.01), None, None, True, False, False)
paramsNode.to(device)

#Create #EEG Params
paramsEEG = EEG_Params(torch.eye(num_regions))
paramsEEG.LF = LF_Norm.to(device)
paramsEEG.to(device)

#Create BOLD Params
paramsBOLD = BOLD_Params()
paramsBOLD.to(device)


# %%
# Training a CNMM Model - Fixed PSD with Batched Paradigm
# ------------------------------------------------------------------
#

# Simulation Length
step_size = 0.1 # Step Size in msecs
sim_len = 1500 # Simulation length in msecs
model = RWWEI2(num_regions, paramsNode, Con_Mtx, dist_mtx, step_size, sim_len, device = device)

demoPSD = torch.rand(100).to(device)
objFun = CostsFixedPSD(num_regions = num_regions, simKey = "E", sampleFreqHz = 10000, minFreq = 1, maxFreq = 100, targetValue = demoPSD, rmTransient = 5000, device = device)

empSubject = {}
num_epochs = 2
num_recordings = 1
batch_size = 50

# Create a Fitting Object
F = Fitting_Batch(model, objFun, device)

# %%
# model training
start_time = time.time()
F.train(stim = 0, empDatas = [empSubject], num_epochs = num_epochs, batch_size = batch_size, learningrate = 0.05, staticIC = False)
end_time = time.time()
print(str((end_time - start_time)/60) + " minutes")

# %%
# Plots of loss over Training
plt.plot(np.arange(1,len(F.trainingStats.loss)+1), F.trainingStats.loss)
plt.title("Total Loss over Training Epochs")


# %%
# Training a CNMM Model - Multimodal Objective with FNG-FPG Paradigm
# -------------------------------------------------------------------
#

# Simulation Length
step_size = 0.1 # Step Size in msecs
sim_len = 5000 # Simulation length in msecs
model = RWWEI2_EEG_BOLD(num_regions, num_channels, model.params, paramsEEG, paramsBOLD, Con_Mtx, dist_mtx, step_size, sim_len, device)

targetValue = torch.tensor([0.164]).to(device)
objFun = CostsmmRWWEI2(num_regions, simKey = "E", targetValue = targetValue, device = device)

# Create a Fitting Object
F = Fitting_FNGFPG(model, objFun, device)

# Training Data
empSubject = {}
empSubject['EEG_FC'] = channelFC
empSubject['BOLD_FC'] = sourceFC
num_epochs = 3
num_recordings = 1
block_len = 100 # in msec

# model training
start_time = time.time()
F.train(stim = 0, empDatas = [empSubject], num_epochs = num_epochs, block_len = block_len, learningrate = 0.05, resetIC = False)
end_time = time.time()
print(str((end_time - start_time)/60) + " minutes")

# %%
# Plots of loss over Training
plt.plot(np.arange(1,len(F.trainingStats.loss)+1), F.trainingStats.loss)
plt.title("Total Loss over Training Epochs")


# %%
# CNMM Verification Model
# ---------------------------------------------------
#
# The Multi-Modal Model

model.eeg.params.LF = model.eeg.params.LF.cpu()

val_sim_len = 20*1000 # Simulation length in msecs
model_validate = RWWEI2_EEG_BOLD_np(num_regions, num_channels, model.params, model.eeg.params, model.bold.params, Con_Mtx.detach().cpu().numpy(), dist_mtx.detach().cpu().numpy(), step_size, val_sim_len)

sim_vals, hE = model_validate.forward(external = 0, hx = model_validate.createIC(ver = 0), hE = 0)


# %%
# Plots of S_E and S_I Verification
#

plt.figure(figsize = (16, 8))
plt.title("S_E and S_I")
for n in range(num_regions):
    plt.plot(sim_vals['E'][0:10000, n], label = "S_E Node = " + str(n))
    plt.plot(sim_vals['I'][0:10000, n], label = "S_I Node = " + str(n))

plt.xlabel('Time Steps (multiply by step_size to get msec), step_size = ' + str(step_size))
plt.legend()


# %%
# Plots of EEG PSD Verification
#

sampleFreqHz = 1000*(1/step_size)
sdAxis, sdValues = CostsPSD.calcPSD(torch.tensor(sim_vals['eeg']), sampleFreqHz, minFreq = 2, maxFreq = 40)
sdAxis_dS, sdValues_dS = CostsPSD.downSmoothPSD(sdAxis, sdValues, 32)
sdAxis_dS, sdValues_dS_scaled = CostsPSD.scalePSD(sdAxis_dS, sdValues_dS)

plt.figure()
for n in range(num_channels):
    plt.plot(sdAxis_dS, sdValues_dS_scaled.detach()[:,n])
plt.xlabel('Hz')
plt.ylabel('PSD')
plt.title("Simulated EEG PSD: After Training")


# %%
# Plots of BOLD FC Verification
#

skip_trans = int(500/step_size)
sim_FC = np.corrcoef((sim_vals['bold'].T)[:,skip_trans:])

plt.figure(figsize = (8, 8))
plt.title("Simulated BOLD FC: After Training")
mask = np.eye(num_regions)
sns.heatmap(sim_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0)