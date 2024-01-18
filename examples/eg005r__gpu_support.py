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
# os stuff
import os
import sys
sys.path.append('..')

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
# Training a CNMM Model - Multimodal Objective with FNG-FPG Paradigm
# -------------------------------------------------------------------
#

# Simulation Length


# %%
# Plots of loss over Training
plt.plot(np.arange(1,len(F.trainingStats.loss)+1), F.trainingStats.loss)
plt.title("Total Loss over Training Epochs")
