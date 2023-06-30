# -*- coding: utf-8 -*-
r"""
=================================
Fitting S_E Mean to 0.164 using default RWW Parameters
=================================

What is being modeled:

- Created a Sphere'd Cube (chosen points on cube projected onto radius = 1 sphere), so that regions were more evently distributed. All corners of cube chosen as regions, thus there are 8 regions. 

- EEG channels located on the center of each face of the cube. Thus there are 6 EEG channels.

- Added some randomness to initial values - to decorrelate the signals a bit. Looking for FC matrix to look similar to SC matrix.

"""  


# sphinx_gallery_thumbnail_number = 1

# %%
# Importage
# ---------------------------------------------------
#

# whobpyt stuff
import whobpyt
from whobpyt.datatypes import par, Recording
from whobpyt.models.RWW2 import mmRWW2, mmRWW2_np, RWW2, RWW2_np, ParamsRWW2
from whobpyt.models.BOLD import BOLD_Layer, BOLD_np, BOLD_Params
from whobpyt.models.EEG import EEG_Layer, EEG_np, EEG_Params
from whobpyt.optimization import CostsFC, CostsPSD, CostsMean
from whobpyt.run import Model_fitting

# general python stuff
import torch
import numpy as np
import pandas as pd

# viz stuff
import seaborn as sns
import matplotlib.pyplot as plt


# %%
# Defining Model Parameters
# ---------------------------------------------------
#

num_regions = 8
num_channels = 6

# Simulation Length
step_size = 0.1 # Step Size in msecs
sim_len = 1500 # Simulation length in msecs

skip_trans = int(500/step_size)

# Initial Conditions
S_E = 0.6; S_I = 0.1; x = 0.0000; f = 2.4286; v = 1.3283; q = 0.6144 # x,f,v,q might be choosen for different initial S_E
init_state = torch.tensor([[S_E, S_I, x, f, v, q]]).repeat(num_regions, 1)

# Add randomness
init_state = init_state + torch.randn_like(init_state)/30 # Randomizing initial values

# Create a RWW Params
paramsNode = ParamsRWW2(num_regions)

#Create #EEG Params
paramsEEG = EEG_Params(torch.eye(num_regions))

#Create BOLD Params
paramsBOLD = BOLD_Params()


# %%
# Further Adjusting Parameters for Network
# ---------------------------------------------------
#

paramsNode.J = par((0.15  * np.ones(num_regions)), fit_par = True, asLog = True) #This is a parameter that will be updated during training


# %%
# Generating a physically possible (in 3D Space) Structural Connectivity Matrix
# ---------------------------------------------------
#
# First, get corner points on a cube and project onto a sphere
square_points = torch.tensor([[1.,1.,1.],
                              [-1.,1.,1.],
                              [1.,-1.,1.],
                              [-1.,-1.,1.],
                              [1.,1.,-1.],
                              [-1.,1.,-1.],
                              [1.,-1.,-1.],
                              [-1.,-1.,-1.]])
sphere_points = square_points / torch.sqrt(torch.sum(torch.square(square_points), axis = 1)).repeat(3, 1).t()

# Second, find the distance between all pairs of points
dist_mtx = torch.zeros(num_regions, num_regions)
for x in range(num_regions):
    for y in range(num_regions):
        dist_mtx[x,y]= torch.linalg.norm(sphere_points[x,:] - sphere_points[y,:])

# Third, Structural Connectivity defined to be 1/dist and remove self-connection values
SC_mtx = 1/dist_mtx
for z in range(num_regions):
    SC_mtx[z,z] = 0.0

# Fourth, Normalize the matrix
SC_mtx_norm = (1/torch.linalg.matrix_norm(SC_mtx, ord = 2)) * SC_mtx
Con_Mtx = SC_mtx_norm


# Blah

print(max(abs(torch.linalg.eig(SC_mtx_norm).eigenvalues)))
mask = np.eye(num_regions)
sns.heatmap(Con_Mtx, mask = mask, center=0, cmap='RdBu_r', vmin=-0.1, vmax = 0.25)
plt.title("SC of Artificial Data")



# %%
# Generating a Lead Field Matrix
# ---------------------------------------------------
#
# Placing an EEG Electrode in the middle of each cube face. 
# Then electrode is equally distance from four courner on cube face squre.
# Assume no signal from further four points. 

Lead_Field = torch.tensor([[1,1,0,0,1,1,0,0],
                           [1,1,1,1,0,0,0,0],
                           [0,1,0,1,0,1,0,1],
                           [0,0,0,0,1,1,1,1],
                           [1,0,1,0,1,0,1,0],
                           [0,0,1,1,0,0,1,1]], dtype = torch.float)
LF_Norm = (1/torch.linalg.matrix_norm(Lead_Field, ord = 2)) * Lead_Field

paramsEEG.LF = LF_Norm


# %%
# Generating a "Connectivity Matrix" for Channel Space
# ---------------------------------------------------
#
# Generating a physically possible (in 3D Space) "Channel" Connectivity Matrix
# That is a theoretical matrix for the EEG SC to be fit to

# First, get face points on a cube and project onto a sphere
LF_square_points = torch.tensor([[0.,1.,0.],
                                 [0.,0.,1.],
                                 [-1.,0.,0.],
                                 [0.,0.,-1.],
                                 [1.,0.,0.],
                                 [0.,-1.,0.]])
# Note: this does nothing as the points are already on the r=1 sphere
LF_sphere_points = LF_square_points / torch.sqrt(torch.sum(torch.square(LF_square_points), axis = 1)).repeat(3, 1).t()


# Second, find the distance between all pairs of channel points
LF_dist_mtx = torch.zeros(num_channels, num_channels)
for x in range(num_channels):
    for y in range(num_channels):
        LF_dist_mtx[x,y]= torch.linalg.norm(LF_sphere_points[x,:] - LF_sphere_points[y,:])

# Third, Structural Connectivity defined to be 1/dist and remove self-connection values
LF_SC_mtx = 1/LF_dist_mtx
for z in range(num_channels):
    LF_SC_mtx[z,z] = 0.0

# Fourth, Normalize the matrix
LF_SC_mtx_norm = (1/torch.linalg.matrix_norm(LF_SC_mtx, ord = 2)) * LF_SC_mtx
LF_Con_Mtx = LF_SC_mtx_norm



# %%
# Defining the CNMM Model
# ---------------------------------------------------
#
# The Multi-Modal Model


model = mmRWW2(num_regions, num_channels, paramsNode, paramsEEG, paramsBOLD, Con_Mtx, dist_mtx, step_size, sim_len)
#model.track_params = ['J']

# %%
# Defining the Objective Function
# ---------------------------------------------------
#
# Written in such as way as to be able to adjust the relative importance of components that make up the objective function.
# Also, written in such a way as to be able to track and plot indiviual components losses over time. 

class mmObjectiveFunction():
    def __init__(self):
        self.simKey = "E"
    
        # Weights of Objective Function Components
        self.S_E_mean_weight = 1
        self.S_I_mean_weight = 0 # Not Currently Used
        self.EEG_PSD_weight = 0 # Not Currently Used
        self.EEG_FC_weight = 0 # Not Currently Used
        self.BOLD_PSD_weight = 0 # Not Currently Used
        self.BOLD_FC_weight = 0 # Not Currently Used
        
        # Functions of the various Objective Function Components
        self.S_E_mean = CostsMean(num_regions, simKey = "E", targetValue = torch.tensor([0.164]))
        #self.S_I_mean = CostsMean(...) # Not Currently Used
        #self.EEG_PSD = CostsPSD(num_channels, varIdx = 0, sampleFreqHz = 1000*(1/step_size), targetValue = targetEEG)
        #self.EEG_FC = CostsFC(...) # Not Currently Used
        #self.BOLD_PSD = CostsPSD(...) # Not Currently Used
        #self.BOLD_FC = CostsFC(num_regions, varIdx = 4, targetValue = SC_mtx_norm)
                
    def loss(self, node_history, EEG_history, BOLD_history, temp, returnLossComponents = False):
        # sim, ts_window, self.model, next_window
        
        S_E_mean_loss = self.S_E_mean.calcLoss(node_history) 
        S_I_mean_loss = torch.tensor([0]) #self.S_I_mean.calcLoss(node_history)
        EEG_PSD_loss = torch.tensor([0]) #self.EEG_PSD.calcLoss(EEG_history) 
        EEG_FC_loss = torch.tensor([0]) #self.EEG_FC.calcLoss(EEG_history)
        BOLD_PSD_loss = torch.tensor([0]) #self.BOLD_PS.calcLoss(BOLD_history)
        BOLD_FC_loss = torch.tensor([0]) #self.BOLD_FC.calcLoss(BOLD_history)
                
        totalLoss = self.S_E_mean_weight*S_E_mean_loss + self.S_I_mean_weight*S_I_mean_loss \
                  + self.EEG_PSD_weight*EEG_PSD_loss   + self.EEG_FC_weight*EEG_FC_loss \
                  + self.BOLD_PSD_weight*BOLD_PSD_loss + self.BOLD_FC_weight*BOLD_FC_loss
                 
        if returnLossComponents:
            return totalLoss, (S_E_mean_loss.item(), S_I_mean_loss.item(), EEG_PSD_loss.item(), EEG_FC_loss.item(), BOLD_PSD_loss.item(), BOLD_FC_loss.item())
        else:
            return totalLoss


# %%
# Training The Model
# ---------------------------------------------------
#


ObjFun = mmObjectiveFunction()


randData1 = np.random.rand(8, 15000)
randData2 = np.random.rand(8, 15000)
num_epochs = 3
num_recordings = 2
TPperWindow = 15000

print(randData1.shape)
randTS1 = Recording(randData1, step_size)
randTS2 = Recording(randData2, step_size)

# call model fit
F = Model_fitting(model, ObjFun)

# %%
# model training
F.train(u = 0, empRecs = [randTS1, randTS2], num_epochs = num_epochs, TPperWindow = TPperWindow, learningrate = 0.1)

# %%
# Plots of loss over Training
plt.plot(np.arange(1,len(F.trainingStats.loss)+1), F.trainingStats.loss)
plt.title("Total Loss over Training Epochs")

# %%
# Plots of J values over Training
plt.plot(F.trainingStats.fit_params['J'])
plt.title("J_{i} Values Changing Over Training Epochs")


# %%
# Model Simulation
# ---------------------------------------------------
#
F.simulate(u = 0, numTP = randTS1.length)


# %%
# Plots of S_E and S_I
plt.figure(figsize = (16, 8))
plt.title("S_E and S_I")
for n in range(num_regions):
    plt.plot(F.lastRec['E'].npTS()[n,:], label = "S_E Node = " + str(n))
    plt.plot(F.lastRec['I'].npTS()[n,:], label = "S_I Node = " + str(n))

plt.xlabel('Time Steps (multiply by step_size to get msec), step_size = ' + str(step_size))
plt.legend()



# %%
# Plots of EEG PSD
#

sampleFreqHz = 1000*(1/step_size)
sdAxis, sdValues = CostsPSD.calcPSD(torch.tensor(F.lastRec['eeg'].npTS().T), sampleFreqHz, minFreq = 2, maxFreq = 40)
sdAxis_dS, sdValues_dS = CostsPSD.downSmoothPSD(sdAxis, sdValues, 32)
sdAxis_dS, sdValues_dS_scaled = CostsPSD.scalePSD(sdAxis_dS, sdValues_dS)

plt.figure()
plt.plot(sdAxis_dS, sdValues_dS_scaled.detach())
plt.xlabel('Hz')
plt.ylabel('PSD')
plt.title("Simulated EEG PSD: After Training")



# %%
# Plots of BOLD FC
#

sim_FC = np.corrcoef(F.lastRec['bold'].npTS()[:,skip_trans:])

plt.figure(figsize = (8, 8))
plt.title("Simulated BOLD FC: After Training")
mask = np.eye(num_regions)
sns.heatmap(sim_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0)



# %%
# CNMM Validation Model
# ---------------------------------------------------
#
# The Multi-Modal Model

val_sim_len = 20*1000 # Simulation length in msecs
model_validate = mmRWW2_np(num_regions, num_channels, model.params, model.eeg.params, model.bold.params, Con_Mtx.detach().numpy(), dist_mtx.detach().numpy(), step_size, val_sim_len)

sim_vals, hE = model_validate.forward(external = 0, hx = model_validate.createIC(ver = 0), hE = 0)


# %%
# Plots of S_E and S_I Validation
#

plt.figure(figsize = (16, 8))
plt.title("S_E and S_I")
for n in range(num_regions):
    plt.plot(sim_vals['E'], label = "S_E Node = " + str(n))
    plt.plot(sim_vals['I'], label = "S_I Node = " + str(n))

plt.xlabel('Time Steps (multiply by step_size to get msec), step_size = ' + str(step_size))
plt.legend()


# %%
# Plots of EEG PSD Validation
#

sampleFreqHz = 1000*(1/step_size)
sdAxis, sdValues = CostsPSD.calcPSD(torch.tensor(sim_vals['eeg']), sampleFreqHz, minFreq = 2, maxFreq = 40)
sdAxis_dS, sdValues_dS = CostsPSD.downSmoothPSD(sdAxis, sdValues, 32)
sdAxis_dS, sdValues_dS_scaled = CostsPSD.scalePSD(sdAxis_dS, sdValues_dS)

plt.figure()
plt.plot(sdAxis_dS, sdValues_dS_scaled.detach())
plt.xlabel('Hz')
plt.ylabel('PSD')
plt.title("Simulated EEG PSD: After Training")


# %%
# Plots of BOLD FC Validation
#

sim_FC = np.corrcoef((sim_vals['bold'].T)[:,skip_trans:])

plt.figure(figsize = (8, 8))
plt.title("Simulated BOLD FC: After Training")
mask = np.eye(num_regions)
sns.heatmap(sim_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0)