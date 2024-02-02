# -*- coding: utf-8 -*-
r"""
=================================
Fitting HGF
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
from whobpyt.datatypes import par, Recording
from whobpyt.models.HGF import HGF, ParamsHGF
from whobpyt.optimization.custom_cost_HGF import CostsHGF
from whobpyt.run import Model_fitting

import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

u = np.concatenate([bernoulli.rvs(0.5, size=500), bernoulli.rvs(0.8, size =200), bernoulli.rvs(0.2, size =200),  \
                    bernoulli.rvs(0.8, size =200), bernoulli.rvs(0.2, size =200),  bernoulli.rvs(0.8, size =200),\
                    bernoulli.rvs(0.2, size =200),  bernoulli.rvs(0.5, size =500)])

### make a fake data target (convert from 0-1 to -2 +2)
u_ne = 2*np.ones_like(u)
u_ne[u==0]=-2

num_epochs = 5
TPperWindow = 1
node_size = 1
data_mean = dataloader(np.array([u_ne]).T, num_epochs, TPperWindow)
#data_mean = Recording(eeg_data, EEGstep) #dataloader(eeg_data.T, num_epochs, batch_size)

# %%
# get model parameters structure and define the fitted parameters by setting non-zero variance for the model

params = ParamsHGF( omega_2=par(0.1),\
                   omega_3=par(0.01), \
                   x2mean = par(-4,-4, 1, True),\
                    c = par(np.log(1),np.log(1.1),0.1, True, True),\
                    g_x2_x3 = par(1),\
                    g_x3_x2 = par(np.log(1),np.log(1.1),0.1, True, True),\
                    deca2 = par(0.0), \
                    deca3 = par(0.0), kappa = par(1))#par(np.log(1),np.log(1),0.1, True, True))

model = HGF(params, TRperWindow = TPperWindow, step_size=.05)
# %%
# create objective function
ObjFun = CostsHGF(model)

# %%
# call model fit
F = Model_fitting(model, ObjFun)


F.train(u = 0, empRec = data_mean, num_epochs = num_epochs, TPperWindow = TPperWindow, warmupWindow=0, learningrate=0.05)

# %%
# Plots of loss over Training
plt.plot(np.array(F.trainingStats.loss).T)
plt.title("Total Loss over Training Epochs")

fig, ax = plt.subplots(1,2)
ax[0].plot(F.trainingStats.outputs['x1_training'].T)
ax[0].set_title("simulated output")
ax[1].plot(F.trainingStats.states['training'][0,0].T)
ax[1].set_title("level 2 x2")
plt.show()

fig, ax = plt.subplots(1,2)
"""ax[0].plot(F.trainingStats.fit_params['g_x2_x3'])
ax[0].set_title('level 2 to 3')"""
ax[1].plot(F.trainingStats.fit_params['g_x3_x2'])
ax[0].set_title('level 3 to 2')
plt.show()