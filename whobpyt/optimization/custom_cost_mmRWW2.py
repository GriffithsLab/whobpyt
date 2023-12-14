import numpy as np
import torch
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.optimization import CostsMean
from whobpyt.optimization import CostsPSD
from whobpyt.optimization import CostsFixedFC

class CostsmmRWWEI2(AbstractLoss):
    def __init__(self, num_regions, simKey, targetValue, device = torch.device('cpu')):  
        # Defining the Objective Function
        # ---------------------------------------------------
        # Written in such as way as to be able to adjust the relative importance of components that make up the objective function.
        # Also, written in such a way as to be able to track and plot indiviual components losses over time. 
        
        # Weights of Objective Function Components
        self.S_E_mean_weight = 1
        self.S_I_mean_weight = 0 # Not Currently Used
        self.EEG_PSD_weight = 0 # Not Currently Used
        self.EEG_FC_weight = 1
        self.BOLD_PSD_weight = 0 # Not Currently Used
        self.BOLD_FC_weight = 1
        
        self.device = device
        
        # Functions of the various Objective Function Components
        self.S_E_mean = CostsMean(num_regions, simKey, targetValue, device = device)
        #self.S_I_mean = CostsMean(...) # Not Currently Used
        #self.EEG_PSD = CostsPSD(num_channels, varIdx = 0, sampleFreqHz = 1000*(1/step_size), targetValue = targetEEG)
        self.EEG_FC = CostsFixedFC(simKey = "eeg", device = device)
        #self.BOLD_PSD = CostsPSD(...) # Not Currently Used
        self.BOLD_FC = CostsFixedFC(simKey = "bold", device = device)
        
    def loss(self, simData, empData, returnLossComponents = False):
        
        S_E_mean_loss = self.S_E_mean.loss(simData) 
        S_I_mean_loss = torch.tensor([0]).to(self.device) #self.S_I_mean.loss(node_history)
        EEG_PSD_loss = torch.tensor([0]).to(self.device) #self.EEG_PSD.loss(EEG_history) 
        EEG_FC_loss = self.EEG_FC.loss(simData, empData['EEG_FC'])
        BOLD_PSD_loss = torch.tensor([0]).to(self.device) #self.BOLD_PS.loss(BOLD_history)
        BOLD_FC_loss = self.BOLD_FC.loss(simData, empData['BOLD_FC'])
                
        totalLoss = self.S_E_mean_weight*S_E_mean_loss + self.S_I_mean_weight*S_I_mean_loss \
                  + self.EEG_PSD_weight*EEG_PSD_loss   + self.EEG_FC_weight*EEG_FC_loss \
                  + self.BOLD_PSD_weight*BOLD_PSD_loss + self.BOLD_FC_weight*BOLD_FC_loss
                 
        if returnLossComponents:
            return totalLoss, (S_E_mean_loss.item(), S_I_mean_loss.item(), EEG_PSD_loss.item(), EEG_FC_loss.item(), BOLD_PSD_loss.item(), BOLD_FC_loss.item())
        else:
            return totalLoss
            