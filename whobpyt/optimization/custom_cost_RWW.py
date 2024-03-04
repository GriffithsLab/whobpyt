"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch
from whobpyt.datatypes.parameter import par
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.datatypes.AbstractNMM import AbstractNMM
from whobpyt.optimization.cost_FC import CostsFC
from whobpyt.functions.arg_type_check import method_arg_type_check

class CostsRWW(AbstractLoss):
    def __init__(self, model : AbstractNMM):
        self.simKey = model.output_names[0]
        self.mainLoss = CostsFC(simKey = self.simKey, model = model)

    def loss(self, simData: dict, empData: torch.Tensor):
        
        method_arg_type_check(self.loss) # Check that the passed arguments (excluding self) abide by their expected data types
        sim = simData
        emp = empData
        
        
        state_vals = sim
        
        # define some constants
        w_cost = 10

        
        loss_main = self.mainLoss.main_loss(sim, emp)

        loss_EI = 0

        E_window = state_vals['E']
        I_window = state_vals['I']
        f_window = state_vals['f']
        v_window = state_vals['v']
        x_window = state_vals['x']
        q_window = state_vals['q']
        

        loss_prior = self.mainLoss.prior_loss()

        #print(loss_main)
          
        # total loss
        loss = w_cost * loss_main + sum(loss_prior) 
        return loss, loss_main
