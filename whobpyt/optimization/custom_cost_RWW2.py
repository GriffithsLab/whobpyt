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

class CostsRWW2(AbstractLoss):
    def __init__(self, model : AbstractNMM):
        
        self.simKey = "bold"
        self.simKeyeeg = "states"
        self.mainLoss = CostsFC(self.simKey)
        self.secondLoss = CostsTS(self.simKeyeeg)
        self.model = model

    def loss(self, simData: dict, empData: torch.Tensor, empEEG: torch.Tensor):
        
        method_arg_type_check(self.loss) # Check that the passed arguments (excluding self) abide by their expected data types
        
        
        model = self.model
        
        
        # define some constants
        lb = 0.001

        w_cost = 1

        # define the relu function
        m = torch.nn.ReLU()

        

        loss_main = self.mainLoss.loss(simData, empData) + self.mainLoss.loss(sim, empEEG)

        
        loss_prior = []

        

        variables_p = [a for a in dir(model.params) if (type(getattr(model.params, a)) == par)]

        for var_name in variables_p:
            var = getattr(model.params, var_name)
            if var.fit_par:
                loss_prior.append(torch.sum(( m(var.prior_var)) * \
                                            (m(var.val) - m(var.prior_mean)) ** 2) \
                                  + torch.sum(-torch.log( m(var.prior_var))))
        # total loss
        loss = w_cost * loss_main + sum(loss_prior) 
        return loss
