"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch
from ..datatypes import Parameter as par
from ..datatypes import AbstractLoss 
from .cost_TS import CostsTS
from ..functions.arg_type_check import method_arg_type_check

class CostsJR(AbstractLoss):
    def __init__(self, model):
        
        self.simKey = model.output_names[0]
        self.mainLoss = CostsTS(simKey = self.simKey, model=model)
        
        
    def loss(self, simData: dict, empData: torch.Tensor):
        
        method_arg_type_check(self.loss) # Check that the passed arguments (excluding self) abide by their expected data types
        sim = simData
        emp = empData
       
        # define some constants
        w_cost = 10

    

        loss_main = self.mainLoss.main_loss(sim, emp)

        loss_EI = 0
        loss_prior = self.mainLoss.prior_loss()

        
        loss = 0.1 * w_cost * loss_main + 1 * sum(loss_prior) + 1 * loss_EI
        return loss, loss_main
