"""
Authors: Andrew Clappison, John Griffiths, Zheng Wang, Davide Momi, Sorenza Bastiaens, Taha Morshedzadeh, Kevin Kadak, Parsa Oveisi, Shreyas Harita
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
        self.mainLoss = CostsFC("bold")
        self.simKey = "bold"
        self.model = model

    def loss(self, simData, empData):
        
        #method_arg_type_check(self.loss) # Check that the passed arguments (excluding self) abide by their expected data types
        sim = simData
        emp = empData
        
        model = self.model
        
        
        # define some constants
        lb = 0.001

        w_cost = 10

        # define the relu function
        m = torch.nn.ReLU()

        
        loss_main = self.mainLoss.loss(sim, emp)

        
        loss_prior = []

        variables_p = [a for a in dir(model.params) if type(getattr(model.params, a)) == par]
        # get penalty on each model parameters due to prior distribution
        for var_name in variables_p:
            # print(var)
            var = getattr(model.params, var_name)
            if var.fit_hyper:
                loss_prior.append(torch.sum((lb + m(var.prior_var_inv)) * \
                                                (m(var.val) - m(var.prior_mean)) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_var_inv)))) #TODO: Double check about converting _v_inv 
        loss = w_cost * loss_main + sum(loss_prior) 
        return loss, loss_main
