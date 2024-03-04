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
        self.mainLoss = CostsFC("bold")
        self.simKey = "bold"
        self.model = model

    def loss(self, simData: dict, empData: torch.Tensor):
        
        method_arg_type_check(self.loss) # Check that the passed arguments (excluding self) abide by their expected data types
        sim = simData
        emp = empData
        
        model = self.model
        state_vals = sim
        
        # define some constants
        lb = 0.001

        w_cost = 10

        # define the relu function
        m = torch.nn.ReLU()

        exclude_param = []
        if model.use_fit_gains:
            exclude_param.append('gains_con')

        loss_main = self.mainLoss.loss(sim, emp)

        loss_EI = 0

        """E_window = state_vals['E']
        I_window = state_vals['I']
        f_window = state_vals['f']
        v_window = state_vals['v']
        x_window = state_vals['x']
        q_window = state_vals['q']"""
        

        loss_prior = []

        variables_p = [a for a in dir(model.params) if not a.startswith('__') and (type(getattr(model.params, a)) == par)]
        # get penalty on each model parameters due to prior distribution
        for var_name in variables_p:
            # print(var)
            var = getattr(model.params, var_name)
            if var.has_prior and var_name not in ['std_in'] and var_name not in exclude_param:
                loss_prior.append(torch.sum((lb + m(var.prior_var)) * (m(var.val) - m(var.prior_mean)) ** 2) \
                + torch.sum(-torch.log(lb + m(var.prior_var)))) #TODO: Double check about converting _v_inv to just variance representation
          
        # total loss
        loss = w_cost * loss_main + sum(loss_prior) 
        return loss
