"""
Authors: Zheng Wang, John Griffiths, Andrew Clappisan, Hussain Ather
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch
from whobpyt.datatypes.parameter import par
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.optimization.cost_TS import CostsTS


class CostsJR(AbstractLoss):
    def __init__(self):
        super(CostsJR, self).__init__()
        self.mainLoss = CostsTS("eeg")
        self.simKey = "eeg"
        
    def loss(self, sim, emp, model: torch.nn.Module, state_vals):
        # define some constants
        lb = 0.001

        w_cost = 10

        # define the relu function
        m = torch.nn.ReLU()

        exclude_param = []
        if model.use_fit_gains:
            exclude_param.append('gains_con') #TODO: Is this correct?

        if model.use_fit_lfm:
            exclude_param.append('lm') #TODO: Is this correct?

        loss_main = self.mainLoss.loss(sim, emp)

        loss_EI = 0
        loss_prior = []

        variables_p = [a for a in dir(model.params) if (type(getattr(model.params, a)) == par)]

        for var_name in variables_p:
            var = getattr(model.params, var_name)
            if var.has_prior and var_name not in ['std_in'] and \
                        var_name not in exclude_param:
                loss_prior.append(torch.sum((lb + m(var.prior_var)) * \
                                            (m(var.val) - m(var.prior_mean)) ** 2) \
                                  + torch.sum(-torch.log(lb + m(var.prior_var))))
        
        # total loss
        loss = 0.1 * w_cost * loss_main + 1 * sum(loss_prior) + 1 * loss_EI
        return loss
