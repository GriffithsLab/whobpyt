"""
Authors: Zheng Wang, John Griffiths, Andrew Clappisan, Hussain Ather
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.models.optimization.cost_TS import CostsTS
from whobpyt.optimization.cost_FC import CostsFC


class CostsLIN:
    def __init__(self, method = 0):
        super(CostsLIN, self).__init__()
        
        if (method == 0):
            self.mainLoss = CostsTS()
        else:
            self.mainLoss = CostsFC()

    def loss(self, sim, emp, model: torch.nn.Module, state_vals):
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
        loss_prior = []

        variables_p = [a for a in dir(model.params) if
                       not a.startswith('__') and not callable(getattr(model.params, a))]

        for var in variables_p:
            if np.any(getattr(model.params, var)[1] > 0) and var not in ['std_in'] and \
                    var not in exclude_param:
                # print(var)
                dict_np = {'m': var + '_m', 'v': var + '_v_inv'}
                loss_prior.append(torch.sum((lb + m(model.get_parameter(dict_np['v']))) * \
                                            (m(model.get_parameter(var)) - m(
                                                model.get_parameter(dict_np['m']))) ** 2) \
                                  + torch.sum(-torch.log(lb + m(model.get_parameter(dict_np['v'])))))
        
        # total loss
        loss = 0.1 * w_cost * loss_main + sum(loss_prior) + 1 * loss_EI
        return loss
