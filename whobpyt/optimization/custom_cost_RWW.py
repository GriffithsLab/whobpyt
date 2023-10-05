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

        E_window = state_vals['E']
        I_window = state_vals['I']
        f_window = state_vals['f']
        v_window = state_vals['v']
        x_window = state_vals['x']
        q_window = state_vals['q']
        if model.use_Gaussian_EI and model.use_Bifurcation:
            loss_EI = torch.mean(model.E_v_inv * (E_window - model.E_m) ** 2) \
                      + torch.mean(-torch.log(model.E_v_inv)) + \
                      torch.mean(model.I_v_inv * (I_window - model.I_m) ** 2) \
                      + torch.mean(-torch.log(model.I_v_inv)) + \
                      torch.mean(model.q_v_inv * (q_window - model.q_m) ** 2) \
                      + torch.mean(-torch.log(model.q_v_inv)) + \
                      torch.mean(model.v_v_inv * (v_window - model.v_m) ** 2) \
                      + torch.mean(-torch.log(model.v_v_inv)) \
                      + 5.0 * (m(model.sup_ca) * m(model.g_IE) ** 2
                               - m(model.sup_cb) * m(model.params.g_IE.value())
                               + m(model.sup_cc) - m(model.params.g_EI.value())) ** 2
        if model.use_Gaussian_EI and not model.use_Bifurcation:
            loss_EI = torch.mean(model.E_v_inv * (E_window - model.E_m) ** 2) \
                      + torch.mean(-torch.log(model.E_v_inv)) + \
                      torch.mean(model.I_v_inv * (I_window - model.I_m) ** 2) \
                      + torch.mean(-torch.log(model.I_v_inv)) + \
                      torch.mean(model.q_v_inv * (q_window - model.q_m) ** 2) \
                      + torch.mean(-torch.log(model.q_v_inv)) + \
                      torch.mean(model.v_v_inv * (v_window - model.v_m) ** 2) \
                      + torch.mean(-torch.log(model.v_v_inv))

        if not model.use_Gaussian_EI and model.use_Bifurcation:
            loss_EI = .1 * torch.mean(
                torch.mean(E_window * torch.log(E_window) + (1 - E_window) * torch.log(1 - E_window) \
                           + 0.5 * I_window * torch.log(I_window) + 0.5 * (1 - I_window) * torch.log(
                    1 - I_window), dim=1)) + \
                      + 5.0 * (m(model.sup_ca) * m(model.params.g_IE.value()) ** 2
                               - m(model.sup_cb) * m(model.params.g_IE.value())
                               + m(model.sup_cc) - m(model.params.g_EI.value())) ** 2

        if not model.use_Gaussian_EI and not model.use_Bifurcation:
            loss_EI = .1 * torch.mean(
                torch.mean(E_window * torch.log(E_window) + (1 - E_window) * torch.log(1 - E_window) \
                           + 0.5 * I_window * torch.log(I_window) + 0.5 * (1 - I_window) * torch.log(
                    1 - I_window), dim=1))

        loss_prior = []

        variables_p = [a for a in dir(model.params) if not a.startswith('__') and (type(getattr(model.params, a)) == par)]
        # get penalty on each model parameters due to prior distribution
        for var_name in variables_p:
            # print(var)
            var = getattr(model.params, var_name)
            if model.use_Bifurcation:
                if var.has_prior and var_name not in ['std_in', 'g_EI', 'g_IE'] and \
                        var_name not in exclude_param:
                    loss_prior.append(torch.sum((lb + m(var.prior_var)) * \
                                                (m(var.val) - m(var.prior_mean)) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_var)))) #TODO: Double check about converting _v_inv to just variance representation
            else:
                if var.has_prior and var_name not in ['std_in'] and \
                        var_name not in exclude_param:
                    loss_prior.append(torch.sum((lb + m(var.prior_var)) * \
                                                (m(var.val) - m(var.prior_mean)) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_var)))) #TODO: Double check about converting _v_inv to just variance representation
          
        # total loss
        loss = w_cost * loss_main + sum(loss_prior) + 1 * loss_EI
        return loss
