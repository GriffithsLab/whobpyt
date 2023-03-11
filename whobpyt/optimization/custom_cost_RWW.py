"""
Authors: Zheng Wang, John Griffiths, Andrew Clappisan, Hussain Ather
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch
from whobpyt.datatypes.parameter import par
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.optimization.cost_FC import CostsFC

class CostsRWW(AbstractLoss):
    def __init__(self):
        super(CostsRWW, self).__init__()
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

        E_window = state_vals['E_window']
        I_window = state_vals['I_window']
        f_window = state_vals['f_window']
        v_window = state_vals['v_window']
        x_window = state_vals['x_window']
        q_window = state_vals['q_window']
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
                               - m(model.sup_cb) * m(model.g_IE)
                               + m(model.sup_cc) - m(model.g_EI)) ** 2
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
                      + 5.0 * (m(model.sup_ca) * m(model.g_IE) ** 2
                               - m(model.sup_cb) * m(model.g_IE)
                               + m(model.sup_cc) - m(model.g_EI)) ** 2

        if not model.use_Gaussian_EI and not model.use_Bifurcation:
            loss_EI = .1 * torch.mean(
                torch.mean(E_window * torch.log(E_window) + (1 - E_window) * torch.log(1 - E_window) \
                           + 0.5 * I_window * torch.log(I_window) + 0.5 * (1 - I_window) * torch.log(
                    1 - I_window), dim=1))

        loss_prior = []

        variables_p = [a for a in dir(model.param) if not a.startswith('__') and (type(getattr(model.param, a)) == par)]
        # get penalty on each model parameters due to prior distribution
        for var_name in variables_p:
            # print(var)
            var = getattr(model.param, var_name)
            if model.use_Bifurcation:
                #if np.any(getattr(model.param, var)[1] > 0) and var not in ['std_in', 'g_EI', 'g_IE'] and \
                #TODO: This currenlty assumes there is a penalty only if the hyper_parameters are being fit. Need a better solution. 
                if var.fit_hyper and var_name not in ['std_in', 'g_EI', 'g_IE'] and \
                        var_name not in exclude_param:
                    # print(var)
                    #dict_np = {'m': var + '_m', 'v': var + '_v_inv'}
                    #loss_prior.append(torch.sum((lb + m(model.get_parameter(dict_np['v']))) * \
                    #                            (m(model.get_parameter(var)) - m(
                    #                                model.get_parameter(dict_np['m']))) ** 2) \
                    #                  + torch.sum(-torch.log(lb + m(model.get_parameter(dict_np['v'])))))
                    loss_prior.append(torch.sum((lb + m(var.prior_var)) * \
                                                (m(var.val) - m(var.prior_mean)) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_var)))) #TODO: Double check about converting _v_inv to just variance representation
            else:
                #if np.any(getattr(model.param, var)[1] > 0) and var not in ['std_in'] and \
                #TODO: This currenlty assumes there is a penalty only if the hyper_parameters are being fit. Need a better solution.
                if var.fit_hyper and var_name not in ['std_in'] and \
                        var_name not in exclude_param:
                    # print(var)
                    #dict_np = {'m': var + '_m', 'v': var + '_v_inv'}
                    #loss_prior.append(torch.sum((lb + m(model.get_parameter(dict_np['v']))) * \
                    #                            (m(model.get_parameter(var)) - m(
                    #                                model.get_parameter(dict_np['m']))) ** 2) \
                    #                  + torch.sum(-torch.log(lb + m(model.get_parameter(dict_np['v'])))))
                    loss_prior.append(torch.sum((lb + m(var.prior_var)) * \
                                                (m(var.val) - m(var.prior_mean)) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_var)))) #TODO: Double check about converting _v_inv to just variance representation
          
        # total loss
        loss = w_cost * loss_main + sum(loss_prior) + 1 * loss_EI
        return loss
