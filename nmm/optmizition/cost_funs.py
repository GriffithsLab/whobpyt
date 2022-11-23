"""
Authors: Zheng Wang, John Griffiths, Andrew Clappisan, Hussain Ather
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch


class Costs:
    def __init__(self, method):
        self.method = method

    def cost_dist(self, sim, emp):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, the probability and negative log-likelihood.
        Parameters
        ----------
        sim: tensor with node_size X datapoint
            simulated EEG
        emp: tensor with node_size X datapoint
            empirical EEG
        """

        losses = torch.sqrt(torch.mean((sim - emp) ** 2))  #
        return losses

    def cost_r(self, logits_series_tf, labels_series_tf):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, the probability and negative log-likelihood.
        Parameters
        ----------
        logits_series_tf: tensor with node_size X datapoint
            simulated BOLD
        labels_series_tf: tensor with node_size X datapoint
            empirical BOLD
        """
        # get node_size() and TRs_per_window()
        node_size = logits_series_tf.shape[0]
        truncated_backprop_length = logits_series_tf.shape[1]

        # remove mean across time
        labels_series_tf_n = labels_series_tf - torch.reshape(torch.mean(labels_series_tf, 1),
                                                              [node_size, 1])  # - torch.matmul(

        logits_series_tf_n = logits_series_tf - torch.reshape(torch.mean(logits_series_tf, 1),
                                                              [node_size, 1])  # - torch.matmul(

        # correlation
        cov_sim = torch.matmul(logits_series_tf_n, torch.transpose(logits_series_tf_n, 0, 1))
        cov_def = torch.matmul(labels_series_tf_n, torch.transpose(labels_series_tf_n, 0, 1))

        # fc for sim and empirical BOLDs
        FC_sim_T = torch.matmul(torch.matmul(torch.diag(torch.reciprocal(torch.sqrt(
            torch.diag(cov_sim)))), cov_sim),
            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_sim)))))
        FC_T = torch.matmul(torch.matmul(torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_def)))), cov_def),
                            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_def)))))

        # mask for lower triangle without diagonal
        ones_tri = torch.tril(torch.ones_like(FC_T), -1)
        zeros = torch.zeros_like(FC_T)  # create a tensor all ones
        mask = torch.greater(ones_tri, zeros)  # boolean tensor, mask[i] = True iff x[i] > 1

        # mask out fc to vector with elements of the lower triangle
        FC_tri_v = torch.masked_select(FC_T, mask)
        FC_sim_tri_v = torch.masked_select(FC_sim_T, mask)

        # remove the mean across the elements
        FC_v = FC_tri_v - torch.mean(FC_tri_v)
        FC_sim_v = FC_sim_tri_v - torch.mean(FC_sim_tri_v)

        # corr_coef
        corr_FC = torch.sum(torch.multiply(FC_v, FC_sim_v)) \
                  * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_v, FC_v)))) \
                  * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_sim_v, FC_sim_v))))

        # use surprise: corr to calculate probability and -log
        losses_corr = -torch.log(0.5000 + 0.5 * corr_FC)  # torch.mean((FC_v -FC_sim_v)**2)#
        return losses_corr

    def cost_eff(self, sim, emp, model: torch.nn.Module, next_window):
        # define some constants
        lb = 0.001

        w_cost = 10

        # define the relu function
        m = torch.nn.ReLU()

        exclude_param = []
        if model.use_fit_gains:
            exclude_param.append('gains_con')

        if model.model_name == "JR" and model.use_fit_lfm:
            exclude_param.append('lm')

        if self.method == 0:
            loss_main = self.cost_dist(sim, emp)
        else:
            loss_main = self.cost_r(sim, emp)
        loss_EI = 0

        if model.model_name == 'RWW':
            E_window = next_window['E_window']
            I_window = next_window['I_window']
            f_window = next_window['f_window']
            v_window = next_window['v_window']
            x_window = next_window['x_window']
            q_window = next_window['q_window']
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

            variables_p = [a for a in dir(model.param) if
                           not a.startswith('__') and not callable(getattr(model.param, a))]
            # get penalty on each model parameters due to prior distribution
            for var in variables_p:
                # print(var)
                if model.use_Bifurcation:
                    if np.any(getattr(model.param, var)[1] > 0) and var not in ['std_in', 'g_EI', 'g_IE'] and \
                            var not in exclude_param:
                        # print(var)
                        dict_np = {'m': var + '_m', 'v': var + '_v_inv'}
                        loss_prior.append(torch.sum((lb + m(model.get_parameter(dict_np['v']))) * \
                                                    (m(model.get_parameter(var)) - m(
                                                        model.get_parameter(dict_np['m']))) ** 2) \
                                          + torch.sum(-torch.log(lb + m(model.get_parameter(dict_np['v'])))))
                else:
                    if np.any(getattr(model.param, var)[1] > 0) and var not in ['std_in'] and \
                            var not in exclude_param:
                        # print(var)
                        dict_np = {'m': var + '_m', 'v': var + '_v_inv'}
                        loss_prior.append(torch.sum((lb + m(model.get_parameter(dict_np['v']))) * \
                                                    (m(model.get_parameter(var)) - m(
                                                        model.get_parameter(dict_np['m']))) ** 2) \
                                          + torch.sum(-torch.log(lb + m(model.get_parameter(dict_np['v'])))))
        else:
            lose_EI = 0
            loss_prior = []

            variables_p = [a for a in dir(model.param) if
                           not a.startswith('__') and not callable(getattr(model.param, a))]

            for var in variables_p:
                if np.any(getattr(model.param, var)[1] > 0) and var not in ['std_in'] and \
                        var not in exclude_param:
                    # print(var)
                    dict_np = {'m': var + '_m', 'v': var + '_v_inv'}
                    loss_prior.append(torch.sum((lb + m(model.get_parameter(dict_np['v']))) * \
                                                (m(model.get_parameter(var)) - m(
                                                    model.get_parameter(dict_np['m']))) ** 2) \
                                      + torch.sum(-torch.log(lb + m(model.get_parameter(dict_np['v'])))))
        # total loss
        loss = 0
        if model.model_name == 'RWW':
            loss = 0.1 * w_cost * loss_main + 1 * sum(
                loss_prior) + 1 * loss_EI
        elif model.model_name == 'JR':
            loss = w_cost * loss_main + sum(loss_prior) + 1 * loss_EI
        elif model.model_name == 'LIN':
            loss = 0.1 * w_cost * loss_main + sum(loss_prior) + 1 * loss_EI
        return loss
