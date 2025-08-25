import numpy as np  # for numerical operations
from torch import (Tensor as ptTensor, reshape as ptreshape, mean as ptmean, matmul as ptmatmul, transpose as pttranspose, 
                   diag as ptdiag, reciprocal as ptreciprocal, sqrt as ptsqrt, tril as pttril, ones_like as ptones_like, 
                   zeros_like as ptzeros_like, greater as ptgreater, masked_select as ptmasked_select, sum as ptsum, 
                   multiply as ptmultiply, log as ptlog, device as ptdevice)

from ..datatypes import AbstractLoss 
from ..functions.arg_type_check import method_arg_type_check

class CostsFreqs(AbstractLoss):
    def __init__(self, model):
        self.model = model


    def loss(self, simData: dict, empData: torch.Tensor):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, compute the probability and negative log-likelihood.

        Parameters
        ----------
        simData: dict of tensor with node_size X datapoint
            simulated EEG
        empData: tensor with node_size X datapoint
            empirical EEG
        """
        method_arg_type_check(self.loss) # Check that the passed arguments (excluding self) abide by their expected data types
        sim = simData
        emp = empData
        loss_main = torch.sqrt(torch.mean((torch.log(sim) - torch.log(emp)) ** 2))  #
        model = self.model

        # define some constants
        lb = 0.001

        w_cost = 10

        # define the relu function
        m = torch.nn.ReLU()

        exclude_param = []
        if model.use_fit_gains:
            exclude_param.append('gains_con') #TODO: Is this correct?





        loss_EI = 0
        loss_prior = []

        variables_p = [a for a in dir(model.params) if (type(getattr(model.params, a)) == par)]

        for var_name in variables_p:
            var = getattr(model.params, var_name)
            if var.fit_hyper and \
                        var_name not in exclude_param:
                loss_prior.append(torch.sum(( m(var.prior_var_inv) * \
                                            (m(var.val) - m(var.prior_mean)) ** 2)) \
                                  + torch.sum(-torch.log( m(var.prior_var_inv))))

        # total loss
        loss = 200 * w_cost * loss_main + 1 * sum(loss_prior) + 1 * loss_EI
        return loss, loss_main