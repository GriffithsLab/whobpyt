"""
Authors: Zheng Wang, John Griffiths, Andrew Clappisan, Hussain Ather
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch
from whobpyt.datatypes.AbstractLoss import AbstractLoss


class CostsTS(AbstractLoss):
    def __init__(self, simKey):
        super(CostsTS, self).__init__()
        self.simKey = simKey

    def loss(self, sim, emp, model: torch.nn.Module = None, state_vals = None):
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
