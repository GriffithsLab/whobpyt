"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.functions.arg_type_check import method_arg_type_check


class CostsTS(AbstractLoss):
    def __init__(self, simKey):
        super(CostsTS, self).__init__()
        self.simKey = simKey

    def loss(self, sim: torch.Tensor, emp: torch.Tensor, model: torch.nn.Module = None, state_vals = None):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, compute the probability and negative log-likelihood.
        
        Parameters
        ----------
        sim: tensor with node_size X datapoint
            simulated EEG
        emp: tensor with node_size X datapoint
            empirical EEG
        """
        method_arg_type_check(self.loss, exclude = ['model', 'state_vals']) # Check that the passed arguments (excluding self) abide by their expected data types

        losses = torch.sqrt(torch.mean((sim - emp) ** 2))  #
        return losses
