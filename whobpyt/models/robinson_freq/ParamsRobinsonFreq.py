"""
Authors: Zheng Wang, John Griffiths, Davide Momi, Kevin Kadak, Parsa Oveisi, Taha Morshedzadeh, Sorenza Bastiaens
Neural Mass Model fitting
module for Robinson with forward backward and lateral connection for EEG
"""

import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes.parameter import par
from whobpyt.datatypes.AbstractParams import AbstractParams
from whobpyt.datatypes.AbstractNMM import AbstractNMM
import numpy as np  # for numerical operations

class ParamsCT_fq(AbstractParams):

    def __init__(self, **kwargs):
        param = {
                'gamma': par(100),
                'beta': par(200),
                'alpha': par(50),
                 't0_2': par(0.08),
                'ii': par(0.0528),
                'ee': par(0.0528),

                'es': par(1.2),
                'sr': par(-0.01),
                'sn': par(10.0),

                'eis': par(-0.48),
                'eie': par(-0.008000000000000002),
                'srs': par(-0.0010000000000000002),
                'g_ese': par(-0.576),
                'g_esre':par(0.00047999999999999996)
        }
        
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])