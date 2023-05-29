"""
Authors: Zheng Wang, John Griffiths, Davide Momi, Kevin Kadak, Parsa Oveisi, Taha Morshedzadeh, Sorenza Bastiaens
Neural Mass Model fitting
module for Robinson with forward backward and lateral connection for EEG
"""

# @title new function PyTepFit

# Pytorch stuff


"""
Importage
"""
import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes.parameter import par
from whobpyt.datatypes.AbstractParams import AbstractParams
from whobpyt.datatypes.AbstractNMM import AbstractNMM
import numpy as np  # for numerical operations

class ParamsCT(AbstractParams):

    def __init__(self, **kwargs):

        param = {
            "Q_max": par(250), 
            "sig_theta": par(15/1000), 
            "sigma": par(3.3/1000), 
            "gamma": par(100), 
            "beta": par(200),
            "alpha": par(200/4), 
            "t0": par(0.08),
            "g": par(100), 
            "nu_ee": par(0.0528/1000),
            "nu_ii": par(0.0528/1000),
            "nu_ie": par(0.02/1000),
            "nu_es": par(1.2/1000),
            "nu_is": par(1.2/1000),
            "nu_se": par(1.2/1000),
            "nu_si": par(0.0), 
            "nu_ei": par(0.4/1000),
            "nu_sr": par(0.01/1000),
            "nu_sn": par(0.0), 
            "nu_re": par(0.1/1000),
            "nu_ri": par(0.0), 
            "nu_rs": par(0.1/1000),
            "nu_ss": par(0.0), 
            "nu_rr": par(0.0), 
            "nu_rn": par(0.0), 
            "mu": par(5), 
            "cy0": par(5), 
            "y0": par(2)
        }
        
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])