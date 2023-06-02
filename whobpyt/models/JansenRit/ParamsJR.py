"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for JR with forward backward and lateral connection for EEG
"""

# @title new function PyTepFit

# Pytorch stuff


"""
Importage
"""
import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes import AbstractNMM, AbstractParams, par
import numpy as np  # for numerical operations

class ParamsJR(AbstractParams):

    def __init__(self, **kwargs):

        param = {
            "A ": par(3.25), 
            "a": par(100), 
            "B": par(22), 
            "b": par(50), 
            "g": par(1000),
            
            "c1": par(135), 
            "c2": par(135 * 0.8), 
            "c3 ": par(135 * 0.25), 
            "c4": par(135 * 0.25),
            
            "std_in": par(100), 
            "vmax": par(5), 
            "v0": par(6), 
            "r": par(0.56), 
            "y0": par(2),
            
            "mu": par(.5), 
            "k": par(5), 
            "cy0": par(5), 
            "ki": par(1)
        }
        
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])
