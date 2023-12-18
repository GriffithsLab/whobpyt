"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Sorenza Bastiaens, Parsa Oveisi
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
    """
    A class for setting the parameters of a neural mass model for EEG data fitting.

    Attributes:
        A (par): The amplitude of the EPSP (excitatory post synaptic potential).
        a (par): A metric of the rate constant for the EPSP.
        B (par): The amplitude of the IPSP (inhibitory post synaptic potential).
        b (par): A metric of the rate constant for the IPSP.
        g (par): The gain of ???.
        c1 (par): The connectivity parameter from the pyramidal to excitatory interneurons.
        c2 (par): The connectivity parameter from the excitatory interneurons to the pyramidal cells.
        c3 (par): The connectivity parameter from the pyramidal to inhibitory interneurons.
        c4 (par): The connectivity parameter from the inhibitory interneurons to the pyramidal cells.
        std_in (par): The standard deviation of the input noise.
        vmax (par): The maximum value of the sigmoid function.
        v0 (par): The midpoint of the sigmoid function.
        r (par): The slope of the sigmoid function.
        y0 (par): ???.
        mu (par): The mean of the input.
        k (par): ???.
        cy0 (par): ???.
        ki (par): ???.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ParamsJR object.

        Args:
            **kwargs: Keyword arguments for the model parameters.

        Returns:
            None
        """
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
