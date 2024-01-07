"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Davide Momi, Sorenza Bastiaens, Parsa Oveisi
Neural Mass Model fitting
module for CBnet with forward backward and lateral connection for EEG
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

class ParamsCBnet(AbstractParams):
    """
    A class for setting the parameters of a neural mass model(CBnet) for EEG data fitting.

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
            #noise std
            "sigma_V" : par(0, asLog=True),
            "sigma_g" : par(0, asLog=True),

            # base voltage for each reciptor
            "VL" : par(-70) ,
            "VE" : par(0),
            "VI" : par(-90),
            "VNMDA" : par(10),

             # saturation function parameter
            "alpha_mg": par(0.06),
             # estimate of V distributin
            "VR" : par(-40), #mean
            "pi_sigma" : par(np.log(np.array([32, 32, 32])), asLog=True), #std

            "gL" : par(1), #gain on leak

            # time constant or self feedback gains each states voltage c and each reciptors kappa
            "C"   : par(np.array([np.log(4/1000), np.log(16/1000), np.log(100/1000)]), asLog=True),
            "kappa" : par(np.array([np.log(1000/8), np.log(1000/8), np.log(1000/8)]), asLog =True),


            # connection between populations
            "gamma_gE" :   par(np.log(64)*np.ones((3,3)), asLog=True), # connection gains in E reciptor among populations
            "gamma_gE_sc" : par(np.array([[0, 1, 1],[1, 0, 1],[1, 1, 0]])),
            "gamma_gI" :   par(np.log(64)*np.ones((3,3)), asLog=True), # connection gains in I reciptor among populations
            "gamma_gI_sc" : par(np.array([[0, 1, 1],[1, 0, 1],[1,1,0]])),
            "gamma_gNMDA" :   par(np.log(64)*np.ones((3,3)), asLog=True), # connection gains in NMDA reciptor among populations
            "gamma_gNMDA_sc" : par(np.array([[0, 0, 1],[1/4, 0,1],[0, 0, 0]])),

            # conduct velocity (delay between regions)
            "mu": par(.5),

            # input gain related to stimulus
            "k": par(5),
            #global gain
            "g": par(100),
            "g_f": par(10),
            "g_b": par(10),
            # parameters for channal eeg (magitude related)
            "cy0": par(5),
            "ki": par(1)
        }

        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])