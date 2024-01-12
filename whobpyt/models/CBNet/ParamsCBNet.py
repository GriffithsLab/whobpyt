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
        # base voltage for each reciptor
        "VL" 
        "VE" 
        "VI"
        "VNMDA" 

         # saturation function parameter
        "alpha_mg"
         # estimate of V distributin
         
        "VR" 
        "pi_sigma"  #std

        "gL" #gain on leak

        # time constant or self feedback gains each states voltage c and each reciptors kappa
        "C"   
        "kappa" 


        # connection between populations
        "gamma_gE" 
        "gamma_gE_sc" 
        "gamma_gI" 
        "gamma_gI_sc" 
        "gamma_gNMDA"
        "gamma_gNMDA_sc" 

        # conduct velocity (delay between regions)
        "mu"
        
        # for the match magnitude of eeg
        y0 (par)
        k (par)
        cy0 (par)
        ki (par) 
    """
    def __init__(self, **kwargs):
        """
        Initializes the ParamsCB object.

        Args:
            **kwargs: Keyword arguments for the model parameters.

        Returns:
            None
        """
        super(ParamsCBnet, self).__init__(**kwargs)
        params = {
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

        for var in params:
            if var not in self.params:
                self.params[var] = params[var]
        
        self.setParamsAsattr()