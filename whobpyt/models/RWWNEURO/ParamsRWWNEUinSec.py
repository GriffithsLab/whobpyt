"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Clemens Pellengahr, Hussain Ather, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita
Neural Mass Model fitting
module for wong-wang model
"""

import torch
from whobpyt.datatypes import AbstractParams, par

class ParamsRWWNEU(AbstractParams):
    
    def __init__(self, **kwargs):
        
        super(ParamsRWWNEU, self).__init__(**kwargs)
        
        params = {

            "std_in": par(0.02),  # standard deviation of the Gaussian noise
            "std_out": par(0.02),  # standard deviation of the Gaussian noise
            # Parameters for the ODEs
            # Excitatory population
            "W_E": par(1.),  # scale of the external input
            "tau_E": par(.100),  # decay time
            "gamma_E": par(0.641 ),  # other dynamic parameter (?)

            # Inhibitory population
            "W_I": par(0.7),  # scale of the external input
            "tau_I": par(0.01),  # decay time
            "gamma_I": par(1.),  # other dynamic parameter (?)

            # External input
            "I_0": par(0.32),  # external input
            "I_external": par(0.),  # external stimulation

            # Coupling parameters
            "g": par(20.),  # global coupling (from all nodes E_j to single node E_i)
            "g_EE": par(.1),  # local self excitatory feedback (from E_i to E_i)
            "g_IE": par(.1),  # local inhibitory coupling (from I_i to E_i)
            "g_EI": par(0.1),  # local excitatory coupling (from E_i to I_i)

            "aE": par(310),
            "bE": par(125),
            "dE": par(0.16),
            "aI": par(615),
            "bI": par(177),
            "dI": par(0.087),
            "mu": par(0.5)

        }

        for var in params:
            if var not in self.params:
                self.params[var] = params[var]

        self.setParamsAsattr()
