import numpy as np
from whobpyt.datatypes import AbstractParams, par

# Multi-Modal Connectome-based Neural Mass Modelling

## EQUATIONS & BIOLOGICAL VARIABLES FROM:
#
# Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitation–inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
# Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
# Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
# Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.
# https://github.com/GriffithsLab/tepfit/blob/main/tepfit/fit.py (for state variable value bound equations)

import numpy

class ParamsBOLD(AbstractParams):
    def __init__(self, **kwargs):
        #############################################
        ## BOLD Constants
        #############################################
        
        #Friston 2003 - Table 1 - Priors on biophysical parameters
        super(ParamsBOLD, self).__init__(**kwargs)
        
        params = {

            # Output (BOLD signal)
            "alpha": par(0.32),
            "rho": par(0.34),
            "k1": par(2.38),
            "k2": par(2.0),
            "k3": par(0.48),  # adjust this number from 0.48 for BOLD fluctruate around zero
            "V": par(.02),
            "E0": par(0.34),
            "tau_s": par(1 / 0.65),
            "tau_f": par(1 / 0.41),
            "tau_0": par(0.98)

        }

        for var in params:
            if var not in self.params:
                self.params[var] = params[var]

        self.setParamsAsattr()
