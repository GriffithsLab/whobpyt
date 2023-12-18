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

class BOLD_Params(AbstractParams):
    def __init__(self):
        #############################################
        ## BOLD Constants
        #############################################
        
        #Friston 2003 - Table 1 - Priors on biophysical parameters
        self.kappa = par(0.65) # Rate of signal decay (1/s)
        self.gammaB = par(0.42) # Rate of flow-dependent elimination (1/s)
        self.tao = par(0.98) # Hemodynamic transit time (s)
        self.alpha = par(0.32) # Grubb's exponent
        self.ro = par(0.34) #Resting oxygen extraction fraction
        
        self.V_0 = par(0.02)
        self.k_1 = par(7*self.ro.npValue())
        self.k_2 = par(2)
        self.k_3 = par(2*self.ro.npValue() - 0.2)
