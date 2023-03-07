import numpy as np

# Multi-Modal Connectome-based Neural Mass Modelling

## EQUATIONS & BIOLOGICAL VARIABLES FROM:
#
# Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitation–inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
# Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
# Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
# Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.
# https://github.com/GriffithsLab/tepfit/blob/main/tepfit/fit.py (for state variable value bound equations)

import numpy
import math

class BOLD_Params():
    def __init__(self):
        #############################################
        ## BOLD Constants
        #############################################
        
        #Friston 2003 - Table 1 - Priors on biophysical parameters
        self.kappa = 0.65 # Rate of signal decay (1/s)
        self.gammaB = 0.42 # Rate of flow-dependent elimination (1/s)
        self.tao = 0.98 # Hemodynamic transit time (s)
        self.alpha = 0.32 # Grubb's exponent
        self.ro = 0.34 #Resting oxygen extraction fraction
        
        self.V_0 = 0.02
        self.k_1 = 7*self.ro
        self.k_2 = 2
        self.k_3 = 2*self.ro - 0.2
