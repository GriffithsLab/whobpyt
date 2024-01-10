import torch
from whobpyt.datatypes import AbstractParams, par

class ParamsRWWEI2(AbstractParams):
    ## EQUATIONS & BIOLOGICAL VARIABLES FROM:
    #
    # Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitationâ€“inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
    # Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
    # Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
    # Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.
  
    def __init__(self, num_regions = 1):
        #############################################
        ## RWW Constants
        #############################################
        
        #Zeroing the components which deal with a connected network
        self.G = par(1)
        self.Lambda = par(0) #1 or 0 depending on using long range feed forward inhibition (FFI)

        #Excitatory Gating Variables
        self.a_E = par(310)                     # nC^(-1)
        self.b_E = par(125)                     # Hz
        self.d_E = par(0.16)                    # s
        self.tau_E = self.tau_NMDA = par(100)   # ms
        self.W_E = par(1)
        
        #Inhibitory Gating Variables
        self.a_I = par(615)               # nC^(-1)
        self.b_I = par(177)               # Hz
        self.d_I = par(0.087)             # s
        self.tau_I = self.tau_GABA = par(10)   # ms
        self.W_I = par(0.7)
        
        #Setting other variables
        self.w_plus = par(1.4) # Local excitatory recurrence
        self.J_NMDA = par(0.15) # Excitatory synaptic coupling in nA
        self.J = par(1.0) # Local feedback inhibitory synaptic coupling. 1 in no-FIC case, different in FIC case #TODO: Currently set to J_NMDA but should calculate based on paper
        self.gamma = par(0.641/1000) #a kinetic parameter in ms
        self.sig = par(0.01) #0.01 # Noise amplitude at node in nA
        self.I_0 = par(0.382) # The overall effective external input in nA
        
        self.I_external = par(0.00) #External input current 
        
        
        #############################################
        ## Model Additions/modifications
        #############################################
        
        self.gammaI = par(1/1000) #Zheng suggested this to get oscillations
        self.J_new = par(1)