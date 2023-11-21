import numpy as np

# Multi-Modal Connectome-based Neural Mass Modelling
#
# This is to simulate a RWW Network with addtional BOLD and EEG layers.

## EQUATIONS & BIOLOGICAL VARIABLES FROM:
#
# Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitation–inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
# Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
# Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
# Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.
# https://github.com/GriffithsLab/tepfit/blob/main/tepfit/fit.py (for state variable value bound equations)

import numpy
from math import sqrt
    
class RWWEI2_np():
    def __init__(self, num_regions, params, Con_Mtx, Dist_Mtx, step_size = 0.1):        
        
        # Initialize the RWW Model 
        #
        # INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: RWW_Params - The parameters that all nodes in the network will share
        #  Con_Mtx: Tensor [num_regions, num_regions] - With connectivity (eg. structural connectivity)
            
        self.num_regions = num_regions
        self.params = params
        self.Con_Mtx = Con_Mtx
        self.Dist_Mtx = Dist_Mtx
        
        self.step_size = step_size
        
        
    def forward(self, external, hx, hE, withOptVars = False):
                
        # Runs the RWW Model 
        #
        # INPUT
        #  init_state: Tensor [regions, state_vars] # Regions is number of nodes and should match self.num_regions. There are 2 state variables. 
        #  step_size: Float - The step size in msec 
        #  sim_len: Int - The length of time to simulate in msec
        #  withOptVars: Boolean - Whether to include the Current and Firing rate variables of excitatory and inhibitory populations in layer_history
        #
        # OUTPUT
        #  state_vars:  Tensor - [regions, state_vars]
        #  layer_history: Tensor - [time_steps, regions, state_vars (+ opt_params)]
        #
        
        
        # Defining NMM Parameters to simplify later equations
        G = self.params.G.npValue()
        Lambda = self.params.Lambda.npValue()       # 1 or 0 depending on using long range feed forward inhibition (FFI)
        a_E = self.params.a_E.npValue()             # nC^(-1)
        b_E = self.params.b_E.npValue()             # Hz
        d_E = self.params.d_E.npValue()             # s
        tau_E = self.params.tau_E.npValue()         # ms
        tau_NMDA = self.params.tau_NMDA.npValue()   # ms
        W_E = self.params.W_E.npValue()
        a_I = self.params.a_I.npValue()             # nC^(-1)
        b_I = self.params.b_I.npValue()             # Hz
        d_I = self.params.d_I.npValue()             # s
        tau_I = self.params.tau_I.npValue()         # ms
        tau_GABA = self.params.tau_GABA.npValue()   # ms
        W_I = self.params.W_I.npValue()
        w_plus = self.params.w_plus.npValue()       # Local excitatory recurrence
        J_NMDA = self.params.J_NMDA.npValue()       # Excitatory synaptic coupling in nA
        J = self.params.J.npValue()                 # Local feedback inhibitory synaptic coupling. 1 in no-FIC case, different in FIC case #TODO: Currently set to J_NMDA but should calculate based on paper
        gamma = self.params.gamma.npValue()         # a kinetic parameter in ms
        sig = self.params.sig.npValue()             # 0.01 # Noise amplitude at node in nA
        I_0 = self.params.I_0.npValue()             # The overall effective external input in nA
        I_external = self.params.I_external.npValue() #External input current 
        gammaI = self.params.gammaI.npValue()
        J_new = self.params.J_new.npValue()
        
        
        def H_for_E_Vnp(I_E):
            
            numer = (a_E*I_E - b_E) 
            denom = (1 - numpy.exp(-d_E*(a_E*I_E - b_E)))

            r_E = numer / denom
            
            return r_E
            
        def H_for_I_Vnp(I_I):
            
            numer = (a_I*I_I - b_I) 
            denom = (1 - numpy.exp(-d_I*(a_I*I_I - b_I)))
            
            r_I = numer / denom
            
            return r_I

        init_state = hx
        sim_len = self.sim_len     
        step_size = self.step_size
        
        Ev = numpy.random.normal(0,1,size = (len(numpy.arange(0, sim_len, step_size)), self.num_regions))
        Iv = numpy.random.normal(0,1,size = (len(numpy.arange(0, sim_len, step_size)), self.num_regions))
        state_hist = numpy.zeros((int(sim_len/step_size), self.num_regions, 2))
        if(withOptVars):
            opt_hist = numpy.zeros((int(sim_len/step_size), self.num_regions, 4))
        
        # RWW and State Values
        S_E = init_state[:, 0]
        S_I = init_state[:, 1]
        
        num_steps = int(sim_len/step_size)
        for i in range(num_steps):
        
            Network_S_E =  numpy.matmul(self.Con_Mtx, S_E)
        
            # Currents
            I_E = W_E*I_0 + w_plus*J_NMDA*S_E + G*J_NMDA*Network_S_E - J*S_I + I_external
            I_I = W_I*I_0 + J_NMDA*S_E - J_new*S_I + Lambda*G*J_NMDA*Network_S_E
            
            # Firing Rates
            # Orig
            # r_E = (self.a_E*I_E - self.b_E) / (1 - numpy.exp(-self.d_E*(self.a_E*I_E - self.b_E)))
            # r_I = (self.a_I*I_I - self.b_I) / (1 - numpy.exp(-self.d_I*(self.a_I*I_I - self.b_I)))
            
            # EDIT: Version to address numpy.exp() returning nan
            r_E = H_for_E_Vnp(I_E)
            r_I = H_for_I_Vnp(I_I)
            
            # Average Synaptic Gating Variable
            dS_E = - S_E/tau_E + (1 - S_E)*gamma*r_E #+ self.sig*v_of_T[i, :] Noise now added later
            dS_I = - S_I/tau_I + gammaI*r_I #+ self.sig*v_of_T[i, :] Noise now added later
            
            # UPDATE VALUES
            S_E = S_E + step_size*dS_E + sqrt(step_size)*sig*Ev[i, :]
            S_I = S_I + step_size*dS_I + sqrt(step_size)*sig*Iv[i, :]
            
            state_hist[i, :, 0] = S_E
            state_hist[i, :, 1] = S_I 
            
            if(withOptVars):
                opt_hist[i, :, 0] = I_I
                opt_hist[i, :, 1] = I_E
                opt_hist[i, :, 2] = r_I
                opt_hist[i, :, 3] = r_E
            
        state_vals = numpy.stack((S_E, S_I)).transpose()
        
        if(withOptVars):
            layer_hist = numpy.cat((state_hist, opt_hist), 2)
        else:
            layer_hist = state_hist
        
        sim_vals = {}
        sim_vals["NMM_state"] = state_vals
        sim_vals["E"] = layer_hist[:,:,0]
        sim_vals["I"] = layer_hist[:,:,1]
        
        return sim_vals, hE #state_vals, layer_hist
        
