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

class BOLD_np():
    def __init__(self, num_regions, params):        
                
        # Initialize the BOLD Model 
        #
        # INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: BOLD_Params - The parameters that all nodes in the network will share
        
        self.num_regions = num_regions
        
        self.params = params
        
        
    def forward(self, init_state, step_size, sim_len, node_history, useGPU = False):
    
        hE = np.array(1.0) #Dummy variable
        
        # Runs the BOLD Model
        #
        # INPUT
        #  init_state: Array [regions, state_vars] # Number of regions should match node_history. There are 4 state variables. 
        #  step_size: Float - The step size in msec which must match node_history step size.
        #                     (NOTE: bold equations are in sec so step_size will be divide by 1000)
        #  sim_len: Int - The amount of BOLD to simulate in msec, and should match time simulated in node_history. 
        #  node_history: Array - [time_points, regions] # This would be S_E if input coming from RWW
        #  useGPU: Boolean - Whether to run on GPU or CPU - default is CPU and GPU has not been tested for Network_NMM code
        #
        # OUTPUT
        #  state_vars: Array - [regions, state_vars]
        #  layer_history: Array - [time_steps, regions, state_vars + 1 (BOLD)]
        #
        
        # Defining parameters to simplify later equations
        kappa = self.params.kappa.npValue()    # Rate of signal decay (1/s)
        gammaB = self.params.gammaB.npValue()  # Rate of flow-dependent elimination (1/s)
        tao = self.params.tao.npValue()        # Hemodynamic transit time (s)
        alpha = self.params.alpha.npValue()    # Grubb's exponent
        ro = self.params.ro.npValue()          #Resting oxygen extraction fraction
        V_0 = self.params.V_0.npValue()
        k_1 = self.params.k_1.npValue()
        k_2 = self.params.k_2.npValue()
        k_3 = self.params.k_3.npValue()
        
        
        layer_hist = numpy.zeros((int(sim_len/step_size), self.num_regions, 4 + 1))
        
        # BOLD State Values
        x = init_state[:, 0]
        f = init_state[:, 1]
        v = init_state[:, 2]
        q = init_state[:, 3]
        
        num_steps = int(sim_len/step_size)
        for i in range(num_steps):
            
            z = node_history[i,:] 
            
            #BOLD State Variables
            dx = z - kappa*x - gammaB*(f - 1)
            df = x
            dv = (f - v**(1/alpha))/tao
            dq = ((f/ro) * (1 - (1 - ro)**(1/f)) - q*v**(1/alpha - 1))/tao
            
            # UPDATE VALUES
            # NOTE: bold equations are in sec so step_size will be divide by 1000
            x = x + step_size/1000*dx
            f = f + step_size/1000*df
            v = v + step_size/1000*dv
            q = q + step_size/1000*dq
                       
            #BOLD Calculation
            BOLD = V_0*(k_1*(1 - q) + k_2*(1 - q/v) + k_3*(1 - v))
            
            layer_hist[i, :, 0] = x
            layer_hist[i, :, 1] = f
            layer_hist[i, :, 2] = v
            layer_hist[i, :, 3] = q
            layer_hist[i, :, 4] = BOLD
            
        state_vals = numpy.stack((x, f, v, q)).transpose()
        
        sim_vals = {}
        sim_vals["BOLD_state"] = state_vals
        sim_vals["x"] = layer_hist[:,:,0]
        sim_vals["f"] = layer_hist[:,:,1]
        sim_vals["v"] = layer_hist[:,:,2]
        sim_vals["q"] = layer_hist[:,:,3]
        sim_vals["bold"] = layer_hist[:,:,4]
        
        return sim_vals, hE #state_vals, layer_hist