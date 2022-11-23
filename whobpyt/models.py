"""
=====================
WhoBPyt Model Classes
=====================

For each model 'M', two classes are defined: 

- `M_Params` - Parameters class. Parameters to be fit should be overwritten after initialization as PyTorch parameters. 
- `M_Layer`  - Model implementation. Includes in particular a `forward()` method that implements numerical integration. 

"""

import torch

class RWW_Params():
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
        self.G = 1
        ##self.SUM_Cij_Sj_E = 0
        self.Lambda = 0 #1 or 0 depending on using long range feed forward inhibition (FFI)

        #Excitatory Gating Variables
        self.a_E = 310               # nC^(-1)
        self.b_E = 125               # Hz
        self.d_E = 0.16              # s
        self.tau_E = self.tau_NMDA = 100  # ms
        self.W_E = 1
        
        #Inhibitory Gating Variables
        self.a_I = 615               # nC^(-1)
        self.b_I = 177               # Hz
        self.d_I = 0.087             # s
        self.tau_I = self.tau_GABA = 10   # ms
        self.W_I = 0.7
        
        #Setting other variables
        self.w_plus = 1.4 # Local excitatory recurrence
        self.J_NMDA = 0.15 # Excitatory synaptic coupling in nA
        self.J = 0.15 * torch.ones(num_regions) # Local feedback inhibitory synaptic coupling. 1 in no-FIC case, different in FIC case #TODO: Currently set to J_NMDA but should calculate based on paper
        self.gamma = 0.641/1000 #a kinetic parameter in ms
        self.sig = 0.01 #0.01 # Noise amplitude at node in nA
        #self.v_of_T = 0 # Uncorrelated standarg Gaussian noise  # NOTE: Now defined at time of running forward model
        self.I_0 = 0.382 # The overall effective external input in nA
        
        self.I_external = 0.00 #External input current 
        
        #Starting Condition
        #S_E = 0.25 # The average synaptic gating variable of excitatory 
        #S_I = 0.25 # The average synaptic gating variable of inhibitory
        
        #############################################
        ## Model Additions/modifications
        #############################################
        
        self.gammaI = 1/1000 #Zheng suggested this to get oscillations
        
        
class RWW_Layer(torch.nn.Module):
    ## EQUATIONS & BIOLOGICAL VARIABLES FROM:
    #
    # Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitationâ€“inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
    # Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
    # Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
    # Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.  
  
    def __init__(self, num_regions, params, Con_Mtx, Dist_Mtx, step_size = 0.0001, useBC = False):        
        super(RWW_Layer, self).__init__() # To inherit parameters attribute
        
        # Initialize the RWW Model 
        #
        #  INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: RWW_Params - The parameters that all nodes in the network will share
        #  Con_Mtx: Tensor [num_regions, num_regions] - With connectivity (eg. structural connectivity)
        #  Dist_Mtx: Tensor [num_regions, num_regions]
        #  step_size: Float - The step size in msec 
        #  useBC: Boolean - Whether to use extra boundary conditions to make more numerically stable. Not fully tested.
        #                   NOTE: This is discouraged as it will likely influence results. Instead, choose a smaller step size. 

        self.step_size = step_size
        
        self.num_regions = num_regions
        self.Con_Mtx = Con_Mtx
        self.Dist_Mtx = Dist_Mtx
        
        self.max_delay = 100 #msec #This should be greater than what is possible of max(Dist_Mtx)/velocity
        self.buffer_len = int(self.max_delay/self.step_size)
        self.delayed_S_E = torch.zeros(self.buffer_len, num_regions)

        self.buffer_idx = 0
        self.mu = torch.tensor([0]) #Connection Speed addition

        #############################################
        ## RWW Constants
        #############################################
        
        #Zeroing the components which deal with a connected network
        self.G = params.G
        ##self.SUM_Cij_Sj_E = params.SUM_Cij_Sj_E
        self.Lambda = params.Lambda #1 or 0 depending on using long range feed forward inhibition (FFI)
        
        #Excitatory Gating Variables
        self.a_E = params.a_E               # nC^(-1)
        self.b_E = params.b_E               # Hz
        self.d_E = params.d_E              # s
        self.tau_E = params.tau_E   # ms
        self.tau_NMDA = params.tau_NMDA  # ms
        self.W_E = params.W_E
        
        #Inhibitory Gating Variables
        self.a_I = params.a_I               # nC^(-1)
        self.b_I = params.b_I               # Hz
        self.d_I = params.d_I             # s
        self.tau_I = params.tau_I    # ms
        self.tau_GABA = params.tau_GABA   # ms
        self.W_I = params.W_I
        
        #Setting other variables
        self.w_plus = params.w_plus # Local excitatory recurrence
        self.J_NMDA = params.J_NMDA # Excitatory synaptic coupling in nA
        self.J = params.J # Local feedback inhibitory synaptic coupling. 1 in no-FIC case, different in FIC case #TODO: Currently set to J_NMDA but should calculate based on paper
        self.gamma = params.gamma #a kinetic parameter in ms
        self.sig = params.sig #0.01 # Noise amplitude at node in nA
        self.v_of_T = None #params.v_of_T # Uncorrelated standarg Gaussian noise # NOTE: Now set at time of running forward model
        self.I_0 = params.I_0 # The overall effective external input in nA
        
        self.I_external = params.I_external #External input current 
        
        #Starting Condition
        #S_E = 0.25 # The average synaptic gating variable of excitatory 
        #S_I = 0.25 # The average synaptic gating variable of inhibitory
        
        #############################################
        ## Model Additions/modifications
        #############################################
        
        self.gammaI = params.gammaI
        
        #############################################
        ## Other
        #############################################
        
        self.useBC = useBC   #useBC: is if we want the model to use boundary conditions
        
    def H_for_E_V3(self, I_E, update = False):
        
        numer = torch.abs(self.a_E*I_E - self.b_E) + 1e-9*1
        denom = torch.where((-self.d_E*(self.a_E*I_E - self.b_E) > 50), 
                            torch.abs(1 - 1e9*(-self.d_E*(self.a_E*I_E - self.b_E))) + 1e-9*self.d_E,
                            torch.abs(1 - torch.exp(torch.min(-self.d_E*(self.a_E*I_E - self.b_E), torch.tensor([51])))) + 1e-9*self.d_E)
        r_E = numer / denom
        
        return r_E
        
    def H_for_I_V3(self, I_I, update = False):
        
        numer = torch.abs(self.a_I*I_I - self.b_I) + 1e-9*1
        denom = torch.where((-self.d_I*(self.a_I*I_I - self.b_I) > 50),
                            torch.abs(1 - 1e5*(-self.d_I*(self.a_I*I_I - self.b_I))) + 1e-9*self.d_I,
                            torch.abs(1 - torch.exp(torch.min(-self.d_I*(self.a_I*I_I - self.b_I), torch.tensor([51])))) + 1e-9*self.d_I)
        r_I = numer / denom
        
        return r_I
    
    def forward(self, init_state, sim_len, useDelays = False, useLaplacian = False, withOptVars = False, useGPU = False, debug = False):
                
        # Runs the RWW Model 
        #
        # INPUT
        #  init_state: Tensor [regions, state_vars] # Regions is number of nodes and should match self.num_regions. There are 2 state variables. 
        #  sim_len: Int - The length of time to simulate in msec
        #  withOptVars: Boolean - Whether to include the Current and Firing rate variables of excitatory and inhibitory populations in layer_history
        #  useGPU:  Boolean - Whether to run on GPU or CPU - default is CPU and GPU has not been tested for Network Code
        #
        # OUTPUT
        #  state_vars:  Tensor - [regions, state_vars]
        #  layer_history: Tensor - [time_steps, regions, state_vars (+ opt_params)]
        #
        
        if(useGPU):
            v_of_T = torch.normal(0,1,size = (len(torch.arange(0, sim_len, self.step_size)), self.num_regions)).cuda()
            state_hist = torch.zeros(int(sim_len/self.step_size), self.num_regions, 2).cuda()
            if(withOptVars):
                opt_hist = torch.zeros(int(sim_len/self.step_size), self.num_regions, 4).cuda()
        else:
            v_of_T = torch.normal(0,1,size = (len(torch.arange(0, sim_len, self.step_size)), self.num_regions))
            state_hist = torch.zeros(int(sim_len/self.step_size), self.num_regions, 2)
            if(withOptVars):
                opt_hist = torch.zeros(int(sim_len/self.step_size), self.num_regions, 4)
        
        # RWW and State Values
        S_E = init_state[:, 0]
        S_I = init_state[:, 1]

        num_steps = int(sim_len/self.step_size)
        for i in range(num_steps):
            
            if((not useDelays) & (not useLaplacian)):
                Network_S_E =  torch.matmul(self.Con_Mtx, S_E)

            if(useDelays & (not useLaplacian)):
                # WARNING: This has not been tested
                
                speed = (1.5 + torch.nn.functional.relu(self.mu)) * (self.step_size * 0.001)
                self.delays_idx = (self.Dist_Mtx / speed).type(torch.int64) #TODO: What is the units of the distance matrix then? Needs to be in meters?

                S_E_history_new = self.delayed_S_E # TODO: Check if this needs to be cloned to work
                S_E_delayed_Mtx = S_E_history_new.gather(0, (self.buffer_idx - self.delays_idx)%self.buffer_len)  # delayed E #TODO: Is distance matrix symmetric, should this be transposed?

                Delayed_S_E = torch.sum(torch.mul(self.Con_Mtx, S_E_delayed_Mtx), 1) # weights on delayed E

                Network_S_E = Delayed_S_E

            if(useLaplacian & (not useDelays)):
                # WARNING: This has not been tested
                
                # NOTE: We are acutally using the NEGATIVE Laplacian
                
                Laplacian_diagonal = -torch.diag(torch.sum(self.Con_Mtx, axis=1))    #Con_Mtx should be normalized, so this should just add a diagonal of -1's
                S_E_laplacian = torch.matmul(self.Con_Mtx + Laplacian_diagonal, S_E)

                Network_S_E = S_E_laplacian 


            if(useDelays & useLaplacian):
                # WARNING: This has not been tested
                
                # NOTE: We are acutally using the NEGATIVE Laplacian

                Laplacian_diagonal = -torch.diag(torch.sum(self.Con_Mtx, axis=1))    #Con_Mtx should be normalized, so this should just add a diagonal of -1's
                           
                speed = (1.5 + torch.nn.functional.relu(self.mu)) * (self.step_size * 0.001)
                self.delays_idx = (self.Dist_Mtx / speed).type(torch.int64) #TODO: What is the units of the distance matrix then?
                
                S_E_history_new = self.delayed_S_E # TODO: Check if this needs to be cloned to work
                S_E_delayed_Mtx = S_E_history_new.gather(0, (self.buffer_idx - self.delays_idx)%self.buffer_len) 
                
                S_E_delayed_Vector = torch.sum(torch.mul(self.Con_Mtx, S_E_delayed_Mtx), 1) # weights on delayed E
                
                Delayed_Laplacian_S_E = (S_E_delayed_Vector + torch.matmul(Laplacian_diagonal, S_E))
                
                Network_S_E = Delayed_Laplacian_S_E
                


            # Currents
            I_E = self.W_E*self.I_0 + self.w_plus*self.J_NMDA*S_E + self.G*self.J_NMDA*Network_S_E - self.J*S_I + self.I_external
            I_I = self.W_I*self.I_0 + self.J_NMDA*S_E - S_I + self.Lambda*self.G*self.J_NMDA*Network_S_E
            
            # Firing Rates
            # Orig
            #r_E = (self.a_E*I_E - self.b_E) / (1 - torch.exp(-self.d_E*(self.a_E*I_E - self.b_E)))
            #r_I = (self.a_I*I_I - self.b_I) / (1 - torch.exp(-self.d_I*(self.a_I*I_I - self.b_I)))
            #
            # EDIT5: Version to address torch.exp() returning nan and prevent gradient returning 0, and works with autograd
            r_E = self.H_for_E_V3(I_E)
            r_I = self.H_for_I_V3(I_I)
            
            # Average Synaptic Gating Variable
            dS_E = - S_E/self.tau_E + (1 - S_E)*self.gamma*r_E + self.sig*v_of_T[i, :]
            dS_I = - S_I/self.tau_I + self.gammaI*r_I + self.sig*v_of_T[i, :]
            
            # UPDATE VALUES
            S_E = S_E + self.step_size*dS_E
            S_I = S_I + self.step_size*dS_I
            
            # Bound the possible values of state variables (From fit.py code for numerical stability)
            if(self.useBC):
                S_E = torch.tanh(0.00001 + torch.nn.functional.relu(S_E - 0.00001))
                S_I = torch.tanh(0.00001 + torch.nn.functional.relu(S_I - 0.00001))
            
            state_hist[i, :, 0] = S_E
            state_hist[i, :, 1] = S_I 

            if useDelays:
                self.delayed_S_E = self.delayed_S_E.clone(); self.delayed_S_E[self.buffer_idx, :] = S_E #TODO: This means that not back-propagating the network just the individual nodes

                if (self.buffer_idx == (self.buffer_len - 1)):
                    self.buffer_idx = 0
                else: 
                    self.buffer_idx = self.buffer_idx + 1
            
            if(withOptVars):
                opt_hist[i, :, 0] = I_I
                opt_hist[i, :, 1] = I_E
                opt_hist[i, :, 2] = r_I
                opt_hist[i, :, 3] = r_E
            
        state_vals = torch.cat((torch.unsqueeze(S_E, 1), torch.unsqueeze(S_I, 1)), 1)
        
        # So that RAM does not accumulate in later batches/epochs 
        # & Because can't back pass through twice
        self.delayed_S_E = self.delayed_S_E.detach() 

        if(withOptVars):
            layer_hist = torch.cat((state_hist, opt_hist), 2)
        else:
            layer_hist = state_hist
        
        return state_vals, layer_hist
        
    





        
class EEG_Params():
    def __init__(self, Lead_Field):
        
        #############################################
        ## EEG Lead Field
        #############################################
        
        self.LF = Lead_Field # This should be [num_regions, num_channels]
        
        
class EEG_Layer():
    def __init__(self, num_regions, params, num_channels):        
        super(EEG_Layer, self).__init__() # To inherit parameters attribute
                
        # Initialize the EEG Model 
        #
        # INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: EEG_Params - This contains the EEG Parameters, to maintain a consistent paradigm
        
        self.num_regions = num_regions
        self.num_channels = num_channels
        
        #############################################
        ## EEG Lead Field
        #############################################
        
        self.LF = params.LF
        
    def forward(self, step_size, sim_len, node_history, useGPU = False):
        
        # Runs the EEG Model
        #
        # INPUT
        #  step_size: Float - The step size in msec which must match node_history step size.
        #  sim_len: Int - The amount of EEG to simulate in msec, and should match time simulated in node_history. 
        #  node_history: Tensor - [time_points, regions, two] # This would be S_E and S_I if input coming from RWW
        #  useGPU: Boolean - Whether to run on GPU or CPU - default is CPU and GPU has not been tested for Network_NMM code
        #
        # OUTPUT
        #  layer_history: Tensor - [time_steps, regions, one]
        #
        
        if(useGPU):
            layer_hist = torch.zeros(int(sim_len/step_size), self.num_channels, 1).cuda()
        else:
            layer_hist = torch.zeros(int(sim_len/step_size), self.num_channels, 1)
        
        num_steps = int(sim_len/step_size)
        for i in range(num_steps):
            layer_hist[i, :, 0] = torch.matmul(self.LF, node_history[i, :, 0] - node_history[i, :, 1])
            
        return layer_hist


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
        
        #Starting Condition
        #x = 1   # vasodilatory signal
        #f = 1   # inflow
        #v = 1   # blood volumne
        #q = 1   # deoxyhemoglobin content 
        
class BOLD_Layer(torch.nn.Module):
    def __init__(self, num_regions, params, useBC = False):        
        super(BOLD_Layer, self).__init__() # To inherit parameters attribute
                
        # Initialize the BOLD Model 
        #
        # INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: BOLD_Params - The parameters that all nodes in the network will share
        #  useBC: Boolean - Whether to use extra boundary conditions to make more numerically stable. Not fully tested.
        #                   NOTE: This is discouraged as it will likely influence results. Instead, choose a smaller step size. 
        
        self.num_regions = num_regions
        
        #############################################
        ## BOLD Constants
        #############################################
        
        #Friston 2003 - Table 1 - Priors on biophysical parameters
        self.kappa = params.kappa # Rate of signal decay (1/s)
        self.gammaB = params.gammaB # Rate of flow-dependent elimination (1/s)
        self.tao = params.tao # Hemodynamic transit time (s)
        self.alpha = params.alpha # Grubb's exponent
        self.ro = params.ro #Resting oxygen extraction fraction
        
        self.V_0 = params.V_0
        self.k_1 = params.k_1
        self.k_2 = params.k_2
        self.k_3 = params.k_3
        
        #Starting Condition
        #x = 1   # vasodilatory signal
        #f = 1   # inflow
        #v = 1   # blood volumne
        #q = 1   # deoxyhemoglobin content 
        
        #############################################
        ## Other
        #############################################
        
        self.useBC = useBC   #useBC: is if we want the model to use boundary conditions
        
        
    def forward(self, init_state, step_size, sim_len, node_history, useGPU = False):
        
        # Runs the BOLD Model
        #
        # INPUT
        #  init_state: Tensor [regions, state_vars] # Number of regions should match node_history. There are 4 state variables. 
        #  step_size: Float - The step size in msec which must match node_history step size.
        #                     (NOTE: bold equations are in sec so step_size will be divide by 1000)
        #  sim_len: Int - The amount of BOLD to simulate in msec, and should match time simulated in node_history. 
        #  node_history: Tensor - [time_points, regions] # This would be S_E if input coming from RWW
        #  useGPU: Boolean - Whether to run on GPU or CPU - default is CPU and GPU has not been tested for Network_NMM code
        #
        # OUTPUT
        #  state_vars: Tensor - [regions, state_vars]
        #  layer_history: Tensor - [time_steps, regions, state_vars + 1 (BOLD)]
        #
        
        if(useGPU):
            layer_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 4 + 1).cuda()
        else:
            layer_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 4 + 1)
        
        # BOLD State Values
        x = init_state[:, 0]
        f = init_state[:, 1]
        v = init_state[:, 2]
        q = init_state[:, 3]
        
        num_steps = int(sim_len/step_size)
        for i in range(num_steps):
            
            z = node_history[i,:] 
            
            #BOLD State Variables
            dx = z - self.kappa*x - self.gammaB*(f - 1)
            df = x
            dv = (f - v**(1/self.alpha))/self.tao
            dq = ((f/self.ro) * (1 - (1 - self.ro)**(1/f)) - q*v**(1/self.alpha - 1))/self.tao
            
            # UPDATE VALUES
            # NOTE: bold equations are in sec so step_size will be divide by 1000
            x = x + step_size/1000*dx
            f = f + step_size/1000*df
            v = v + step_size/1000*dv
            q = q + step_size/1000*dq
            
            # Bound the possible values of state variables (From fit.py code for numerical stability)
            if(self.useBC):
                x = torch.tanh(x)
                f = (1 + torch.tanh(f - 1))
                v = (1 + torch.tanh(v - 1))
                q = (1 + torch.tanh(q - 1))
            
            #BOLD Calculation
            BOLD = self.V_0*(self.k_1*(1 - q) + self.k_2*(1 - q/v) + self.k_3*(1 - v))
            
            layer_hist[i, :, 0] = x
            layer_hist[i, :, 1] = f
            layer_hist[i, :, 2] = v
            layer_hist[i, :, 3] = q
            layer_hist[i, :, 4] = BOLD
            
        state_vals = torch.cat((torch.unsqueeze(x, 1), torch.unsqueeze(f, 1), torch.unsqueeze(v, 1), torch.unsqueeze(q, 1)),1)
        
        return state_vals, layer_hist

### New linear version of NMM

### zheng's version
#from Model_pytorch import wwd_model_pytorch_new
import matplotlib.pyplot as plt # for plotting
import numpy as np # for numerical operations
import pandas as pd # for data manipulation
import seaborn as sns # for plotting 
import time # for timer
import torch
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from torch.nn.parameter import Parameter


class OutputNM():
    mode_all = ['train', 'test']
    stat_vars_all = ['m', 'v']

    def __init__(self, model_name, node_size, param, fit_weights=False, fit_lfm=False):
        self.loss = np.array([])
        if model_name == 'WWD':
            state_names = ['E', 'I', 'x', 'f', 'v', 'q']
            self.output_name = "bold"
        elif model_name == "JR":
            state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
            self.output_name = "eeg"
        for name in state_names + [self.output_name]:
            for m in self.mode_all:
                setattr(self, name + '_' + m, [])

        vars = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars:
            if np.any(getattr(param, var)[1] > 0):
                if var != 'std_in':
                    setattr(self, var, np.array([]))
                    for stat_var in self.stat_vars_all:
                        setattr(self, var + '_' + stat_var, [])
                else:
                    setattr(self, var, [])
        if fit_weights == True:
            self.weights = []
        if model_name == 'JR' and fit_lfm == True:
            self.leadfield = []

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class ParamsJR():

    def __init__(self, model_name, **kwargs):
        if model_name == 'WWD':
            param = {

                "std_in": [0.02, 0],  # standard deviation of the Gaussian noise
                "std_out": [0.02, 0],  # standard deviation of the Gaussian noise
                # Parameters for the ODEs
                # Excitatory population
                "W_E": [1., 0],  # scale of the external input
                "tau_E": [100., 0],  # decay time
                "gamma_E": [0.641 / 1000., 0],  # other dynamic parameter (?)

                # Inhibitory population
                "W_I": [0.7, 0],  # scale of the external input
                "tau_I": [10., 0],  # decay time
                "gamma_I": [1. / 1000., 0],  # other dynamic parameter (?)

                # External input
                "I_0": [0.32, 0],  # external input
                "I_external": [0., 0],  # external stimulation

                # Coupling parameters
                "g": [20., 0],  # global coupling (from all nodes E_j to single node E_i)
                "g_EE": [.1, 0],  # local self excitatory feedback (from E_i to E_i)
                "g_IE": [.1, 0],  # local inhibitory coupling (from I_i to E_i)
                "g_EI": [0.1, 0],  # local excitatory coupling (from E_i to I_i)

                "aE": [310, 0],
                "bE": [125, 0],
                "dE": [0.16, 0],
                "aI": [615, 0],
                "bI": [177, 0],
                "dI": [0.087, 0],

                # Output (BOLD signal)

                "alpha": [0.32, 0],
                "rho": [0.34, 0],
                "k1": [2.38, 0],
                "k2": [2.0, 0],
                "k3": [0.48, 0],  # adjust this number from 0.48 for BOLD fluctruate around zero
                "V": [.02, 0],
                "E0": [0.34, 0],
                "tau_s": [0.65, 0],
                "tau_f": [0.41, 0],
                "tau_0": [0.98, 0],
                "mu": [0.5, 0]

            }
        elif model_name == "JR":
            param = {
                "A ": [3.25, 0], "a": [100, 0.], "B": [22, 0], "b": [50, 0], "g": [1000, 0], \
                "c1": [135, 0.], "c2": [135 * 0.8, 0.], "c3 ": [135 * 0.25, 0.], "c4": [135 * 0.25, 0.], \
                "std_in": [100, 0], "vmax": [5, 0], "v0": [6, 0], "r": [0.56, 0], "y0": [2, 0], \
                "mu": [.5, 0], "k": [5, 0], "cy0": [5, 0], "ki": [1, 0]
            }
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])
        
        
        
def sys2nd(A, a,  u, x, v):
    return A*a*u -2*a*v-a**2*x


class LinearRNN_Params():
    def __init__(self, num_regions):
        self.SC = torch.rand((num_regions, num_regions))/num_regions

class LinearRNN_Layer(torch.nn.Module):
    def __init__(self, num_regions, params, step_size = 0.0001, useBC = False):
        super(LinearRNN_Layer, self).__init__()
        self.step_size = step_size
        self.num_regions = num_regions
        self.SC = params.SC
    
    def forward(self, init_state, sim_len, useDelays = False, useLaplacian = True, withOptVars = False, useGPU = False, debug = False):
        
        whiteNoise = torch.normal(0, 1, size = (len(torch.arange(0, sim_len, self.step_size)), self.num_regions))
        
        state_hist = torch.zeros(int(sim_len/self.step_size), self.num_regions) # initializing state history vector
        E = init_state
        
        if(useLaplacian): # using laplacian to make signal more stable (make sure it does not explode)
            lap_sc = -(torch.diag(sum(self.SC,1))-self.SC)
            init_sc = lap_sc
        else:
            init_sc = self.sc
            
        num_steps = int(sim_len/self.step_size)        
        
        dt = torch.tensor(self.step_size) 
        
        for i in range(num_steps):
            
            E = E + (torch.matmul(init_sc, E))*dt + torch.sqrt(dt)*whiteNoise[i, :] # calculating current E value
            state_hist[i, :] = E 
            
        return state_hist, E
            
            
            


	
	
	
	
	
	
### New JR Shrey-Sorenza Draft

class JR_Params():
   def __init__(self, num_regions = 1): 
        #############################################
        ## JR Constants
        #############################################
	
	        
        #Zeroing the components which deal with a connected network
        self.G = 1
        ##self.SUM_Cij_Sj_E = 0
        self.Lambda = 0 #1 or 0 depending on using long range feed forward inhibition (FFI)
        self.A = 3.25 # magnitude of second order system for populations E and P
        self.a = 100 # decay rate of the 2nd order system for population E and P
        self.B = 22 # magnitude of second order system for population I
        self.b = 50 # decay rate of the 2nd order system for population I
        self.g= 1000 # global gain
        self.c1= 135 # local gain from P to E (pre)
        self.c2= 135 * 0.8 # local gain from P to E (post)
        self.c3= 135 * 0.25 # local gain from P to I
        self.c4= 135 * 0.25 # local gain from P to I
        self.mu = 0.5
        self.y0 = 2
        self.std_in= 100 # local gain from P to I
        self.cy0 = 5
        self.vmax = 5
        self.v0 = 6
        self.r = 0.56
        self.k = 1

class Jansen_Layer(torch.nn.Module):
    def __init__(self, num_regions, params, Con_Mtx, useBC = False):        
        super(Jansen_Layer, self).__init__() # To inherit parameters attribute
        
        # Initialize the RNN Model 
        #
        # INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: RWW_Params - The parameters that all nodes in the network will share
        #  Con_Mtx: Tensor [num_regions, num_regions] - With connectivity (eg. structural connectivity)
        #  useBC: Boolean - Whether to use extra boundary conditions to make more numerically stable. Not fully tested.
        #                   NOTE: This is discouraged as it will likely influence results. Instead, choose a smaller step size. 
        
        
        self.num_regions = num_regions
        self.Con_Mtx = Con_Mtx
        
        #############################################
        ## RNNJANSEN Constants
        #############################################
        
        #Zeroing the components which deal with a connected network
        self.G = params.G
        ##self.SUM_Cij_Sj_E = params.SUM_Cij_Sj_E
        self.Lambda = params.Lambda #1 or 0 depending on using long range feed forward inhibition (FFI)
        
        #############################################
        ## JR Constants
        #############################################
        self.A = params.A # magnitude of second order system for populations E and P
        self.a = params.a # decay rate of the 2nd order system for population E and P
        self.B = params.B # magnitude of second order system for population I
        self.b = params.b # decay rate of the 2nd order system for population I
        self.g= params.g # global gain
        self.c1= params.c1 # local gain from P to E (pre)
        self.c2= params.c2 # local gain from P to E (post)
        self.c3= params.c3 # local gain from P to I
        self.c4= params.c4 # local gain from P to I
        self.mu = params.mu
        self.y0 = params.y0
        self.std_in= params.std_in # local gain from P to I
        self.cy0 = params.cy0
        self.vmax = params.vmax
        self.v0 = params.v0
        self.r = params.r
        self.k = params.k

    	# std_in is noise input

        #Starting Condition
    	# Do we need for JR?
        #S_E = 0.25 # The average synaptic gating variable of excitatory 
        #S_I = 0.25 # The average synaptic gating variable of inhibitory
       
        
        #############################################
        ## Other
        #############################################
        
        self.useBC = useBC   #useBC: is if we want the model to use boundary conditions
        
    
    def forward(self, init_state, step_size, sim_len, withOptVars = False, useGPU = False, debug = False):
                
        # Runs the RNN Model 
        #
        # INPUT
        #  init_state: Tensor [regions, state_vars] # Regions is number of nodes and should match self.num_regions. There are 2 state variables. 
        #  step_size: Float - The step size in msec 
        #  sim_len: Int - The length of time to simulate in msec
        #  withOptVars: Boolean - Whether to include the Current and Firing rate variables of excitatory and inhibitory populations in layer_history
        #  useGPU:  Boolean - Whether to run on GPU or CPU - default is CPU and GPU has not been tested for Network Code
        #
        # OUTPUT
        #  state_vars:  Tensor - [regions, state_vars]
        #  layer_history: Tensor - [time_steps, regions, state_vars (+ opt_params)]
        #
        
        #if(useGPU):
        #    v_of_T = torch.normal(0,1,size = (len(torch.arange(0, sim_len, step_size)), self.num_regions)).cuda()
        #    state_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 2).cuda()
        #    if(withOptVars):
        #        opt_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 4).cuda()
        #else:
        #    v_of_T = torch.normal(0,1,size = (len(torch.arange(0, sim_len, step_size)), self.num_regions))
        #    state_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 2)
        #    if(withOptVars):
        opt_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 4)
        
        # JR and State Values
        M = init_state[:, 0]
        E = init_state[:, 1]
        I = init_state[:, 2]
        Mv = init_state[:, 3]
        Ev = init_state[:, 4]
        Iv = init_state[:, 5]
        num_steps = int(sim_len/step_size)
        # Might need to change the c to add global gain g
        for i in range(num_steps):    
            dM = Mv
            dMv = self.A*self.a*sigmoid(E - I, self.vmax, self.v0, self.r) - 2*self.a*Mv-M*self.a**(2)
            dE = Ev
            dEv = self.A*self.a*(std_in + self.c2*sigmoid(self.c1*M, self.vmax, self.v0, self.r)) - 2*self.a*Ev - E*self.a**(2)
            dI = Iv
            dIv = self.B*self.b*(self.c4*sigmoid(self.c3*M, self.vmax, self.v0, self.r)) - 2*self.b*Iv - I*self.b**(2)


            # UPDATE VALUES

            M = M + step_size*dM
            E = E + step_size*dE
            I = I + step_size*dI
            Mv = Mv + step_size*dMv
            Ev = Ev + step_size*dEv
            Iv = Iv + step_size*dIv
	        
	          # Not sure about this boundary			
            # Bound the possible values of state variables (From fit.py code for numerical stability)
            if(self.useBC):
                E = 1000*torch.tanh(dE/1000)#torch.tanh(0.00001+torch.nn.functional.relu(dE))
                I = 1000*torch.tanh(dI/1000)#torch.tanh(0.00001+torch.nn.functional.relu(dI))
                M = 1000*torch.tanh(dM/1000)
                Ev = 1000*torch.tanh(dEv/1000)#(con_1 + torch.tanh(df - con_1))
                Iv = 1000*torch.tanh(dIv/1000)#(con_1 + torch.tanh(dv - con_1))
                Mv = 1000*torch.tanh(dMv/1000)#(con_1 + torch.tanh(dq - con_1))

            state_hist[i, :, 0] = M
            state_hist[i, :, 1] = E 
            state_hist[i, :, 2] = I
            state_hist[i, :, 3] = Mv 
            state_hist[i, :, 4] = Ev
            state_hist[i, :, 5] = Iv
            
	          # Not sure if needed with JR
            #if(withOptVars):
            #   opt_hist[i, :, 0] = I_I
            #  opt_hist[i, :, 1] = I_E
            #   opt_hist[i, :, 2] = r_I
            #   opt_hist[i, :, 3] = r_E
            
            state_vals = torch.cat((torch.unsqueeze(M, 1), torch.unsqueeze(E, 1), torch.unsqueeze(I, 1), torch.unsqueeze(Mv, 1), torch.unsqueeze(Ev, 1), torch.unsqueeze(Iv, 1)), 1)
        
            #if(withOptVars):
            #    layer_hist = torch.cat((state_hist, opt_hist), 2)
            #else:
            #    layer_hist = state_hist
            
        
        return state_vals, layer_hist
 
def sigmoid(x, vmax, v0, r):
    return vmax/(1+torch.exp(r*(v0-x)))
        

### zheng's version
def sys2nd(A, a,  u, x, v):
    return A*a*u -2*a*v-a**2*x

#def sigmoid(x, vmax, v0, r):
#    return vmax/(1+torch.exp(r*(v0-x)))


class ParamsJR():

    def __init__(self, model_name, **kwargs):
        if model_name == 'WWD':
            param = {

                "std_in": [0.02, 0],  # standard deviation of the Gaussian noise
                "std_out": [0.02, 0],  # standard deviation of the Gaussian noise
                # Parameters for the ODEs
                # Excitatory population
                "W_E": [1., 0],  # scale of the external input
                "tau_E": [100., 0],  # decay time
                "gamma_E": [0.641 / 1000., 0],  # other dynamic parameter (?)

                # Inhibitory population
                "W_I": [0.7, 0],  # scale of the external input
                "tau_I": [10., 0],  # decay time
                "gamma_I": [1. / 1000., 0],  # other dynamic parameter (?)

                # External input
                "I_0": [0.32, 0],  # external input
                "I_external": [0., 0],  # external stimulation

                # Coupling parameters
                "g": [20., 0],  # global coupling (from all nodes E_j to single node E_i)
                "g_EE": [.1, 0],  # local self excitatory feedback (from E_i to E_i)
                "g_IE": [.1, 0],  # local inhibitory coupling (from I_i to E_i)
                "g_EI": [0.1, 0],  # local excitatory coupling (from E_i to I_i)

                "aE": [310, 0],
                "bE": [125, 0],
                "dE": [0.16, 0],
                "aI": [615, 0],
                "bI": [177, 0],
                "dI": [0.087, 0],

                # Output (BOLD signal)

                "alpha": [0.32, 0],
                "rho": [0.34, 0],
                "k1": [2.38, 0],
                "k2": [2.0, 0],
                "k3": [0.48, 0],  # adjust this number from 0.48 for BOLD fluctruate around zero
                "V": [.02, 0],
                "E0": [0.34, 0],
                "tau_s": [0.65, 0],
                "tau_f": [0.41, 0],
                "tau_0": [0.98, 0],
                "mu": [0.5, 0]

            }
        elif model_name == "JR":
            param = {
                "A ": [3.25, 0], "a": [100, 0.], "B": [22, 0], "b": [50, 0], "g": [1000, 0], \
                "c1": [135, 0.], "c2": [135 * 0.8, 0.], "c3 ": [135 * 0.25, 0.], "c4": [135 * 0.25, 0.], \
                "std_in": [100, 0], "vmax": [5, 0], "v0": [6, 0], "r": [0.56, 0], "y0": [2, 0], \
                "mu": [.5, 0], "k": [5, 0], "cy0": [5, 0], "ki": [1, 0]
            }
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])
        """self.A = A # magnitude of second order system for populations E and P
        self.a = a # decay rate of the 2nd order system for population E and P
        self.B = B # magnitude of second order system for population I
        self.b = b # decay rate of the 2nd order system for population I
        self.g= g # global gain
        self.c1= c1# local gain from P to E (pre)
        self.c2= c2 # local gain from P to E (post)
        self.c3= c3 # local gain from P to I
        self.c4= c4 # local gain from P to I
        self.mu = mu
        self.y0 = y0
        self.std_in= std_in # local gain from P to I
        self.cy0 = cy0
        self.vmax = vmax
        self.v0 = v0
        self.r = r
        self.k = k"""
	

class RNNJANSEN(torch.nn.Module):
    """
    A module for forward model (JansenRit) to simulate a batch of EEG signals
    Attibutes
    ---------
    state_size : int
        the number of states in the JansenRit model
    input_size : int
        the number of states with noise as input
    tr : float
        tr of image
    step_size: float
        Integration step for forward model
    hidden_size: int
        the number of step_size in a tr
    batch_size: int
        the number of EEG signals to simulate
    node_size: int
        the number of ROIs
    sc: float node_size x node_size array
        structural connectivity
    fit_gains: bool
        flag for fitting gains 1: fit 0: not fit
    g, c1, c2, c3,c4: tensor with gradient on
        model parameters to be fit
    w_bb: tensor with node_size x node_size (grad on depends on fit_gains)
        connection gains
    std_in std_out: tensor with gradient on
        std for state noise and output noise
    hyper parameters for prior distribution of model parameters
    Methods
    -------
    forward(input, noise_out, hx)
        forward model (JansenRit) for generating a number of EEG signals with current model parameters
    """
    state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
    model_name = "JR"

    def __init__(self, input_size: int, node_size: int,
                 batch_size: int, step_size: float, output_size: int, tr: float, sc: float, lm: float, dist: float,
                 fit_gains_flat: bool, \
                 fit_lfm_flat: bool, param: ParamsJR) -> None:
        """
        Parameters
        ----------
        state_size : int
        the number of states in the JansenRit model
        input_size : int
            the number of states with noise as input
        tr : float
            tr of image
        step_size: float
            Integration step for forward model
        hidden_size: int
            the number of step_size in a tr
        batch_size: int
            the number of EEG signals to simulate
        node_size: int
            the number of ROIs
        output_size: int
            the number of channels EEG
        sc: float node_size x node_size array
            structural connectivity
        fit_gains: bool
            flag for fitting gains 1: fit 0: not fit
        param from ParamJR
        """
        super(RNNJANSEN, self).__init__()
        self.state_size = 6  # 6 states WWD model
        self.input_size = input_size  # 1 or 2 or 3
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = torch.tensor(step_size, dtype=torch.float32)  # integration step 0.1 ms
        self.hidden_size = int(tr / step_size)
        self.batch_size = batch_size  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.output_size = output_size  # num of EEG channels
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.dist = torch.tensor(dist, dtype=torch.float32)
        self.fit_gains_flat = fit_gains_flat  # flag for fitting gains
        self.fit_lfm_flat = fit_lfm_flat
        self.param = param

        self.output_size = lm.shape[0]  # number of EEG channels

        # set model parameters (variables: need to calculate gradient) as Parameter others : tensor
        # set w_bb as Parameter if fit_gain is True
        if self.fit_gains_flat == True:
            self.w_bb = Parameter(torch.tensor(np.zeros((node_size, node_size)) + 0.05,
                                               dtype=torch.float32))  # connenction gain to modify empirical sc
        else:
            self.w_bb = torch.tensor(np.zeros((node_size, node_size)), dtype=torch.float32)

        if self.fit_lfm_flat == True:
            self.lm = Parameter(torch.tensor(lm, dtype=torch.float32))  # leadfield matrix from sourced data to eeg
        else:
            self.lm = torch.tensor(lm, dtype=torch.float32)  # leadfield matrix from sourced data to eeg

        vars = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars:
            if np.any(getattr(param, var)[1] > 0):
                setattr(self, var, Parameter(
                    torch.tensor(getattr(param, var)[0] + 1 / getattr(param, var)[1] * np.random.randn(1, )[0],
                                 dtype=torch.float32)))
                if var != 'std_in':
                    dict_nv = {}
                    dict_nv['m'] = getattr(param, var)[0]
                    dict_nv['v'] = getattr(param, var)[1]

                    dict_np = {}
                    dict_np['m'] = var + '_m'
                    dict_np['v'] = var + '_v'

                    for key in dict_nv:
                        setattr(self, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
            else:
                setattr(self, var, torch.tensor(getattr(param, var)[0], dtype=torch.float32))

    """def check_input(self, input: Tensor) -> None:
        expected_input_dim = 2 
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))
        if self.batch_size != input.size(0):
            raise RuntimeError(
                'input.size(0) must be equal to batch_size. Expected {}, got {}'.format(
                    self.batch_size, input.size(0)))"""

    def forward(self, input, hx, hE):
        """
        Forward step in simulating the EEG signal.
        Parameters
        ----------
        input: tensor with node_size x hidden_size x batch_size x input_size
            noise for states
        noise_out: tensor with node_size x batch_size
            noise for EEG
        hx: tensor with node_size x state_size
            states of JansenRit model
        Outputs
        -------
        next_state: dictionary with keys:
        'current_state''EEG_batch''E_batch''I_batch''M_batch''Ev_batch''Iv_batch''Mv_batch'
            record new states and EEG
        """

        # define some constants
        conduct_lb = 1.5  # lower bound for conduct velocity
        u_2ndsys_ub = 500  # the bound of the input for second order system
        noise_std_lb = 150  # lower bound of std of noise
        lb = 0.01  # lower bound of local gains
        s2o_coef = 0.0001  # coefficient from states (source EEG) to EEG
        k_lb = 0.5  # lower bound of coefficient of external inputs

        next_state = {}

        M = hx[:, 0:1]  # current of main population
        E = hx[:, 1:2]  # current of excitory population
        I = hx[:, 2:3]  # current of inhibitory population

        Mv = hx[:, 3:4]  # voltage of main population
        Ev = hx[:, 4:5]  # voltage of exictory population
        Iv = hx[:, 5:6]  # voltage of inhibitory population

        dt = self.step_size
        # Generate the ReLU module for model parameters gEE gEI and gIE

        m = torch.nn.ReLU()

        # define constant 1 tensor
        con_1 = torch.tensor(1.0, dtype=torch.float32)
        if self.sc.shape[0] > 1:

            # Update the Laplacian based on the updated connection gains w_bb.
            w = torch.exp(self.w_bb) * torch.tensor(self.sc, dtype=torch.float32)
            w_n = torch.log1p(0.5 * (w + torch.transpose(w, 0, 1))) / torch.linalg.norm(
                torch.log1p(0.5 * (w + torch.transpose(w, 0, 1))))
            self.sc_m = w_n
            dg = -torch.diag(torch.sum(w_n, axis=1))
        else:
            l_s = torch.tensor(np.zeros((1, 1)), dtype=torch.float32)

        self.delays = (self.dist / (conduct_lb * con_1 + m(self.mu))).type(torch.int64)
        # print(torch.max(self.delays), self.delays.shape)

        # placeholder for the updated corrent state
        current_state = torch.zeros_like(hx)

        # placeholders for output BOLD, history of E I x f v and q
        eeg_batch = []
        E_batch = []
        I_batch = []
        M_batch = []
        Ev_batch = []
        Iv_batch = []
        Mv_batch = []

        # Use the forward model to get EEGsignal at ith element in the batch.
        for i_batch in range(self.batch_size):
            # Get the noise for EEG output.
            

            for i_hidden in range(self.hidden_size):
                Ed = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)  # delayed E

                """for ind in range(self.node_size):
                    #print(ind, hE[ind,:].shape, self.delays[ind,:].shape)
                    Ed[ind] = torch.index_select(hE[ind,:], 0, self.delays[ind,:])"""
                hE_new = hE.clone()
                Ed = hE_new.gather(1, self.delays)  # delayed E

                LEd = torch.reshape(torch.sum(w_n * torch.transpose(Ed, 0, 1), 1),
                                    (self.node_size, 1))  # weights on delayed E

                # Input noise for M.
                
                u = input[:, i_hidden:i_hidden + 1, i_batch]

                # LEd+torch.matmul(dg,E): Laplacian on delayed E

                rM = sigmoid(E - I, self.vmax, self.v0, self.r)  # firing rate for Main population
                rE =  (noise_std_lb * con_1 + m(self.std_in)) * torch.randn((self.node_size, 1)) + (lb * con_1 + m(self.g)) * (
                            LEd + 1 * torch.matmul(dg, E)) \
                     + (lb * con_1 + m(self.c2)) * sigmoid((lb * con_1 + m(self.c1)) * M, self.vmax, self.v0,
                                                           self.r)  # firing rate for Excitory population
                rI = (lb * con_1 + m(self.c4)) * sigmoid((lb * con_1 + m(self.c3)) * M, self.vmax, self.v0,
                                                         self.r)  # firing rate for Inhibitory population

                # Update the states by step-size.
                ddM = M + dt * Mv
                ddE = E + dt * Ev
                ddI = I + dt * Iv
                ddMv = Mv + dt * sys2nd(0 * con_1 + m(self.A), 1 * con_1 + m(self.a),
                                        + u_2ndsys_ub * torch.tanh(rM / u_2ndsys_ub), M, Mv) 
                ddEv = Ev + dt * sys2nd(0 * con_1 + m(self.A), 1 * con_1 + m(self.a), \
                                        (k_lb * con_1 + m(self.k)) * u \
                                        + u_2ndsys_ub * torch.tanh(rE / u_2ndsys_ub), E, Ev) 

                ddIv = Iv + dt * sys2nd(0 * con_1 + m(self.B), 1 * con_1 + m(self.b), \
                                        +u_2ndsys_ub * torch.tanh(rI / u_2ndsys_ub), I, Iv) 

                # Calculate the saturation for model states (for stability and gradient calculation).
                E = ddE  # 1000*torch.tanh(ddE/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddE))
                I = ddI  # 1000*torch.tanh(ddI/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddI))
                M = ddM  # 1000*torch.tanh(ddM/1000)
                Ev = ddEv  # 1000*torch.tanh(ddEv/1000)#(con_1 + torch.tanh(df - con_1))
                Iv = ddIv  # 1000*torch.tanh(ddIv/1000)#(con_1 + torch.tanh(dv - con_1))
                Mv = ddMv  # 1000*torch.tanh(ddMv/1000)#(con_1 + torch.tanh(dq - con_1))

                # update placeholders for E buffer
                hE[:, 0] = E[:, 0]

            # Put M E I Mv Ev and Iv at every tr to the placeholders for checking them visually.
            M_batch.append(M)
            I_batch.append(I)
            E_batch.append(E)
            Mv_batch.append(Mv)
            Iv_batch.append(Iv)
            Ev_batch.append(Ev)
            hE = torch.cat([E, hE[:, :-1]], axis=1)  # update placeholders for E buffer

            # Put the EEG signal each tr to the placeholder being used in the cost calculation.
            lm_t = (self.lm - 1 / self.output_size * torch.matmul(torch.ones((1, self.output_size)), self.lm))
            temp = s2o_coef * self.cy0 * torch.matmul(lm_t, E - I) - 1 * self.y0
            eeg_batch.append(temp)  # torch.abs(E) - torch.abs(I) + 0.0*noiseEEG)

        # Update the current state.
        current_state = torch.cat([M, E, I, Mv, Ev, Iv], axis=1)
        next_state['current_state'] = current_state
        next_state['eeg_batch'] = torch.cat(eeg_batch, axis=1)
        next_state['E_batch'] = torch.cat(E_batch, axis=1)
        next_state['I_batch'] = torch.cat(I_batch, axis=1)
        next_state['P_batch'] = torch.cat(M_batch, axis=1)
        next_state['Ev_batch'] = torch.cat(Ev_batch, axis=1)
        next_state['Iv_batch'] = torch.cat(Iv_batch, axis=1)
        next_state['Pv_batch'] = torch.cat(Mv_batch, axis=1)

        return next_state, hE


class Costs:
    def __init__(self, method):
        self.method = method

    def cost_dist(self, sim, emp):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, the probability and negative log-likelihood.
        Parameters
        ----------
        logits_series_tf: tensor with node_size X datapoint
            simulated EEG
        labels_series_tf: tensor with node_size X datapoint
            empirical EEG
        """

        losses = torch.sqrt(torch.mean((sim - emp) ** 2))  #
        return losses

    def cost_r(self, logits_series_tf, labels_series_tf):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, the probability and negative log-likelihood.
        Parameters
        ----------
        logits_series_tf: tensor with node_size X datapoint
            simulated BOLD
        labels_series_tf: tensor with node_size X datapoint
            empirical BOLD
        """
        # get node_size(batch_size) and batch_size()
        node_size = logits_series_tf.shape[0]
        truncated_backprop_length = logits_series_tf.shape[1]

        # remove mean across time
        labels_series_tf_n = labels_series_tf - torch.reshape(torch.mean(labels_series_tf, 1),
                                                              [node_size, 1])  # - torch.matmul(

        logits_series_tf_n = logits_series_tf - torch.reshape(torch.mean(logits_series_tf, 1),
                                                              [node_size, 1])  # - torch.matmul(

        # correlation
        cov_sim = torch.matmul(logits_series_tf_n, torch.transpose(logits_series_tf_n, 0, 1))
        cov_def = torch.matmul(labels_series_tf_n, torch.transpose(labels_series_tf_n, 0, 1))

        # fc for sim and empirical BOLDs
        FC_sim_T = torch.matmul(torch.matmul(torch.diag(torch.reciprocal(torch.sqrt( \
            torch.diag(cov_sim)))), cov_sim),
            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_sim)))))
        FC_T = torch.matmul(torch.matmul(torch.diag(torch.reciprocal(torch.sqrt( \
            torch.diag(cov_def)))), cov_def),
            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_def)))))

        # mask for lower triangle without diagonal
        ones_tri = torch.tril(torch.ones_like(FC_T), -1)
        zeros = torch.zeros_like(FC_T)  # create a tensor all ones
        mask = torch.greater(ones_tri, zeros)  # boolean tensor, mask[i] = True iff x[i] > 1

        # mask out fc to vector with elements of the lower triangle
        FC_tri_v = torch.masked_select(FC_T, mask)
        FC_sim_tri_v = torch.masked_select(FC_sim_T, mask)

        # remove the mean across the elements
        FC_v = FC_tri_v - torch.mean(FC_tri_v)
        FC_sim_v = FC_sim_tri_v - torch.mean(FC_sim_tri_v)

        # corr_coef
        corr_FC = torch.sum(torch.multiply(FC_v, FC_sim_v)) \
                  * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_v, FC_v)))) \
                  * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_sim_v, FC_sim_v))))

        # use surprise: corr to calculate probability and -log
        losses_corr = -torch.log(0.5000 + 0.5 * corr_FC)  # torch.mean((FC_v -FC_sim_v)**2)#
        return losses_corr

    def cost_eff(self, sim, emp):
        if self.method == 0:
            return self.cost_dist(sim, emp)
        else:
            return self.cost_r(sim, emp)


def h_tf(a, b, d, z):
    """
    Neuronal input-output functions of excitatory pools and inhibitory pools.
    Take the variables a, x, and b and convert them to a linear equation (a*x - b) while adding a small
    amount of noise 0.00001 while dividing that term to an exponential of the linear equation multiplied by the
    d constant for the appropriate dimensions.
    """
    num = 0.00001 + torch.abs(a * z - b)
    den = 0.00001 * d + torch.abs(1.0000 - torch.exp(-d * (a * z - b)))
    return torch.divide(num, den)



	

class RNNWWD(torch.nn.Module):
    """
    A module for forward model (WWD) to simulate a batch of BOLD signals
    Attibutes
    ---------
    state_size : int
        the number of states in the WWD model
    input_size : int
        the number of states with noise as input
    tr : float
        tr of fMRI image
    step_size: float
        Integration step for forward model
    hidden_size: int
        the number of step_size in a tr
    batch_size: int
        the number of BOLD signals to simulate
    node_size: int
        the number of ROIs
    sc: float node_size x node_size array
        structural connectivity
    fit_gains: bool
        flag for fitting gains 1: fit 0: not fit
    g, g_EE, gIE, gEI: tensor with gradient on
        model parameters to be fit
    w_bb: tensor with node_size x node_size (grad on depends on fit_gains)
        connection gains
    std_in std_out: tensor with gradient on
        std for state noise and output noise
    g_m g_v sup_ca sup_cb sup_cc: tensor with gradient on
        hyper parameters for prior distribution of g gIE and gEI
    Methods
    -------
    forward(input, noise_out, hx)
        forward model (WWD) for generating a number of BOLD signals with current model parameters
    """
    state_names = ['E', 'I', 'x', 'f', 'v', 'q']
    model_name = "WWD"
    fit_lfm_flat = False

    def __init__(self, node_size: int,
                 batch_size: int, step_size: float, repeat_size: float, tr: float, sc: float, fit_gains_flat: bool,
                 param: ParamsJR) -> None:
        """
        Parameters
        ----------
        state_size : int
        the number of states in the WWD model
        input_size : int
            the number of states with noise as input
        tr : float
            tr of fMRI image
        step_size: float
            Integration step for forward model
        hidden_size: int
            the number of step_size in a tr
        batch_size: int
            the number of BOLD signals to simulate
        node_size: int
            the number of ROIs
        sc: float node_size x node_size array
            structural connectivity
        fit_gains: bool
            flag for fitting gains 1: fit 0: not fit
        g_mean_ini: float, optional
            prior mean of g (default 100)
        g_std_ini: float, optional
            prior std of g (default 2.5)
        gEE_mean_ini: float, optional
            prior mean of gEE (default 2.5)
        gEE_std_ini: float, optional
            prior std of gEE (default 0.5)
        """
        super(RNNWWD, self).__init__()
        self.state_size = 6  # 6 states WWD model
        #self.input_size = input_size  # 1 or 2
        self.tr = tr  # tr fMRI image
        self.step_size = step_size  # integration step 0.05
        self.hidden_size = int(tr /step_size)
        self.batch_size = batch_size  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.repeat_size = repeat_size
        self.sc = sc  # matrix node_size x node_size structure connectivity
        #self.dist = torch.tensor(dist, dtype=torch.float32)
        self.fit_gains_flat = fit_gains_flat  # flag for fitting gains

        self.param = param

        self.output_size = node_size  # number of EEG channels

        # set states E I f v mean and 1/sqrt(variance)
        self.E_m = Parameter(torch.tensor(0.16, dtype=torch.float32))
        self.I_m = Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.f_m = Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.v_m = Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.E_v = Parameter(torch.tensor(20, dtype=torch.float32))
        self.I_v = Parameter(torch.tensor(20, dtype=torch.float32))
        self.f_v = Parameter(torch.tensor(10, dtype=torch.float32))
        self.v_v = Parameter(torch.tensor(10, dtype=torch.float32))

        # set w_bb as Parameter if fit_gain is True
        if self.fit_gains_flat == True:
            self.w_bb = Parameter(torch.tensor(np.zeros((node_size, node_size)) + 0.05,
                                               dtype=torch.float32))  # connenction gain to modify empirical sc
        else:
            self.w_bb = torch.tensor(np.zeros((node_size, node_size)), dtype=torch.float32)

        vars = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars:
            if np.any(getattr(param, var)[1] > 0):
                setattr(self, var, Parameter(
                    torch.tensor(getattr(param, var)[0] + 1 / getattr(param, var)[1] * np.random.randn(1, )[0],
                                 dtype=torch.float32)))
                if var != 'std_in':
                    dict_nv = {}
                    dict_nv['m'] = getattr(param, var)[0]
                    dict_nv['v'] = getattr(param, var)[1]

                    dict_np = {}
                    dict_np['m'] = var + '_m'
                    dict_np['v'] = var + '_v'

                    for key in dict_nv:
                        setattr(self, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
            else:
                setattr(self, var, torch.tensor(getattr(param, var)[0], dtype=torch.float32))

    """def check_input(self, input: Tensor) -> None:
        expected_input_dim = 2 
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))
        if self.batch_size != input.size(0):
            raise RuntimeError(
                'input.size(0) must be equal to batch_size. Expected {}, got {}'.format(
                    self.batch_size, input.size(0)))"""

    def forward(self, external,  hx, hE):
        """
        Forward step in simulating the BOLD signal.
        Parameters
        ----------
        input: tensor with node_size x hidden_size x batch_size x input_size
            noise for states
        noise_out: tensor with node_size x batch_size
            noise for BOLD
        hx: tensor with node_size x state_size
            states of WWD model
        Outputs
        -------
        next_state: dictionary with keys:
        'current_state''bold_batch''E_batch''I_batch''x_batch''f_batch''v_batch''q_batch'
            record new states and BOLD
        """
        next_state = {}

        # hx is current state (6) 0: E 1:I (neural activitiveties) 2:x 3:f 4:v 5:f (BOLD)

        
        x = hx[:, 2:3]
        f = hx[:, 3:4]
        v = hx[:, 4:5]
        q = hx[:, 5:6]

        dt = torch.tensor(self.step_size, dtype=torch.float32)
        con_1 = torch.ones_like(dt)
        # Generate the ReLU module for model parameters gEE gEI and gIE
        m = torch.nn.ReLU()

        # Update the Laplacian based on the updated connection gains w_bb.
        if self.sc.shape[0] > 1:

            # Update the Laplacian based on the updated connection gains w_bb.
            w = torch.exp(self.w_bb) * torch.tensor(self.sc, dtype=torch.float32)
            w_n = torch.log1p(0.5 * (w + torch.transpose(w, 0, 1))) / torch.linalg.norm(
                torch.log1p(0.5 * (w + torch.transpose(w, 0, 1))))
            self.sc_m = w_n
            l_s = -torch.diag(torch.sum(w_n, axis=1)) + w_n
        else:
            l_s = torch.tensor(np.zeros((1, 1)), dtype=torch.float32)

        
        # placeholder for the updated corrent state
        current_state = torch.zeros_like(hx)

        # placeholders for output BOLD, history of E I x f v and q
        # placeholders for output BOLD, history of E I x f v and q
        bold_batch = torch.zeros((self.node_size,self.batch_size))
        E_batch = torch.zeros((self.node_size,self.batch_size))
        I_batch = torch.zeros((self.node_size,self.batch_size))
        
        x_batch = torch.zeros((self.node_size,self.batch_size))
        f_batch = torch.zeros((self.node_size,self.batch_size))
        v_batch = torch.zeros((self.node_size,self.batch_size))
        q_batch = torch.zeros((self.node_size,self.batch_size))

        E_bat_hid=torch.zeros((self.node_size,self.batch_size, self.hidden_size))
        E_mn = hx[:,0:1]
        I_mn = hx[:,1:2]
        #print(E_m.shape)
        # Use the forward model to get neural activity at ith element in the batch. 
        for i_batch in range(self.batch_size):
            
            
            #print(E.shape)
            
            
            # Since tr is about second we need to use a small step size like 0.05 to integrate the model states. 
            for i_hidden in range(self.hidden_size):
                E = torch.zeros((self.node_size, self.repeat_size))
                I = torch.zeros((self.node_size, self.repeat_size))
                for i_rep in range(self.repeat_size):
                    E[:,i_rep] = E_mn[:,0]+0.02*torch.randn(self.node_size)#hx[i_batch-1,:,0]
                    I[:,i_rep] = I_mn[:,0]+0.001*torch.randn(self.node_size)##hx[i_batch-1,:,1]"""
               
                    
                # Calculate the input recurrents. 
                IE = m(self.W_E*self.I_0 + (0.001*con_1 + m(self.g_EE))*E \
                    + self.g*torch.matmul(l_s, E) - (0.001*con_1 + m(self.g_IE))*I) # input currents for E
                II = m(self.W_I*self.I_0 + (0.001*con_1 + m(self.g_EI))*E -I) # input currents for I 
                
                # Calculate the firing rates. 
                rE = h_tf(self.aE, self.bE, self.dE, IE) # firing rate for E
                rI = h_tf(self.aI, self.bI, self.dI, II) # firing rate for I 
                # Update the states by step-size 0.05. 
                ddE = E + dt*(-E*torch.reciprocal(self.tau_E) +self.gamma_E*(1.-E)*rE) \
                      + torch.sqrt(dt)*(0.02*con_1 + m(self.std_in))*torch.randn((self.node_size, self.repeat_size))### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)
                ddI = I + dt*(-I*torch.reciprocal(self.tau_I) +self.gamma_I*rI) \
                      + torch.sqrt(dt)*torch.randn(self.node_size, self.repeat_size) * (0.02*con_1 + m(self.std_in))
                    
                    
                    
                # Calculate the saturation for model states (for stability and gradient calculation). 
                E = torch.tanh(0.00001+m(ddE))
                I = torch.tanh(0.00001+m(ddI))
                    
                
                I_mn = (I.mean(1)[:,np.newaxis])
                E_mn = (E.mean(1)[:,np.newaxis])
                E_bat_hid[:,i_batch,i_hidden]=E_mn[:,0]
                
            
            """hx[i_batch,:,0] = E_m
            hx[i_batch,:,1] = I_m"""
            E_batch[:,i_batch]=E_mn[:,0]
            I_batch[:,i_batch]=I_mn[:,0]
        
        
        for i_batch in range(self.batch_size):
            
            
            for i_hidden in range(self.hidden_size):
                dx = x + 1*dt*(E_bat_hid[:,i_batch,i_hidden][:,np.newaxis] - torch.reciprocal(self.tau_s) * x - torch.reciprocal(self.tau_f)* (f - con_1))
                df = f + 1*dt*x
                dv = v + 1*dt*(f - torch.pow(v, torch.reciprocal(self.alpha))) * torch.reciprocal(self.tau_0)
                dq = q + 1*dt*(f * (con_1 - torch.pow(con_1 - self.rho, torch.reciprocal(f)))*torch.reciprocal(self.rho) \
                            - q * torch.pow(v, torch.reciprocal(self.alpha)) *torch.reciprocal(v)) \
                              * torch.reciprocal(self.tau_0)
                    
                    
                x = dx#torch.tanh(dx)
                f = df#(con_1 + torch.tanh(df - con_1))
                v = dv#(con_1 + torch.tanh(dv - con_1))
                q = dq#(con_1 + torch.tanh(dq - con_1))    
            # Put x f v q from each tr to the placeholders for checking them visually.
            x_batch[:,i_batch]=x[:,0]
            f_batch[:,i_batch]=f[:,0]
            v_batch[:,i_batch]=v[:,0]
            q_batch[:,i_batch]=x[:,0]
            # Put the BOLD signal each tr to the placeholder being used in the cost calculation.
            #print(q_batch[-1])
            bold_batch[:,i_batch]=((0.001*con_1 + m(self.std_out))*torch.randn(self.node_size,1)+ \
                                100.0*self.V*torch.reciprocal(self.E0)*(self.k1 * (con_1 - q) \
                                + self.k2 * (con_1 - q *torch.reciprocal(v)) + self.k3 * (con_1 - v)) )[:,0]
        
        # Update the current state. 
        #print(E_m.shape)
        current_state = torch.cat([E_mn, I_mn, x, f, v, q], axis = 1)
        next_state['current_state'] = current_state
        next_state['bold_batch'] = bold_batch
        next_state['E_batch'] = E_batch
        next_state['I_batch'] = I_batch
        next_state['x_batch'] = x_batch
        next_state['f_batch'] = f_batch
        next_state['v_batch'] = v_batch
        next_state['q_batch'] = q_batch

        return next_state, hE
        
def h_tf_np(a, b, d, z):
    """
    Neuronal input-output functions of excitatory pools and inhibitory pools.  
            
    Take the variables a, x, and b and convert them to a linear equation (a*x - b) while adding a small 
    amount of noise 0.00001 while dividing that term to an exponential of the linear equation multiplied by the 
    d constant for the appropriate dimensions.  
    """
    num = 0.00001 + np.abs(a * z - b)
    den = 0.00001 * d + np.abs(1.0000 - np.exp(-d * (a * z - b)))
    return num/den
class WWD_np( ):
    """
    A module for forward model (WWD) to simulate a batch of BOLD signals
    
    Attibutes
    ---------
    state_size : int
        the number of states in the WWD model
    input_size : int
        the number of states with noise as input
    tr : float
        tr of fMRI image
    step_size: float
        Integration step for forward model
    hidden_size: int
        the number of step_size in a tr 
    batch_size: int
        the number of BOLD signals to simulate
    node_size: int
        the number of ROIs
    sc: float node_size x node_size array   
        structural connectivity
    fit_gains: bool
        flag for fitting gains 1: fit 0: not fit
    g, g_EE, gIE, gEI: tensor with gradient on
        model parameters to be fit
    w_bb: tensor with node_size x node_size (grad on depends on fit_gains)
        connection gains
    std_in std_out: tensor with gradient on
        std for state noise and output noise
    g_m g_v sup_ca sup_cb sup_cc: tensor with gradient on
        hyper parameters for prior distribution of g gIE and gEI
    Methods
    -------
    forward(input, noise_out, hx)
        forward model (WWD) for generating a number of BOLD signals with current model parameters
    """
    def __init__(self, node_size: int, batch_size: int, step_size: float, tr: float, sc: float, param: ParamsJR) -> None:
        """
        Parameters
        ----------
        state_size : int
        the number of states in the WWD model
        input_size : int
            the number of states with noise as input
        tr : float
            tr of fMRI image
        step_size: float
            Integration step for forward model
        hidden_size: int
            the number of step_size in a tr 
        batch_size: int
            the number of BOLD signals to simulate
        node_size: int
            the number of ROIs
        sc: float node_size x node_size array   
            structural connectivity
        fit_gains: bool
            flag for fitting gains 1: fit 0: not fit
        g_mean_ini: float, optional
            prior mean of g (default 100)
        g_std_ini: float, optional
            prior std of g (default 2.5)
        gEE_mean_ini: float, optional
            prior mean of gEE (default 2.5)
        gEE_std_ini: float, optional
            prior std of gEE (default 0.5)
        """
        super(WWD_np, self).__init__()
        
        
        self.step_size = step_size# integration step 0.05
        
        self.node_size = node_size # num of ROI    
        self.hidden_size =  int(tr/step_size)
        self.batch_size = batch_size
        self.sc = sc # matrix node_size x node_size structure connectivity
         
        vars = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars:
            setattr(self, var, getattr(param, var)[0])
    
        
    def forward(self, hx, u, u_out):
        """
        Forward step in simulating the BOLD signal. 
        Parameters
        ----------
        input: tensor with node_size x hidden_size x batch_size x input_size
            noise for states
        noise_out: tensor with node_size x batch_size
            noise for BOLD
        hx: tensor with node_size x state_size
            states of WWD model
        Outputs
        -------
        next_state: dictionary with keys:
        'current_state''bold_batch''E_batch''I_batch''x_batch''f_batch''v_batch''q_batch'
            record new states and BOLD
        """
        next_state = {}
        dt = self.step_size
        
        
        l_s = -np.diag(self.sc.sum(1)) + self.sc 
        
        
        
        
        E = hx[:,0:1]
        I = hx[:,1:2]
        x = hx[:,2:3]
        f = hx[:,3:4]
        v = hx[:,4:5]
        q = hx[:,5:6] 
        E_batch = np.zeros((self.node_size, self.batch_size))
        I_batch = np.zeros((self.node_size, self.batch_size))
        bold_batch = np.zeros((self.node_size, self.batch_size))
        x_batch = np.zeros((self.node_size, self.batch_size))
        v_batch = np.zeros((self.node_size, self.batch_size))
        f_batch = np.zeros((self.node_size, self.batch_size))
        q_batch = np.zeros((self.node_size, self.batch_size))

        E_bat_hidd=np.zeros((self.node_size, self.batch_size, self.hidden_size))
        # Use the forward model to get neural activity at ith element in the batch. 
        for i_batch in range(self.batch_size):
            
            
            #print(E.shape)
            
            
            # Since tr is about second we need to use a small step size like 0.05 to integrate the model states. 
            for i_hidden in range(self.hidden_size):
                

                noise_E= u[:,i_batch, i_hidden, 0][:,np.newaxis] 
                noise_I= u[:,i_batch, i_hidden, 1][:,np.newaxis]             
                              
                IE = self.W_E*self.I_0 + max([0,self.g_EE])*E \
                    + self.g*l_s.dot(E) - max([0,self.g_IE])*I # input currents for E
                II = self.W_I*self.I_0 + max([0,self.g_EI])*E -I # input currents for I 
                IE[IE<0]=0
                II[II<0]=0
                # Calculate the firing rates. 
                rE = h_tf_np(self.aE, self.bE, self.dE, IE) # firing rate for E
                rI = h_tf_np(self.aI, self.bI, self.dI, II) # firing rate for I 
                # Update the states by step-size 0.05. 
                
                ddE = E + dt*(-E/self.tau_E +self.gamma_E*(1.-E)*rE) \
                      + np.sqrt(dt)*(0.02 + max([0,self.std_in]))*noise_E### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)
                ddI = I + dt*(-I/self.tau_I +self.gamma_I*rI) \
                      + np.sqrt(dt)*noise_I * (0.02 + max([0, self.std_in]))
                ddE[ddE<0] = 0 
                ddI[ddI<0] = 0   
                E = np.tanh(0.00001+ddE)
                I = np.tanh(0.00001+ddI)   
                E_bat_hidd[:,i_batch,i_hidden] = E[:,0]   
               
            E_batch[:,i_batch] = E[:,0]
            I_batch[:,i_batch] = I[:,0]
        
        magic=int(50/dt)
        
        for i_batch in range(self.batch_size):
            
            noise_out= u_out[:,i_batch][:,np.newaxis]
            for i_hidden in range(int(self.hidden_size/magic)):
                
                dx = x + 0.05*((E_bat_hidd[:,i_batch, i_hidden*magic:(1+i_hidden)*magic]).mean(1)[:,np.newaxis] - x/self.tau_s- (f - 1)/self.tau_f)
                df = f + 0.05*x
                dv = v + 0.05*(f - np.power(v, 1/self.alpha)) /self.tau_0
                dq = q + 0.05*(f * (1 - np.power(1 - self.rho, 1/f))/self.rho \
                            - q * np.power(v, 1/self.alpha) /v)/self.tau_0
                    
                    
                x = np.tanh(dx)
                f = (1 + np.tanh(df - 1))
                v = (1 + np.tanh(dv - 1))
                q = (1 + np.tanh(dq - 1))    
            # Put x f v q from each tr to the placeholders for checking them visually.
            x_batch[:,i_batch]=x[:,0]
            f_batch[:,i_batch]=f[:,0]
            v_batch[:,i_batch]=v[:,0]
            q_batch[:,i_batch]=x[:,0]
            # Put the BOLD signal each tr to the placeholder being used in the cost calculation.
            #print(q_batch[-1])
            bold_batch[:,i_batch]=(0.001 + max([0, self.std_out])*noise_out+ \
                                100.0*self.V/self.E0*(self.k1 * (1 - q) \
                                + self.k2 * (1 - q/v) + self.k3 * (1 - v)) )[:,0]
        
        # Update the current state. 
        #print(E_m.shape)
        current_state = np.concatenate([E, I, x, f, v, q], axis = 1)
        next_state['current_state'] = current_state
        next_state['bold_batch'] = bold_batch
        next_state['E_batch'] = E_batch
        next_state['I_batch'] = I_batch
        next_state['x_batch'] = x_batch
        next_state['f_batch'] = f_batch
        next_state['v_batch'] = v_batch
        next_state['q_batch'] = q_batch

        #return next_state # JG_MOD
        return next_state, hE
        
    def update_param(self, param_new):
        vars = [a for a in dir(param_new) if not a.startswith('__') and not callable(getattr(param_new, a))]
        for var in vars:
            setattr(self, var, getattr(param_new, var)[0])
        
