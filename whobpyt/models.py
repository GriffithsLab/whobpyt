"""
WhoBPyt Model Classes
"""

import torch

class RWW_Params():
    ## EQUATIONS & BIOLOGICAL VARIABLES FROM:
    #
    # Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitation–inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
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
    # Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitation–inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
    # Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
    # Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
    # Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.  
  
    def __init__(self, num_regions, params, Con_Mtx, useBC = False):        
        super(RWW_Layer, self).__init__() # To inherit parameters attribute
        
        # Initialize the RWW Model 
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
    
    def forward(self, init_state, step_size, sim_len, withOptVars = False, useGPU = False, debug = False):
                
        # Runs the RWW Model 
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
        
        if(useGPU):
            v_of_T = torch.normal(0,1,size = (len(torch.arange(0, sim_len, step_size)), self.num_regions)).cuda()
            state_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 2).cuda()
            if(withOptVars):
                opt_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 4).cuda()
        else:
            v_of_T = torch.normal(0,1,size = (len(torch.arange(0, sim_len, step_size)), self.num_regions))
            state_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 2)
            if(withOptVars):
                opt_hist = torch.zeros(int(sim_len/step_size), self.num_regions, 4)
        
        # RWW and State Values
        S_E = init_state[:, 0]
        S_I = init_state[:, 1]

        num_steps = int(sim_len/step_size)
        for i in range(num_steps):
            # Currents
            I_E = self.W_E*self.I_0 + self.w_plus*self.J_NMDA*S_E + self.G*self.J_NMDA*torch.matmul(self.Con_Mtx, S_E) - self.J*S_I + self.I_external
            I_I = self.W_I*self.I_0 + self.J_NMDA*S_E - S_I + self.Lambda*self.G*self.J_NMDA*torch.matmul(self.Con_Mtx, S_E)
            
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
            S_E = S_E + step_size*dS_E
            S_I = S_I + step_size*dS_I
            
            # Bound the possible values of state variables (From fit.py code for numerical stability)
            if(self.useBC):
                S_E = torch.tanh(0.00001 + torch.nn.functional.relu(S_E - 0.00001))
                S_I = torch.tanh(0.00001 + torch.nn.functional.relu(S_I - 0.00001))
            
            state_hist[i, :, 0] = S_E
            state_hist[i, :, 1] = S_I 
            
            if(withOptVars):
                opt_hist[i, :, 0] = I_I
                opt_hist[i, :, 1] = I_E
                opt_hist[i, :, 2] = r_I
                opt_hist[i, :, 3] = r_E
            
        state_vals = torch.cat((torch.unsqueeze(S_E, 1), torch.unsqueeze(S_I, 1)), 1)
        
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
