import torch
from whobpyt.datatypes.AbstractParams import AbstractParams
from whobpyt.datatypes.AbstractNMM import AbstractNMM

class RWW2(AbstractNMM):
    ## EQUATIONS & BIOLOGICAL VARIABLES FROM:
    #
    # Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitationâ€“inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
    # Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
    # Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
    # Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.  
  
    def __init__(self, num_regions, params, Con_Mtx, Dist_Mtx, step_size = 0.0001, useBC = False):        
        super(RWW2, self).__init__() # To inherit parameters attribute
        
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
        self.node_size = num_regions #Variable used by modelfitting
        self.Con_Mtx = Con_Mtx
        self.Dist_Mtx = Dist_Mtx
        
        self.max_delay = 100 #msec #This should be greater than what is possible of max(Dist_Mtx)/velocity
        self.buffer_len = int(self.max_delay/self.step_size)
        self.delayed_S_E = torch.zeros(self.buffer_len, num_regions)
        
        self.buffer_idx = 0
        self.mu = torch.tensor([0]) #Connection Speed addition
        
        self.param = params
        
        self.use_fit_gains = False  # flag for fitting gains
        self.use_fit_lfm = False
        self.useBC = False
        
        self.output_size = num_regions

    def info(self):
        return {"state_names": None, "output_name": None}
            
    def setModelParameters(self):
        return setModelParameters(self)
        
    def createIC(self, ver):
        pass
    
    def forward(self, external, hx, hE):
        return forward(self, external, hx, hE)

def setModelParameters(self):
    #############################################
    ## RWW Constants
    #############################################
    
    #Zeroing the components which deal with a connected network
    self.G = self.param.G
    self.Lambda = self.param.Lambda #1 or 0 depending on using long range feed forward inhibition (FFI)
    
    #Excitatory Gating Variables
    self.a_E = self.param.a_E               # nC^(-1)
    self.b_E = self.param.b_E               # Hz
    self.d_E = self.param.d_E              # s
    self.tau_E = self.param.tau_E   # ms
    self.tau_NMDA = self.param.tau_NMDA  # ms
    self.W_E = self.param.W_E
    
    #Inhibitory Gating Variables
    self.a_I = self.param.a_I               # nC^(-1)
    self.b_I = self.param.b_I               # Hz
    self.d_I = self.param.d_I             # s
    self.tau_I = self.param.tau_I    # ms
    self.tau_GABA = self.param.tau_GABA   # ms
    self.W_I = self.param.W_I
    
    #Setting other variables
    self.w_plus = self.param.w_plus # Local excitatory recurrence
    self.J_NMDA = self.param.J_NMDA # Excitatory synaptic coupling in nA
    self.J = self.param.J # Local feedback inhibitory synaptic coupling. 1 in no-FIC case, different in FIC case #TODO: Currently set to J_NMDA but should calculate based on paper
    self.gamma = self.param.gamma #a kinetic parameter in ms
    self.sig = self.param.sig #0.01 # Noise amplitude at node in nA
    #self.v_of_T = None #param.v_of_T # Uncorrelated standarg Gaussian noise # NOTE: Now set at time of running forward model
    self.I_0 = self.param.I_0 # The overall effective external input in nA
    
    self.I_external = self.param.I_external #External input current 
    

    #############################################
    ## Model Additions/modifications
    #############################################
    
    self.gammaI = self.param.gammaI
    
    # To make the parameters work with modelfitting.py
    param_reg = [torch.nn.Parameter(torch.tensor(1.))]
    param_hyper = [torch.nn.Parameter(torch.tensor(1.))]
    vars_name = [a for a in dir(self.param) if not a.startswith('__') and not callable(getattr(self.param, a))]
    for var in vars_name:
        if (type(getattr(self, var).val) == torch.nn.parameter.Parameter):
            param_reg.append(getattr(self, var).val)
    self.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}
        

def forward(self, external, hx, hE):

    def H_for_E_V3(I_E, update = False):
        
        numer = torch.abs(self.a_E*I_E - self.b_E) + 1e-9*1
        denom = torch.where((-self.d_E*(self.a_E*I_E - self.b_E) > 50), 
                            torch.abs(1 - 1e9*(-self.d_E*(self.a_E*I_E - self.b_E))) + 1e-9*self.d_E,
                            torch.abs(1 - torch.exp(torch.min(-self.d_E*(self.a_E*I_E - self.b_E), torch.tensor([51])))) + 1e-9*self.d_E)
        r_E = numer / denom
        
        return r_E
        
    def H_for_I_V3(I_I, update = False):
        
        numer = torch.abs(self.a_I*I_I - self.b_I) + 1e-9*1
        denom = torch.where((-self.d_I*(self.a_I*I_I - self.b_I) > 50),
                            torch.abs(1 - 1e5*(-self.d_I*(self.a_I*I_I - self.b_I))) + 1e-9*self.d_I,
                            torch.abs(1 - torch.exp(torch.min(-self.d_I*(self.a_I*I_I - self.b_I), torch.tensor([51])))) + 1e-9*self.d_I)
        r_I = numer / denom
        
        return r_I


    init_state = hx
    sim_len = self.sim_len
    useDelays = False
    useLaplacian = False
    withOptVars = False
    useGPU = False
    debug = False

            
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
        r_E = H_for_E_V3(I_E)
        r_I = H_for_I_V3(I_I)
        
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
    
    sim_vals = {}
    sim_vals["NMM_state"] = state_vals
    sim_vals["E_window"] = layer_hist[:,:,0]
    sim_vals["I_window"] = layer_hist[:,:,1]
    
    return sim_vals, hE
        