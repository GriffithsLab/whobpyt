import torch
from whobpyt.datatypes import AbstractNMM, AbstractParams, par
from math import sqrt

class RWWEI2(AbstractNMM):
    '''
    Reduced Wong Wang Excitatory Inhibatory (RWWEXcInh) Model - Version 2
    
    EQUATIONS & BIOLOGICAL VARIABLES FROM:
    
    Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitation-inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
    
    Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
    
    Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
    
    Attributes
    -------------
    params : ParamsRWWEI2
        An AbstractParams object which contains the model's parameters
    step_size : Float
        The step size of numerical integration (in msec)
    num_regions : Int
        The number of brain regions used in the model
    node_size : Int
        The number of brain regions used in the model   
    Con_Mtx : Tensor [ node_size x node_size ]
        Matrix with connection strengths between each node
    Dist_Mtx : Tensor [ node_size x node_size ]
        Matrix with distances between each node (used for delays)
    output_size : Int
        The number of brain regions used in the model
    state_names : List of Strings
        A list of the state varaible names of the model
    output_names : List of Strings
        A list of the output variable names of the model
    track_params :
        A list of which parameter to track over training epochs
    num_blocks : Int
        The number of blocks (used by FNGFPG Paradigm)
    max_delay : 
        Related to Delays (delay code to be verified)
    buffer_len :
        Related to Delays (delay code to be verified)  
    delayed_S_E : 
         Related to Delays (delay code to be verified)  
    buffer_idx :
         Related to Delays (delay code to be verified)       
    mu :
        Related to Delays (delay code to be verified)
    use_fit_gains : Bool
        Whether to fit the gains of connection weights (Not Implemented)
    use_fit_lfm : Bool
        Whether to fit the Lead Field Matrix (Lead Field Matrix not Used)
    useBC : Bool
        Whether to add boundary dynamics to state variables
    device : torch.device
        Whether the model is run on CPU or GPU
    '''
    
    def __init__(self, num_regions, params, Con_Mtx, Dist_Mtx, step_size = 0.1, sim_len = 1000, useBC = False, device = torch.device('cpu')):  
        '''
        '''
        super(RWWEI2, self).__init__() # To inherit parameters attribute
        
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
        self.sim_len = sim_len
        
        self.num_regions = num_regions
        self.node_size = num_regions #Variable used by modelfitting
        self.Con_Mtx = Con_Mtx
        self.Dist_Mtx = Dist_Mtx
        
        self.num_blocks = 1
        
        self.device = device
        
        self.max_delay = 100 #msec #This should be greater than what is possible of max(Dist_Mtx)/velocity
        self.buffer_len = int(self.max_delay/self.step_size)
        self.delayed_S_E = torch.zeros(self.buffer_len, num_regions).to(self.device)
        
        self.buffer_idx = 0
        self.mu = torch.tensor([0]).to(self.device) #Connection Speed addition
        
        self.params = params
        
        self.use_fit_gains = False  # flag for fitting gains
        self.use_fit_lfm = False
        self.useBC = False
        
        self.output_size = num_regions
        
        self.state_names = ['E', 'I']
        self.output_names = ['E']
        self.track_params = []  #Is populated during setModelParameters()
        
        self.setModelParameters()


    def info(self):
        return {"state_names": ['E', 'I'], "output_names": ['E']}
            
    def setModelParameters(self):
        return setModelParameters(self)
        
    def createIC(self, ver):
        self.next_start_state = torch.tensor(0.1) + 0.2 * torch.rand((self.node_size, 2, self.num_blocks))
        self.next_start_state = self.next_start_state.to(self.device)
        
        return self.next_start_state
        
    def setBlocks(self, num_blocks):
        self.num_blocks = num_blocks
    
    def genNoise(self, block_len, batched = False):
        '''
        This generates noise to be used by the model. It is particulary useful for the FNGFPG design 
        where the same noise must be used be restructure for two different forward passes. 
        
        Parameters
        ------------
        
        '''
        num_blocks = int(self.sim_len/block_len) #TODO: this is also a model attribute, but is changed when running serial vs. blocked during FNGFPG
        if batched == True:
            num_blocks = self.num_blocks
        
        ## Noise for the epoch (which has 1 batch)
        v_of_T_block = torch.normal(0,1,size = (len(torch.arange(0, block_len, self.step_size)), self.node_size, 2, num_blocks), device = self.device) 
        blockNoise = {}
        blockNoise['E'] = v_of_T_block[:,:,0,:]
        blockNoise['I'] = v_of_T_block[:,:,1,:]
        
        v_of_T_serial = serializeTS(v_of_T_block, self.node_size, 2)
        serialNoise = {}
        serialNoise['E'] = torch.unsqueeze(v_of_T_serial[:,:,0],2)
        serialNoise['I'] = torch.unsqueeze(v_of_T_serial[:,:,1],2)
        
        return [serialNoise, blockNoise]
    
    def forward(self, external, hx, hE, setNoise = None, batched = False):
        '''
        '''
        return forward(self, external, hx, hE, setNoise, batched)

def setModelParameters(self):

    # NOTE: Parameters stored in self.params
 
    # To make the parameters work with modelfitting.py
    param_reg = [torch.nn.Parameter(torch.tensor(1., device = self.device))]
    param_hyper = [torch.nn.Parameter(torch.tensor(1., device = self.device))]
    vars_names = [a for a in dir(self.params) if (type(getattr(self.params, a)) == par)]
    for var_name in vars_names:
        var = getattr(self.params, var_name)
        if (var.fit_par):
            param_reg.append(var.val)
            self.track_params.append(var_name)
    self.params_fitted = {'modelparameter': param_reg, 'hyperparameter': param_hyper}
    


def forward(self, external, hx, hE, setNoise, batched):

    # Some documentation to be written...

    # Defining NMM Parameters to simplify later equations
    G = self.params.G.value()
    Lambda = self.params.Lambda.value()       # 1 or 0 depending on using long range feed forward inhibition (FFI)
    a_E = self.params.a_E.value()             # nC^(-1)
    b_E = self.params.b_E.value()             # Hz
    d_E = self.params.d_E.value()             # s
    tau_E = self.params.tau_E.value()         # ms
    tau_NMDA = self.params.tau_NMDA.value()   # ms
    W_E = self.params.W_E.value()
    a_I = self.params.a_I.value()             # nC^(-1)
    b_I = self.params.b_I.value()             # Hz
    d_I = self.params.d_I.value()             # s
    tau_I = self.params.tau_I.value()         # ms
    tau_GABA = self.params.tau_GABA.value()   # ms
    W_I = self.params.W_I.value()
    w_plus = self.params.w_plus.value()       # Local excitatory recurrence
    J_NMDA = self.params.J_NMDA.value()       # Excitatory synaptic coupling in nA
    J = torch.unsqueeze(self.params.J.value(),1) # NOTE: Extra dimension allows for local tuning if applicable # Local feedback inhibitory synaptic coupling. 1 in no-FIC case, different in FIC case #TODO: Currently set to J_NMDA but should calculate based on paper
    gamma = self.params.gamma.value()         # a kinetic parameter in ms
    sig = self.params.sig.value()             # 0.01 # Noise amplitude at node in nA
    I_0 = self.params.I_0.value()             # The overall effective external input in nA
    I_external = self.params.I_external.value() #External input current 
    gammaI = self.params.gammaI.value()
    J_new = self.params.J_new.value()
    
    const51=torch.tensor([51], device = self.device)

    def H_for_E_V3(I_E, update = False):
        
        numer = torch.abs(a_E*I_E - b_E) + 1e-9*1
        denom = torch.where((-d_E*(a_E*I_E - b_E) > 50), 
                            torch.abs(1 - 1e9*(-d_E*(a_E*I_E - b_E))) + 1e-9*d_E,
                            torch.abs(1 - torch.exp(torch.min(-d_E*(a_E*I_E - b_E), const51))) + 1e-9*d_E)
        r_E = numer / denom
        
        return r_E
        
    def H_for_I_V3(I_I, update = False):
        
        numer = torch.abs(a_I*I_I - b_I) + 1e-9*1
        denom = torch.where((-d_I*(a_I*I_I - b_I) > 50),
                            torch.abs(1 - 1e5*(-d_I*(a_I*I_I - b_I))) + 1e-9*d_I,
                            torch.abs(1 - torch.exp(torch.min(-d_I*(a_I*I_I - b_I), const51))) + 1e-9*d_I)
        r_I = numer / denom
        
        return r_I


    init_state = hx
    sim_len = self.sim_len
    useDelays = False
    useLaplacian = False
    withOptVars = False
    debug = False

            
    # Runs the RWW Model 
    #
    # INPUT
    #  init_state: Tensor [regions, state_vars] # Regions is number of nodes and should match self.num_regions. There are 2 state variables. 
    #  sim_len: Int - The length of time to simulate in msec
    #  withOptVars: Boolean - Whether to include the Current and Firing rate variables of excitatory and inhibitory populations in layer_history
    #
    # OUTPUT
    #  state_vars:  Tensor - [regions, state_vars]
    #  layer_history: Tensor - [time_steps, regions, state_vars (+ opt_params)]
    #
    
    if batched:
        num_steps = int((sim_len/self.step_size))
    else:
        num_steps = int((sim_len/self.step_size)/self.num_blocks)
    
    if setNoise is not None:
        Ev = setNoise["E"]
        Iv = setNoise["I"]
        state_hist = torch.zeros(num_steps, self.num_regions, 2, self.num_blocks, device = self.device) #TODO: Deal with the dimensions
        if(withOptVars):
            opt_hist = torch.zeros(num_steps, self.num_regions, 4, self.num_blocks, device = self.device)
    else:
        Ev = torch.normal(0,1,size = (len(torch.arange(0, sim_len, self.step_size)), self.num_regions, self.num_blocks)).to(self.device)
        Iv = torch.normal(0,1,size = (len(torch.arange(0, sim_len, self.step_size)), self.num_regions, self.num_blocks)).to(self.device)
        state_hist = torch.zeros(int(sim_len/self.step_size), self.num_regions, 2, self.num_blocks).to(self.device)
        if(withOptVars):
            opt_hist = torch.zeros(int(sim_len/self.step_size), self.num_regions, 4, self.num_blocks).to(self.device)
    
    ones_row = torch.ones(1, self.num_blocks, device = self.device)
    
    # RWW and State Values
    S_E = init_state[:, 0, :]
    S_I = init_state[:, 1, :]

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
        I_E = W_E*I_0 + w_plus*J_NMDA*S_E + G*J_NMDA*Network_S_E - torch.matmul(J,ones_row)*S_I + I_external # J * S_I updated to allow for local J fitting in parallel for FNGFPG
        I_I = W_I*I_0 + J_NMDA*S_E - J_new*S_I + Lambda*G*J_NMDA*Network_S_E
        
        # Firing Rates
        # Orig
        #r_E = (self.a_E*I_E - self.b_E) / (1 - torch.exp(-self.d_E*(self.a_E*I_E - self.b_E)))
        #r_I = (self.a_I*I_I - self.b_I) / (1 - torch.exp(-self.d_I*(self.a_I*I_I - self.b_I)))
        #
        # EDIT5: Version to address torch.exp() returning nan and prevent gradient returning 0, and works with autograd
        r_E = H_for_E_V3(I_E)
        r_I = H_for_I_V3(I_I)
        
        # Average Synaptic Gating Variable
        dS_E = - S_E/tau_E + (1 - S_E)*gamma*r_E #+ self.sig*v_of_T[i, :] Noise now added later
        dS_I = - S_I/tau_I + gammaI*r_I #+ self.sig*v_of_T[i, :] Noise now added later
            
        # UPDATE VALUES
        S_E = S_E + self.step_size*dS_E + sqrt(self.step_size)*sig*Ev[i, :, :]
        S_I = S_I + self.step_size*dS_I + sqrt(self.step_size)*sig*Iv[i, :, :]
               
        # Bound the possible values of state variables (From fit.py code for numerical stability)
        if(self.useBC):
            S_E = torch.tanh(0.00001 + torch.nn.functional.relu(S_E - 0.00001))
            S_I = torch.tanh(0.00001 + torch.nn.functional.relu(S_I - 0.00001))
        
        state_hist[i, :, 0, :] = S_E
        state_hist[i, :, 1, :] = S_I

        if useDelays:
            self.delayed_S_E = self.delayed_S_E.clone(); self.delayed_S_E[self.buffer_idx, :] = S_E #TODO: This means that not back-propagating the network just the individual nodes

            if (self.buffer_idx == (self.buffer_len - 1)):
                self.buffer_idx = 0
            else: 
                self.buffer_idx = self.buffer_idx + 1
        
        if(withOptVars):
            opt_hist[i, :, 0, :] = I_I
            opt_hist[i, :, 1, :] = I_E
            opt_hist[i, :, 2, :] = r_I
            opt_hist[i, :, 3, :] = r_E
        
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
    sim_vals["E"] = layer_hist[:,:,0,:].permute((1,0,2))
    sim_vals["I"] = layer_hist[:,:,1,:].permute((1,0,2))
    
    return sim_vals, hE

    
    
def blockTS(data, blocks, numNodes, numSV):
    # data: time x nodes x state_variables
    # return: time x nodes x state_variables x blocks
    
    n = torch.numel(data)
    
    if (not (n%blocks == 0)):
        print("ERROR: data is not divisable by blocks")
        return 
    
    newTimeDim = int(n/(blocks*numNodes*numSV))
    
    data_p = data.permute((2,1,0)) # state_vars x nodes x time
    data_r = torch.reshape(data_p, (numSV, numNodes, blocks, newTimeDim))
    data_p2 = data_r.permute((3, 1, 0, 2))
    
    return data_p2
    
    
def serializeTS(data, numNodes, numSV):
    # data: time x nodes x state_variables x blocks
    # return: time x nodes x state_variables

    n = torch.numel(data)
    newTimeDim = int(n/(numNodes*numSV))
    
    data_p = data.permute((2,1,3,0))
    data_r = torch.reshape(data_p, (numSV, numNodes, newTimeDim))
    data_p2 = data_r.permute((2,1,0))
    
    return data_p2