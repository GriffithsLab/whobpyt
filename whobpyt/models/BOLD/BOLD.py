import torch
from whobpyt.datatypes import AbstractMode
   
class BOLD_Layer(AbstractMode):
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
        self.useBC = useBC   #useBC: is if we want the model to use boundary conditions
        
        self.params = params
    
    def info(self):
        return {"state_names": ['x', 'f', 'v', 'q'], "output_name": "bold"}
            
    def setModelParameters(self):
        return setModelParameters(self)
        
    def createIC(self, ver):
        #Starting Condition
        #x = 1   # vasodilatory signal
        #f = 1   # inflow
        #v = 1   # blood volumne
        #q = 1   # deoxyhemoglobin content 
        pass
    
    def forward(self, init_state, step_size, sim_len, node_history, useGPU = False):
        return forward(self, init_state, step_size, sim_len, node_history, useGPU = False)    

def setModelParameters(self):
    #############################################
    ## BOLD Constants
    #############################################
    
    #Friston 2003 - Table 1 - Priors on biophysical parameters
    self.kappa = self.params.kappa # Rate of signal decay (1/s)
    self.gammaB = self.params.gammaB # Rate of flow-dependent elimination (1/s)
    self.tao = self.params.tao # Hemodynamic transit time (s)
    self.alpha = self.params.alpha # Grubb's exponent
    self.ro = self.params.ro #Resting oxygen extraction fraction
    
    self.V_0 = self.params.V_0
    self.k_1 = self.params.k_1
    self.k_2 = self.params.k_2
    self.k_3 = self.params.k_3
    
def forward(self, init_state, step_size, sim_len, node_history, useGPU = False):
    
    hE = torch.tensor(1.0) #Dummy variable
    
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
    
    sim_vals = {}
    sim_vals["BOLD_state"] = state_vals
    sim_vals["x_window"] = layer_hist[:,:,0]
    sim_vals["f_window"] = layer_hist[:,:,1]
    sim_vals["v_window"] = layer_hist[:,:,2]
    sim_vals["q_window"] = layer_hist[:,:,3]
    sim_vals["bold_window"] = torch.transpose(layer_hist[:,:,4], 0, 1) #TODO: Make dimensions consistent later
    
    return sim_vals, hE

