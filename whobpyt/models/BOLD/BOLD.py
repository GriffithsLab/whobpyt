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
        
        self.setModelParameters()
    
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
    pass
    
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
    
    # Defining parameters to simplify later equations
    kappa = self.params.kappa.value()    # Rate of signal decay (1/s)
    gammaB = self.params.gammaB.value()  # Rate of flow-dependent elimination (1/s)
    tao = self.params.tao.value()        # Hemodynamic transit time (s)
    alpha = self.params.alpha.value()    # Grubb's exponent
    ro = self.params.ro.value()          #Resting oxygen extraction fraction
    V_0 = self.params.V_0.value()
    k_1 = self.params.k_1.value()
    k_2 = self.params.k_2.value()
    k_3 = self.params.k_3.value()
    
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
        
        # Bound the possible values of state variables (From fit.py code for numerical stability)
        if(self.useBC):
            x = torch.tanh(x)
            f = (1 + torch.tanh(f - 1))
            v = (1 + torch.tanh(v - 1))
            q = (1 + torch.tanh(q - 1))
        
        #BOLD Calculation
        BOLD = V_0*(k_1*(1 - q) + k_2*(1 - q/v) + k_3*(1 - v))
        
        layer_hist[i, :, 0] = x
        layer_hist[i, :, 1] = f
        layer_hist[i, :, 2] = v
        layer_hist[i, :, 3] = q
        layer_hist[i, :, 4] = BOLD
        
    state_vals = torch.cat((torch.unsqueeze(x, 1), torch.unsqueeze(f, 1), torch.unsqueeze(v, 1), torch.unsqueeze(q, 1)),1)
    
    sim_vals = {}
    sim_vals["BOLD_state"] = state_vals
    sim_vals["x"] = layer_hist[:,:,0].T
    sim_vals["f"] = layer_hist[:,:,1].T
    sim_vals["v"] = layer_hist[:,:,2].T
    sim_vals["q"] = layer_hist[:,:,3].T
    sim_vals["bold"] = layer_hist[:,:,4].T
    
    return sim_vals, hE

