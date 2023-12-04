import torch
from whobpyt.datatypes import AbstractMode

class EEG_Layer(AbstractMode):
    '''
    
    Lead Field Matrix multiplication which converts Source Space EEG to Channel Space EEG
    
    '''

    def __init__(self, num_regions, params, num_channels, device = torch.device('cpu')): 
        '''
        '''
        super(EEG_Layer, self).__init__() # To inherit parameters attribute
                
        # Initialize the EEG Model 
        #
        # INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: EEG_Params - This contains the EEG Parameters, to maintain a consistent paradigm
        
        self.num_regions = num_regions
        self.num_channels = num_channels
        
        self.device = device
        
        self.num_blocks = 1
        
        self.params = params
        
        self.setModelParameters()
    
    def info(self):
        return {"state_names": ["None"], "output_name": "eeg"}
            
    def setModelParameters(self):
        return setModelParameters(self)
        
    def createIC(self, ver):
        pass
    
    def forward(self, step_size, sim_len, node_history, device = torch.device('cpu')):
        return forward(self, step_size, sim_len, node_history)   

def setModelParameters(self):
    #############################################
    ## EEG Lead Field
    #############################################
    
    self.LF = self.params.LF.to(self.device)
    
def forward(self, step_size, sim_len, node_history):
    
    hE = torch.tensor(1.0).to(self.device) #Dummy variable
    
    # Runs the EEG Model
    #
    # INPUT
    #  step_size: Float - The step size in msec which must match node_history step size.
    #  sim_len: Int - The amount of EEG to simulate in msec, and should match time simulated in node_history. 
    #  node_history: Tensor - [time_points, regions, num_blocks] # This would be input coming from NMM
    #  device: torch.device - Whether to run on GPU or CPU - default is CPU and GPU has not been tested for Network_NMM code
    #
    # OUTPUT
    #  layer_history: Tensor - [time_steps, regions, num_blocks]
    #
    
    layer_hist = torch.zeros(int((sim_len/step_size)/self.num_blocks), self.num_channels, self.num_blocks).to(self.device)

    num_steps = int((sim_len/step_size)/self.num_blocks)
    for i in range(num_steps):
        layer_hist[i, :, :] = torch.matmul(self.LF, node_history[i, :, :]) # TODO: Check dimensions and if correct transpose of LF
    
    sim_vals = {}
    sim_vals["eeg"] = layer_hist.permute((1,0,2)) # time x node x batch -> node x time x batch

    return sim_vals, hE