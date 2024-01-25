import torch
from whobpyt.datatypes import AbstractNMM

class RNNEEG(AbstractNMM):
    '''
    
    Lead Field Matrix multiplication which converts Source Space EEG to Channel Space EEG
    
    '''

    def __init__(self, params, node_size = 200, output_size = 64): 
        '''
        '''
        super(RNNEEG, self).__init__(params) # To inherit parameters attribute
                
        # Initialize the EEG Model 
        #
        # INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: EEG_Params - This contains the EEG Parameters, to maintain a consistent paradigm
        
        self.node_size = node_size
        self.output_size = output_size
        self.output_names = ['eeg']
        
        
        self.setModelParameters()
    
    
            
    
        
    def forward(self, neuroAct: torch.tensor):
       
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
        lm = self.params.lm.value()
        lm_t = (lm.T / torch.sqrt(lm ** 2).sum(1)).T
        output_eeg =  torch.matmul(lm_t, neuroAct) # time x node x batch -> node x time x batch
    
        return output_eeg  


    

    
    