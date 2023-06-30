import numpy as np

import numpy

class EEG_np():
    def __init__(self, num_regions, params, num_channels):        
                
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
    
        hE = np.array(1.0) #Dummy variable
        
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
        
        layer_hist = numpy.zeros((int(sim_len/step_size), self.num_channels, 1))
        
        num_steps = int(sim_len/step_size)
        for i in range(num_steps):
            layer_hist[i, :, 0] = numpy.matmul(self.LF, node_history[i,:]) #numpy.matmul(self.LF, node_history[i, :, 0] - node_history[i, :, 1])
            
        sim_vals = {}
        sim_vals["eeg"] = layer_hist[:,:,0]

            
        return sim_vals, hE #layer_hist