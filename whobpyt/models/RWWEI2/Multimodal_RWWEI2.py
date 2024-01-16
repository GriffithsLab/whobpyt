## Multi-modal Reduce Wong Wang Neural Mass model with BOLD and EEG

import torch
from whobpyt.datatypes import AbstractNMM, AbstractMode, AbstractParams
from whobpyt.models.RWWEI2 import RWWEI2, ParamsRWWEI2
from whobpyt.models.BOLD import BOLD_Layer, BOLD_Params
from whobpyt.models.EEG import EEG_Layer, EEG_Params

class RWWEI2_EEG_BOLD(torch.nn.Module):

    model_name = "RWWEI2_EEG_BOLD"
    
    def __init__(self, num_regions, num_channels, paramsNode, paramsEEG, paramsBOLD, Con_Mtx, dist_mtx, step_size, sim_len, device = torch.device('cpu')):
        super(RWWEI2_EEG_BOLD, self).__init__() # To inherit parameters attribute
        self.eeg = EEG_Layer(num_regions, paramsEEG, num_channels, device = device)
        self.bold = BOLD_Layer(num_regions, paramsBOLD, device = device)
        self.NMM = RWWEI2(num_regions, paramsNode, Con_Mtx, dist_mtx, step_size, sim_len=sim_len, useBC = False, device = device)
        self.params = paramsNode
        self.node_size = num_regions
        self.step_size = step_size
        self.sim_len = sim_len
        self.output_size = num_regions
        self.device = device
        
        self.batch_size = 1
        self.use_fit_gains = self.NMM.use_fit_gains  # flag for fitting gains
        self.use_fit_lfm = self.NMM.use_fit_lfm
        self.useBC =self.NMM.useBC
        self.tr = sim_len
        self.steps_per_TR = 1
        self.TRs_per_window = 1
        
        self.state_names = ['E', 'I', 'x', 'f', 'v', 'q']
        self.output_names = ["bold", "eeg"]
        self.track_params = []  #Is populated during setModelParameters()
        self.params_fitted={}
        self.params_fitted['modelparameter'] =[]
        self.params_fitted['hyperparameter'] =[]
        
        self.NMM.setModelParameters()
        
        self.eeg.setModelParameters()
        self.bold.setModelParameters()
        
        self.params_fitted['modelparameter'] = self.NMM.params_fitted['modelparameter'] + self.eeg.params_fitted['modelparameter'] +self.bold.params_fitted['modelparameter']
        self.params_fitted['hyperparameter'] = self.NMM.params_fitted['hyperparameter'] + self.eeg.params_fitted['hyperparameter'] +self.bold.params_fitted['hyperparameter']
        self.track_params = self.NMM.track_params + self.eeg.track_params +self.bold.track_params
    def info(self):
        return {"state_names": ['E', 'I', 'x', 'f', 'v', 'q'], "output_names": ["bold", "eeg"]} #TODO: Update to take multiple output names
            
    
        
    def createIC(self, ver):
        self.NMM.next_start_state = 0.2 * torch.rand((self.node_size, 6, self.batch_size)) \
        + torch.tensor([[0], [0], [0], [1.0], [1.0], [1.0]]).repeat(self.node_size, 1, self.batch_size)
        self.NMM.next_start_state = self.NMM.next_start_state.to(self.device)
    
        return self.NMM.next_start_state
        
    def createDelayIC(self, ver):
        # Creates a time series of state variables to represent their past values as needed when delays are used. 
        
        return torch.tensor(1.0) #Dummy variable if delays are not used
    
        
    def setBlocks(self, num_blocks):
        self.num_blocks = num_blocks
        self.eeg.num_blocks = num_blocks
        self.bold.num_blocks = num_blocks
    
    def forward(self, external, hx, hE, setNoise=None):
        
        NMM_vals, hE = self.NMM.forward(external, self.NMM.next_start_state[:, 0:2, :], hE, setNoise) #TODO: Fix the hx in the future
        print(NMM_vals['E'].shape)
        EEG_vals, hE = self.eeg.forward(self.step_size, self.sim_len, NMM_vals["E"].permute((1,0,2)))
        BOLD_vals, hE = self.bold.forward(self.NMM.next_start_state[:, 2:6, :], self.step_size, self.sim_len, NMM_vals["E"].permute((1,0,2)))
        
        self.NMM.next_start_state = torch.cat((NMM_vals["NMM_state"], BOLD_vals["BOLD_state"]), dim=1).detach()
        
        sim_vals = {**NMM_vals, **EEG_vals, **BOLD_vals}
        sim_vals['current_state'] = torch.tensor(1.0, device = self.device) #Dummy variable
        
        # Reshape if Blocking is being Used
        if self.NMM.num_blocks > 1:
            for simKey in set(self.state_names + self.output_names):
                #print(sim_vals[simKey].shape) # Nodes x Time x Blocks
                #print(torch.unsqueeze(sim_vals[simKey].permute((1,0,2)),2).shape) #Time x Nodes x SV x Blocks
                sim_vals[simKey] = serializeTS(torch.unsqueeze(sim_vals[simKey].permute((1,0,2)),2), sim_vals[simKey].shape[0], 1).permute((1,0,2))[:,:,0]
        else:
            for simKey in set(self.state_names + self.output_names):
                sim_vals[simKey] = sim_vals[simKey][:,:,0]
                
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
