## Multi-modal Reduce Wong Wang Neural Mass model with BOLD and EEG

import torch
from whobpyt.datatypes import AbstractNMM, AbstractMode, AbstractParams
from whobpyt.models.RWWEI2 import RWWEI2, ParamsRWWEI2
from whobpyt.models.BOLD import BOLD_Layer, BOLD_Params
from whobpyt.models.EEG import EEG_Layer, EEG_Params

class RWWEI2_EEG_BOLD(RWWEI2):

    model_name = "RWWEI2_EEG_BOLD"
    
    def __init__(self, num_regions, num_channels, paramsNode, paramsEEG, paramsBOLD, Con_Mtx, dist_mtx, step_size, sim_len, device = torch.device('cpu')):

        self.eeg = EEG_Layer(num_regions, paramsEEG, num_channels, device = device)
        self.bold = BOLD_Layer(num_regions, paramsBOLD, device = device)
        super(RWWEI2_EEG_BOLD, self).__init__(num_regions, paramsNode, Con_Mtx, dist_mtx, step_size, useBC = False, device = device)
        
        self.node_size = num_regions
        self.step_size = step_size
        self.sim_len = sim_len
        
        self.device = device
        
        self.batch_size = 1
        
        self.tr = sim_len
        self.steps_per_TR = 1
        self.TRs_per_window = 1
        
        self.state_names = ['E', 'I', 'x', 'f', 'v', 'q']
        self.output_names = ["bold", "eeg"]
        self.track_params = []  #Is populated during setModelParameters()
        
        self.setModelParameters()

    def info(self):
        return {"state_names": ['E', 'I', 'x', 'f', 'v', 'q'], "output_names": ["bold", "eeg"]} #TODO: Update to take multiple output names
            
    def setModelParameters(self):
        return setModelParameters(self)
        
    def createIC(self, ver):
        return createIC(self, ver)
        
    def setBlocks(self, num_blocks):
        self.num_blocks = num_blocks
        self.eeg.num_blocks = num_blocks
        self.bold.num_blocks = num_blocks
    
    def forward(self, external, hx, hE, setNoise = None):
        return forward(self, external, hx, hE, setNoise)

def setModelParameters(self):
    super(RWWEI2_EEG_BOLD, self).setModelParameters() # Currently this is the only one with parameters being fitted
    self.eeg.setModelParameters()
    self.bold.setModelParameters()    

def createIC(self, ver):
    #super(RWWEI2_EEG_BOLD, self).createIC()
    #self.eeg.createIC()
    #self.bold.createIC()
    
    self.next_start_state = 0.2 * torch.rand((self.node_size, 6, self.batch_size)) + torch.tensor([[0], [0], [0], [1.0], [1.0], [1.0]]).repeat(self.node_size, 1, self.batch_size)
    self.next_start_state = self.next_start_state.to(self.device)
    
    return self.next_start_state
    

def forward(self, external, hx, hE, setNoise):
    
    NMM_vals, hE = super(RWWEI2_EEG_BOLD, self).forward(external, self.next_start_state[:, 0:2, :], hE, setNoise) #TODO: Fix the hx in the future
    EEG_vals, hE = self.eeg.forward(self.step_size, self.sim_len, NMM_vals["E"].permute((1,0,2)))
    BOLD_vals, hE = self.bold.forward(self.next_start_state[:, 2:6, :], self.step_size, self.sim_len, NMM_vals["E"].permute((1,0,2)))
    
    self.next_start_state = torch.cat((NMM_vals["NMM_state"], BOLD_vals["BOLD_state"]), dim=1).detach()
    
    sim_vals = {**NMM_vals, **EEG_vals, **BOLD_vals}
    sim_vals['current_state'] = torch.tensor(1.0, device = self.device) #Dummy variable
    
    # Reshape if Blocking is being Used
    if self.num_blocks > 1:
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
