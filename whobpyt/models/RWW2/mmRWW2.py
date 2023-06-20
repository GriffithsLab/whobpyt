## Multi-modal Reduce Wong Wang Neural Mass model with BOLD and EEG

import torch
from whobpyt.datatypes import AbstractNMM, AbstractMode, AbstractParams
from whobpyt.models.RWW2 import RWW2, ParamsRWW2
from whobpyt.models.BOLD import BOLD_Layer, BOLD_Params
from whobpyt.models.EEG import EEG_Layer, EEG_Params

class mmRWW2(RWW2):

    model_name = "mmRWW2"
    
    def __init__(self, num_regions, num_channels, paramsNode, paramsEEG, paramsBOLD, Con_Mtx, dist_mtx, step_size, sim_len):

        self.eeg = EEG_Layer(num_regions, paramsEEG, num_channels)
        self.bold = BOLD_Layer(num_regions, paramsBOLD)
        super(mmRWW2, self).__init__(num_regions, paramsNode, Con_Mtx, dist_mtx, step_size, useBC = False)
        
        self.node_size = num_regions
        self.step_size = step_size
        self.sim_len = sim_len
        
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
    
    def forward(self, external, hx, hE):
        return forward(self, external, hx, hE)

def setModelParameters(self):
    super(mmRWW2, self).setModelParameters() # Currently this is the only one with parameters being fitted
    self.eeg.setModelParameters()
    self.bold.setModelParameters()    

def createIC(self, ver):
    #super(mmRWW2, self).createIC()
    #self.eeg.createIC()
    #self.bold.createIC()
    
    self.next_start_state = 0.2 * torch.rand((self.node_size, 6)) + torch.tensor([0, 0, 0, 1.0, 1.0, 1.0])
    
    return 0.2 * torch.rand((self.node_size, 6)) + torch.tensor([0, 0, 0, 1.0, 1.0, 1.0])
    

def forward(self, external, hx, hE):
    
    NMM_vals, hE = super(mmRWW2, self).forward(external, self.next_start_state[:, 0:2], hE) #TODO: Fix the hx in the future
    EEG_vals, hE = self.eeg.forward(self.step_size, self.sim_len, NMM_vals["E"].T)
    BOLD_vals, hE = self.bold.forward(self.next_start_state[:, 2:6], self.step_size, self.sim_len, NMM_vals["E"].T)
    
    self.next_start_state = torch.cat((NMM_vals["NMM_state"], BOLD_vals["BOLD_state"]), dim=1).detach()
    
    sim_vals = {**NMM_vals, **EEG_vals, **BOLD_vals}
    sim_vals['current_state'] = torch.tensor(1.0) #Dummy variable
    
    return sim_vals, hE