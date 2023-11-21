## NumPy Validation Version of Multi-modal Reduce Wong Wang Neural Mass model with BOLD and EEG

import numpy as np
from whobpyt.datatypes import AbstractNMM, AbstractMode, AbstractParams
from whobpyt.models.RWWEI2 import RWWEI2_np, ParamsRWWEI2
from whobpyt.models.BOLD import BOLD_np, BOLD_Params
from whobpyt.models.EEG import EEG_np, EEG_Params

class RWWEI2_EEG_BOLD_np(RWWEI2_np):

    model_name = "RWWEI2_EEG_BOLD_np"
    
    def __init__(self, num_regions, num_channels, paramsNode, paramsEEG, paramsBOLD, Con_Mtx, dist_mtx, step_size, sim_len):
        super(RWWEI2_EEG_BOLD_np, self).__init__(num_regions, paramsNode, Con_Mtx, dist_mtx, step_size)
        self.eeg = EEG_np(num_regions, paramsEEG, num_channels)
        self.bold = BOLD_np(num_regions, paramsBOLD)
        
        self.node_size = num_regions
        self.step_size = step_size
        self.sim_len = sim_len
        
        self.tr = sim_len
        self.steps_per_TR = 1
        self.TRs_per_window = 1
        
        self.state_names = ['E', 'I', 'x', 'f', 'v', 'q']
        self.output_names = ["E"] #TODO: This should be made consistent with info()

    def info(self):
        return {"state_names": ['E', 'I', 'x', 'f', 'v', 'q'], "output_names": ["bold", "eeg"]} #TODO: Update to take multiple output names
            
    def setModelParameters(self):
        return setModelParameters(self)
        
    def createIC(self, ver):
        return createIC(self, ver)
    
    def forward(self, external, hx, hE):
        return forward(self, external, hx, hE)

def setModelParameters(self):
    super(RWWEI2_EEG_BOLD, self).setModelParameters() # Currently this is the only one with parameters being fitted
    self.eeg.setModelParameters()
    self.bold.setModelParameters()    

def createIC(self, ver):
    #super(RWWEI2_EEG_BOLD, self).createIC()
    #self.eeg.createIC()
    #self.bold.createIC()
    
    self.next_start_state = 0.2 * np.random.rand(self.node_size, 6) + np.array([0, 0, 0, 1.0, 1.0, 1.0])
    
    return 0.2 * np.random.rand(self.node_size, 6) + np.array([0, 0, 0, 1.0, 1.0, 1.0])
    

def forward(self, external, hx, hE):
    
    NMM_vals, hE = super(RWWEI2_EEG_BOLD_np, self).forward(external, self.next_start_state[:, 0:2], hE) #TODO: Fix the hx in the future
    EEG_vals, hE = self.eeg.forward(self.step_size, self.sim_len, NMM_vals["E"])
    BOLD_vals, hE = self.bold.forward(self.next_start_state[:, 2:6], self.step_size, self.sim_len, NMM_vals["E"])
    
    self.next_start_state = np.concatenate((NMM_vals["NMM_state"], BOLD_vals["BOLD_state"]), axis=1)
    
    sim_vals = {**NMM_vals, **EEG_vals, **BOLD_vals}
    sim_vals['current_state'] = np.array(1.0) #Dummy variable
    
    return sim_vals, hE