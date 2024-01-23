## Multi-modal Reduce Wong Wang Neural Mass model with BOLD and EEG

import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes import par, AbstractNMM, AbstractParams
from whobpyt.models.RWWNEURO import RNNRWWNEU
from whobpyt.models.RWWNEURO import ParamsRWWNEU
from whobpyt.models.BOLD import RNNBOLD, ParamsBOLD
from whobpyt.models.EEG import RNNEEG, ParamsEEG
import numpy as np
class RWW_EEG_BOLD(torch.nn.Module):

    model_name = "RWW_EEG_BOLD"
    
    def __init__(self, paramsRWWNEU, paramsEEG, paramsBOLD, node_size = 68, output_size = 64, TRs_per_window = 20, \
        step_size = 0.05, tr=1.0, tr_eeg= 0.001, sc=np.ones((68,68)), use_fit_gains= True):
        super(RWW_EEG_BOLD, self).__init__() # To inherit parameters attribute
        self.eeg = RNNEEG(paramsEEG, node_size = node_size, output_size = output_size)
        self.bold = RNNBOLD(paramsBOLD, TRs_per_window = TRs_per_window, node_size = node_size, step_size = step_size,  tr=tr)
        self.NMM = RNNRWWNEU(paramsRWWNEU, node_size = node_size, TRs_per_window = TRs_per_window, step_size = step_size,  \
                   tr=tr, sc=sc, use_fit_gains= use_fit_gains)
        self.params = AbstractParams()
        self.state_size = 6
        self.node_size = node_size
        self.step_size = step_size
        self.output_size = output_size
        self.pop_size =1
        
        self.TRs_per_window = TRs_per_window
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        
        self.tr = tr
        self.tr_eeg = tr_eeg
        self.steps_per_TR = int(tr/ step_size)
        
        self.steps_per_TR_eeg = int(tr_eeg/ step_size)
        
        self.sc_fitted = self.NMM.sc_fitted
        
        self.state_names = np.concatenate([self.NMM.state_names, self.bold.state_names])
        self.output_names = self.bold.output_names+self.eeg.output_names
        self.track_params = self.NMM.track_params + self.eeg.track_params + self.bold.track_params
        self.params_fitted ={}
        self.params_fitted['modelparameter']=[]
        self.params_fitted['hyperparameter']=[]
        
        self.setParamsAsattr(paramsRWWNEU, paramsEEG, paramsBOLD)
        self.set_params_fitted()
        
    def set_params_fitted(self):
        for key in self.NMM.params_fitted:
            self.params_fitted[key].extend(self.NMM.params_fitted[key])
            self.params_fitted[key].extend(self.eeg.params_fitted[key])
            self.params_fitted[key].extend(self.bold.params_fitted[key])
        
    def info(self):
        # Information about the model, which may be used by other classes to know which variables to use. 
        
        return {"state_names": self.state_names, 
                "output_names": self.output_names,
                "track_params": self.track_params}    
        
    def setParamsAsattr(self,paramsRWWNEU, paramsEEG, paramsBOLD):
        # Returns a named list of paramters that are being fitted
        # Assumes the par datastructure is being used for parameters
        
        for var in paramsRWWNEU.params:
            setattr(self.params, var, getattr(paramsRWWNEU,var))
        
        for var in paramsEEG.params:
            setattr(self.params, var, getattr(paramsEEG,var))
        
        for var in paramsBOLD.params:
            setattr(self.params, var, getattr(paramsBOLD,var))
    
            
    
        
    def createIC(self, ver):
        """
        
            A function to return an initial state tensor for the model.    
        
        Parameters
        ----------
        
        ver: int
            Ignored Parameter
        
        
        Returns
        ----------
        
        Tensor
            Random Initial Conditions for RWW & BOLD 
        
        
        """
        
        # initial state
        return torch.tensor(0.2 * np.random.uniform(0, 1, (self.node_size, self.pop_size, self.state_size)) + np.array(
                [0, 0, 0, 1.0, 1.0, 1.0]), dtype=torch.float32)

    
    def createDelayIC(self, ver):
        """
        Creates the initial conditions for the delays.

        Parameters
        ----------
        ver : int
            Initial condition version. (in the JR model, the version is not used. It is just for consistency with other models)

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, delays_max) with random values between `state_lb` and `state_ub`.
        """

        delays_max = 500
        state_ub = 0.5
        state_lb = 0.1

        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size,  delays_max)), dtype=torch.float32)
    
    
    def forward(self, external, hx, hE, setNoise=None):
        
        NMM_vals, hE = self.NMM.forward(external, hx[:,:,:2], hE) #TODO: Fix the hx in the future
        #print(NMM_vals["states"].shape)
        EEG_vals = self.eeg.forward((NMM_vals["states"][:,0,0,:,::self.steps_per_TR_eeg]).reshape((self.node_size,-1)))
        BOLD_vals = self.bold.forward(NMM_vals["states"][:,:,0], hx[:,:,2:])
        
        
        next_state = {}
        next_state['current_state'] = torch.cat([NMM_vals['current_state'], BOLD_vals['current_state']], dim=2)
        next_state['eeg'] = EEG_vals
        next_state['bold'] = BOLD_vals['bold']
        next_state['states'] = NMM_vals['states']
        #print(EEG_vals.shape, BOLD_vals['bold'].shape)
        return next_state, hE
    
    
    
    
