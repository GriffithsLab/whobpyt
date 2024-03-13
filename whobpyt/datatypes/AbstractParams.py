import torch
from whobpyt.datatypes.parameter import par

class AbstractParams:
    '''
    This class stores the parameters used by a model. The parameters may be for the Neural Mass Model and/or Neuroimaging Modality. 
    It should be useable by both the pytorch model for training and a numpy model for parameter verification. 
    '''
    def __init__(self, **kwargs):
        '''
        Initialize the AbstractParams object and define the parameters using the par data structure
        
        Args:
            **kwargs: Keyword arguments for the model parameters.

        Returns:
            None
        '''
        pass
    
    def getFittedNames(self):
        '''
        Function to obtain the named list of parameters being fitted

        Returns:
            Named list of parameters that are being fitted
        '''
        # Returns a named list of paramters that are being fitted
        # Assumes the par datastructure is being used for parameters
        
        fp = []
        vars_names = [a for a in dir(self) if not a.startswith('__')]
        for var_name in vars_names:
            var = getattr(model.param, var_name)
            if (type(var) == whobpyt.datatypes.parameter.par):
                if (var.fit_par == True):
                    fp.append(var_name)
                if (var.fit_hyper == True):
                    fp.append(var_name + "_m")
                    fp.append(var_name + "_v_inv")
        return fp
        
    def to(self, device):
        '''
        Moves all parameters between CPU and GPU
        
        Args:
            device: torch.device object defining wether CPU or GPU

        Returns:
            None
        '''
        # Moves all parameters between CPU and GPU
        
        vars_names = [a for a in dir(self) if not a.startswith('__')]
        for var_name in vars_names:
            var = getattr(self, var_name)
            if (type(var) == par):
                var.to(device)
