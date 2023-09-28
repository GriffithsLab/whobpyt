import torch
from whobpyt.datatypes.parameter import par

class AbstractParams:
    # This class stores the parameters used by a model. The parameters may be for the Neural Mass Model and/or Neuroimaging Modality.
    # It should be useable by both the pytorch model for training and a numpy model for parameter verification. 

    def __init__(self, **kwargs):
        # Define the parameters using the par data structure
        pass
    
    def getFittedNames(self):
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
        # Moves all parameters between CPU and GPU
        
        vars_names = [a for a in dir(self) if not a.startswith('__')]
        for var_name in vars_names:
            var = getattr(self, var_name)
            if (type(var) == par):
                var.to(device)