import torch
from whobpyt.datatypes.parameter import par

class AbstractParams:
    # This class stores the parameters used by a model. The parameters may be for the Neural Mass Model and/or Neuroimaging Modality.
    # It should be useable by both the pytorch model for training and a numpy model for parameter verification. 
    params ={}
    def __init__(self, **kwargs):
        # Define the parameters using the par data structure
        
        for var in kwargs:
            self.params[var] = kwargs[var]
    def setParamsAsattr(self):
        # Returns a named list of paramters that are being fitted
        # Assumes the par datastructure is being used for parameters
        
        for var in self.params:
            setattr(self, var, self.params[var])