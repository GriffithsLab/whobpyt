import torch

class AbstractLoss:
    # This is the abstract class for objective function components, or for a custom objective function with multiple components. 

    def __init__(self, simKey = None, device = torch.device('cpu')):
    
        self.simKey = simKey #This is a string key to extract from the dictionary of simulation outputs the time series used by the objective function
        device = device
    
    def loss(self, simData, empData):
        # Calculates a loss to be backpropagated through
        # If the objective function needs additional info, it should be defined at initialization so that the parameter fitting paradigms don't need to change
        
        # simData: is a dictionary of simulated state variable/neuroimaging modality time series. Typically accessed as simData[self.simKey].
        # empData: is the target either as a time series or a calculated phenomena metric
        
        pass
