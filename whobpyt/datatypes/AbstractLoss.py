import torch

class AbstractLoss:
    # This is the abstract class for objective function components, or for a custom objective function with multiple components. 

    def __init__(self):
    
        self.simKey = None #This is a string key to extract from the dictionary of simulation outputs the time series used by the objective function
    
    def loss(self, sim, emp, simKey, model: torch.nn.Module, state_vals):
        # Calculates a loss to be backpropagated through
        # TODO: In some classes this function is called calcLoss, need to make consistent
        pass
