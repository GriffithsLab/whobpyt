import torch

class AbstractLoss:

    def __init__(self):
        super(AbstractLoss, self).__init__()
        self.simKey = None #This is a string key to extract from the dictionary of simulation outputs the time series used by the objective function
    
    def loss(self, sim, emp, simKey, model: torch.nn.Module, state_vals):
        pass
