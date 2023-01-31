import torch

class AbstractLoss(torch.nn.Module):

    def __init__(self):
        super(AbstractLoss, self).__init__()
    
    def loss(self, sim, emp, model: torch.nn.Module, state_vals):
        pass
