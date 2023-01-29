import torch

class AbstractNMM(torch.nn.Module):

    def __init__(self):
        super(AbstractNMM, self).__init__()
            
    def setModelParameters(self):
        pass
    
    def forward(self, external, hx, hE):
        pass