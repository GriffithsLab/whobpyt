import torch

class AbstractMode(torch.nn.Module):

    def __init__(self):
        super(AbstractMode, self).__init__()
        
    def info(self):
        return {"state_names": None, "output_name": None}
            
    def setModelParameters(self):
        pass
        
    def createIC(self, ver):
        pass
    
    def forward(self, external, hx, hE):
        pass