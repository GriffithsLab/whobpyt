import torch

class AbstractNMM(torch.nn.Module):

    def __init__(self):
        super(AbstractNMM, self).__init__()
        
    def info(self):
        return {"state_names": None, "output_name": None}
            
    def setModelParameters(self):
        pass
        
    def createIC(self, ver):
        pass
    
    def forward(self, external, hx, hE):
        pass