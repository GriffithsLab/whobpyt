import torch

class AbstractNMM(torch.nn.Module):

    def __init__(self):
        super(AbstractNMM, self).__init__()
        
        self.state_names = ["None"]
        self.output_names = ["None"]
        self.track_params = [] #Tracking of NMM Parameters over training
        
        self.use_fit_gains = False  
        self.use_fit_lfm = False
        
    def info(self):
        return {"state_names": self.state_names, 
                "output_names": self.output_names,
                "track_params": self.track_params}
            
    def setModelParameters(self):
        pass
        
    def createIC(self, ver):
        pass
        
    def createDelayIC(self, ver):
        return torch.tensor(1.0) #Dummy variable if delays are not used
    
    def forward(self, external, hx, hE):
        # Returns a dictionary with values of state variables during the simulation
        pass