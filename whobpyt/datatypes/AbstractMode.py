import torch

class AbstractMode:

    def __init__(self):
        super(AbstractMode, self).__init__()
         
        self.state_names = ["None"]
        self.output_names = ["None"]
        self.track_params = []
               
    def info(self):
        return {"state_names": self.state_names, 
                "output_names": self.output_names, 
                "track_params": self.track_params}
            
    def setModelParameters(self):
        pass
        
    def createIC(self, ver):
        pass
    
    def forward(self, external, hx, hE):
        pass