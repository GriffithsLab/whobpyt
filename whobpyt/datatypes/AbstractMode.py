import torch

class AbstractMode:
    # Neuroimaging Modalities such as EEG and fMRI BOLD may be seperate from the NMM model
    # and implemented by inheriting from this class.
    # Going forward, the recommendation is to have the modalities integrated with the model.

    def __init__(self):
        # Define the information of the mode so that other class methods know which variables to use.
        
        self.state_names = ["None"]
        self.output_names = ["None"]
        self.track_params = []
        
    def info(self):
        # Information about the model, which may be used by other classes to know which variables to use. 
        
        return {"state_names": self.state_names, 
                "output_names": self.output_names, 
                "track_params": self.track_params}
            
    def setModelParameters(self):
        # Setting the parameters that will be optimized as either model parameters or 2ndLevel/hyper
        # parameters (for various optional features). 
        # This should be called in the __init__() function implementation for convenience if possible.
        pass
        
    def createIC(self, ver):
        # Create the initial conditions for the state variables, if applicable.
        pass
    
    def forward(self, external, hx, hE):
        # Run the mode simulation.
        pass