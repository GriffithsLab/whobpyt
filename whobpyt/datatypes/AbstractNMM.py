import torch

class AbstractNMM(torch.nn.Module):
    # This is the abstract class for the models (typically neural mass models) that are being trained. 
    # The neuroimaging modality might be integrated into the model as well. 

    def __init__(self):
        super(AbstractNMM, self).__init__() # May not want to enherit from torch.nn.Module in the future 
        
        self.state_names = ["None"] # The names of the state variables of the model
        self.output_names = ["None"] # The variable to be used as output from the NMM, for purposes such as the input to an objective function
        self.track_params = [] # Which NMM Parameters to track over training
        
        self.use_fit_gains = False # Whether to fit the Connectivity Matrix
        self.use_fit_lfm = False # Whether to fit the Lead Field Matrix
        
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
        # Create the initial conditions for the model state variables. 
        pass
        
    def createDelayIC(self, ver):
        # Creates a time series of state variables to represent their past values as needed when delays are used. 
        
        return torch.tensor(1.0) #Dummy variable if delays are not used
    
    def forward(self, external, hx, hE):
        # Run the mode simulation.
        # Returns a dictionary with values of state variables during the simulation.
        pass