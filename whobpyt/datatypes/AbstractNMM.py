"""
Authors: Andrew Clappison, John Griffiths, Zheng Wang, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita
"""

import torch
from whobpyt.datatypes.parameter import par
from torch.nn.parameter import Parameter
class AbstractNMM(torch.nn.Module):
    # This is the abstract class for the models (typically neural mass models) that are being trained. 
    # The neuroimaging modality might be integrated into the model as well. 

    def __init__(self, params):
        super(AbstractNMM, self).__init__() # May not want to enherit from torch.nn.Module in the future 
        self.params = params
        self.state_names = ["None"] # The names of the state variables of the model
        self.output_names = ["None"] # The variable to be used as output from the NMM, for purposes such as the input to an objective function
        self.track_params = [] # Which NMM Parameters to track over training
        
        
    def info(self):
        # Information about the model, which may be used by other classes to know which variables to use. 
        
        return {"state_names": self.state_names, 
                "output_names": self.output_names,
                "track_params": self.track_params}
            
    def setModelParameters(self):
        # Setting the parameters that will be optimized as either model parameters or 2ndLevel/hyper
        # parameters (for various optional features). 
        # This should be called in the __init__() function implementation for convenience if possible.
        """
        Sets the parameters of the model.
        """

        param_reg = []
        param_hyper = []

        
        var_names = [a for a in dir(self.params) if (type(getattr(self.params, a)) == par)]
        for var_name in var_names:
            var = getattr(self.params, var_name)
            if (var.fit_par):
                if var_name == 'lm':
                    size = var.val.shape
                    var.val = Parameter(var.val) 
                    var.prior_mean = Parameter(var.prior_mean)
                    var.prior_precision = Parameter(var.prior_precision)
                    param_reg.append(var.val)
                    if var.fit_hyper:
                        param_hyper.append(var.prior_mean)
                        param_hyper.append(var.prior_precision)
                    self.track_params.append(var_name)
                else:
                    var.val = Parameter(var.val) # TODO: This is not consistent with what user would expect giving a variance
                    var.prior_mean = Parameter(var.prior_mean)
                    var.prior_precision = Parameter(var.prior_precision)
                    param_reg.append(var.val)
                    if var.fit_hyper:
                        param_hyper.append(var.prior_mean)
                        param_hyper.append(var.prior_precision)
                    self.track_params.append(var_name)



        self.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}
        
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