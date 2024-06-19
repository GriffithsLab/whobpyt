"""
Authors: Andrew Clappison, John Griffiths, Zheng Wang, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita
"""

import torch
from .parameter import Parameter as par

class AbstractLoss:
    # This is the abstract class for objective function components, or for a custom objective function with multiple components. 

    def __init__(self, simKey = None, model = None, device = torch.device('cpu')):
    
        self.simKey = simKey #This is a string key to extract from the dictionary of simulation outputs the time series used by the objective function
        self.device =  device
        self.model = model
    
    def main_loss(self, simData, empData):
        # Calculates a loss to be backpropagated through
        # If the objective function needs additional info, it should be defined at initialization so that the parameter fitting paradigms don't need to change
        
        # simData: is a dictionary of simulated state variable/neuroimaging modality time series. Typically accessed as simData[self.simKey].
        # empData: is the target either as a time series or a calculated phenomena metric
        
        pass
    
    def prior_loss(self):
        loss_prior = []
        lb =0.001
        m = torch.nn.ReLU()
        variables_p = [a for a in dir(self.model.params) if type(getattr(self.model.params, a)) == par]
        # get penalty on each model parameters due to prior distribution
        for var_name in variables_p:
            # print(var)
            var = getattr(self.model.params, var_name)
            if var.fit_hyper:
                loss_prior.append(torch.sum((lb + m(var.prior_precision)) * \
                                                (m(var.val) - m(var.prior_mean)) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_precision)))) #TODO: Double check about converting _v_inv 
        return loss_prior
