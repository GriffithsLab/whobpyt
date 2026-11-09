"""
Authors: Andrew Clappison, John Griffiths, Zheng Wang, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita
"""

import torch
from .parameter import Parameter as par

class AbstractLoss:
    """Base class for objective functions.

    Subclasses implement :meth:`main_loss`, which compares simulated and
    empirical data. A full objective can combine it with :meth:`prior_loss`,
    as :class:`~whobpyt.optimization.custom_cost_JR.CostsJR` does.

    Parameters
    ----------
    simKey : str
        Which of the model's outputs or state variables to compare with the
        data, for example ``'eeg'``.
    model : AbstractNeuralModel
        The model being fitted. Needed by :meth:`prior_loss`.
    device : torch.device
        Device to compute on.
    """

    def __init__(self, simKey = None, model = None, device = torch.device('cpu')):
    
        self.simKey = simKey #This is a string key to extract from the dictionary of simulation outputs the time series used by the objective function
        self.device =  device
        self.model = model
    
    def main_loss(self, simData, empData):
        """Return the loss comparing simulated and empirical data.

        Any other information the loss needs should be given when the object
        is created, so the fitting classes can call every loss the same way.

        Parameters
        ----------
        simData : dict of torch.Tensor
            Simulated time series, keyed by variable name. Usually accessed as
            ``simData[self.simKey]``.
        empData : torch.Tensor
            The target: an empirical time series or a summary measure
            computed from one.

        Returns
        -------
        torch.Tensor
            A scalar loss to back-propagate.
        """
        pass
    
    def prior_loss(self):
        """Return the penalties keeping learnable parameters near their priors.

        For every parameter whose prior is learned (``fit_hyper=True``), the
        penalty is the precision-weighted squared distance between the value
        and the prior mean, minus the log precision.

        Returns
        -------
        list of torch.Tensor
            One penalty per parameter; sum them to add to the loss.
        """
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
