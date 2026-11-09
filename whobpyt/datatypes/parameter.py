"""
Authors: Andrew Clappison, John Griffiths, Zheng Wang, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita
"""


import torch
import numpy
import numpy as np

class Parameter:
    """A model parameter: its value and, optionally, a prior and whether it is learnable.

    Parameters
    ----------
    val : float or ndarray
        The value, or an array of values (for example one per region). For a
        learnable parameter with a prior and ``asRand=True``, this is
        replaced by a random draw from the prior (only its shape is used for
        arrays).
    prior_mean : float, optional
        Mean of the Gaussian prior.
    prior_std : float, optional
        Standard deviation of the Gaussian prior.
    fit_par : bool
        If True, the value is learned during training. If a prior is also
        given, the prior's mean and precision are learned too.
    asLog : bool
        If True, ``val`` (and the prior) are given on a log scale and the
        parameter's value is ``lb + exp(val)``, which keeps it positive.
    asRand : bool
        If True (and the parameter is learnable with a prior), start from a
        random draw from the prior instead of ``val``.
    lb : float
        Lower bound, added to the value.
    device : torch.device
        Device to store the parameter on.

    Attributes
    ----------
    val : torch.Tensor
        The stored (possibly log-scaled) value. Use :meth:`value` to get the
        parameter's actual value.
    prior_mean : torch.Tensor
        Prior mean, if a prior was given.
    prior_precision : torch.Tensor
        Prior precision (1 / variance), if a prior was given.
    fit_par : bool
        Whether the value is learned.
    fit_hyper : bool
        Whether the prior mean and precision are learned.

    Examples
    --------
    >>> from whobpyt.datatypes import Parameter as par
    >>> A = par(3.25)                               # fixed
    >>> a = par(100, 100, 2, True)                  # learnable, prior N(100, 2**2)
    >>> mu = par(np.log(1.5), np.log(1.5), 0.1, True, True, lb=0.1)  # positive
    """

    def __init__(self, val, prior_mean = None, prior_std = None, fit_par = False, asLog = False, asRand = True, lb = 0, device = torch.device('cpu')):
        self.fit_par = fit_par
        self.device = device
        self.asLog = asLog
        self.asRand = asRand
        self.lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        self.fit_hyper = False

        if self.fit_par:
            if np.all(prior_mean != None) & np.all(prior_std != None):

                prior_mean_ts = torch.tensor(prior_mean, dtype=torch.float32).to(self.device)
                self.prior_mean = prior_mean_ts
                prior_std_ts = torch.tensor(prior_std, dtype=torch.float32).to(self.device)
                self.prior_precision = 1/prior_std_ts**2
                if self.asRand == True:
                    if type(val) is np.ndarray:
                        val = prior_mean + prior_std * torch.randn_like(torch.tensor(val, dtype=torch.float32)).detach().numpy()
                    else:
                        val = prior_mean + prior_std * np.random.randn(1,)
                
                    
                val_ts = torch.tensor(val, dtype=torch.float32).to(self.device)
                self.val = val_ts
                self.fit_hyper = True

            else:
                val_ts = torch.tensor(val, dtype=torch.float32).to(self.device)
                self.val = val_ts
        else:
            self.val = torch.tensor(val, dtype=torch.float32).to(self.device)




    def to(self, device):
        '''Move the parameter to ``device`` (a :class:`torch.device`).'''
        self.device = device
        
        self.val = self.val.to(self.device)


    def npValue(self):
        '''
        Returns
        -------
        ndarray
            The parameter value(s), including the lower bound and undoing
            the log transform if ``asLog`` is set.
        '''
        
        if self.asLog:
            return self.lb.detach().clone().cpu().numpy() + np.exp(self.val.detach().clone().cpu().numpy())
        else:
            return self.lb.detach().clone().cpu().numpy() + self.val.detach().clone().cpu().numpy()

    def value(self):
        '''
        Returns
        -------
        torch.Tensor
            The parameter value(s), including the lower bound and undoing
            the log transform if ``asLog`` is set. Gradients flow through it.
        '''


        if self.asLog:
            return self.lb + torch.exp(self.val)
        else:
            return self.lb + self.val


    


