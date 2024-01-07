import torch
import numpy

class par:
    '''
    Features of this class:
     - This class contains a global parameter value or array of values (one per node)
     - It can also contain associated priors (mean and variance)
     - It also has attributes for whether the parameter and/or priors should be fit during training
     - It has a method to return the parameter as numpy value
     - It has a method to set a random val using based on priors
     - It has functionality to represent val as log(val) so that during training val will be constrained to be positive

    Attributes
    ------------
    val : Tensor
        The parameter value (or an array of node specific parameter values)
    prior_mean : Tensor
        Prior mean of the data value
    prior_var : Tensor
        Prior variance of the value
    has_prior : Bool
        Whether the user provided a prior mean and variance
    fit_par: Bool
        Whether the parameter value should be set to as a PyTorch Parameter
    fit_hyper : Bool
        Whether the parameter prior mean and prior variance should be set as a PyTorch Parameter
    asLog : Bool
        Whether the log of the parameter value will be stored instead of the parameter itself (will prevent parameter from being negative).
    isPlastic : Bool
        Not yet implemented
    '''

    def __init__(self, val, prior_mean = None, prior_std = None, fit_par = False, asLog = False, device = torch.device('cpu')):
        '''

        Parameters
        ----------
        val : Float (or Array)
            The parameter value (or an array of node specific parameter values)
        prior_mean : Float
            Prior mean of the data value
        prior_std : Float
            Prior std of the value
        fit_par: Bool
            Whether the parameter value should be set to as a PyTorch Parameter
        device: torch.device
            Whether to run on CPU or GPU
        '''
        self.fit_par = fit_par
        self.device = device
        self.asLog = asLog

        if self.fit_par:
            if np.all(prior_mean != None) & np.all(prior_std != None):

                prior_mean_ts = torch.tensor(prior_mean, dtype=torch.float32).to(self.device)
                self.prior_mean = prior_mean_ts
                prior_std_ts = torch.tensor(prior_std, dtype=torch.float32).to(self.device)
                self.prior_var = 1/prior_std_ts**2
                if type(val) is np.ndarray:
                    val = prior_mean + prior_std * torch.randn_like(torch.tensor(val, dtype=torch.float32)).detach().numpy()
                else:
                    val = prior_mean + prior_std * np.random.randn(1,)
                val_ts = torch.tensor(val, dtype=torch.float32).to(self.device)
                self.val = val_ts

            else:
                prior_mean = val
                prior_mean_ts = torch.tensor(prior_mean, dtype=torch.float32).to(self.device)
                self.prior_mean = prior_mean_ts
                prior_std = np.abs(val)/10
                prior_std_ts = torch.tensor(prior_std, dtype=torch.float32).to(self.device)
                self.prior_var = 1/prior_std_ts**2
                if type(val) is np.ndarray:
                    val = prior_mean + prior_std * torch.randn_like(torch.tensor(val, dtype=torch.float32)).detach().numpy()
                else:
                    val = prior_mean + prior_std * np.random.randn(1,)
                val_ts = torch.tensor(val, dtype=torch.float32).to(self.device)
                self.val = val_ts




        else:
            self.val = torch.tensor(val, dtype=torch.float32).to(self.device)









    def value(self):
        '''
        Returns
        ---------
        Tensor of Value
            The parameter value(s) as a PyTorch Tensor
        '''


        if self.asLog:
            return torch.exp(self.val)
        else:
            return self.val
