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

    def __init__(self, val, prior_mean = None, prior_var = None, fit_par = False, fit_hyper = False, asLog = False, isPlastic = False, device = torch.device('cpu')):
        '''
        
        Parameters
        ----------
        val : Float (or Array)
            The parameter value (or an array of node specific parameter values)
        prior_mean : Float
            Prior mean of the data value
        prior_var : Float
            Prior variance of the value
        fit_par: Bool
            Whether the parameter value should be set to as a PyTorch Parameter
        fit_hyper : Bool
            Whether the parameter prior mean and prior variance should be set as a PyTorch Parameter
        asLog : Bool
            Whether the log of the parameter value will be stored instead of the parameter itself (will prevent parameter from being negative).
        isPlastic : Bool
            A future potential feature to be implemented
        device: torch.device
            Whether to run on CPU or GPU
        '''
        
        if numpy.all(prior_mean != None) & numpy.all(prior_var != None) & (asLog == False):
            self.has_prior = True
        elif numpy.all(prior_mean != None) & numpy.all(prior_var != None):
            raise ValueError("currently asLog representation can not be used with priors")
        elif numpy.all(prior_mean != None) | numpy.all(prior_var != None):
            raise ValueError("prior_mean and prior_var must either be both None or both set")
        else:
            self.has_prior = False
            prior_mean = 0
            prior_var = 0
    
        self.val = torch.tensor(val, dtype=torch.float32).to(device)
        self.asLog = asLog # Store log(val) instead of val directly, so that val itself will always stay positive during training
        if asLog:
            self.val = torch.log(self.val) #TODO: It's not ideal that the attribute is called val when it might be log(val)
            
        self.prior_mean = torch.tensor(prior_mean, dtype=torch.float32).to(device)
        self.prior_var = torch.tensor(prior_var, dtype=torch.float32).to(device)
        self.fit_par = fit_par
        self.fit_hyper = fit_hyper
        
        self.isPlastic = isPlastic #Custom functionality, may not be compatible with other features
        
        self.device = device
        
        if fit_par:
            self.val = torch.nn.parameter.Parameter(self.val)
        
        if fit_hyper:
            self.prior_mean = torch.nn.parameter.Parameter(self.prior_mean)
            self.prior_var = torch.nn.parameter.Parameter(self.prior_var)
    
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
    
    def npValue(self):
        '''
        Returns
        --------
        NumPy of Value
            The parameter value(s) as a NumPy Array
        '''
        
        if self.asLog:
            return numpy.exp(self.val.detach().clone().cpu().numpy())
        else:
            return self.val.detach().clone().cpu().numpy()
    
    def to(self, device):
        '''
        '''
        self.device = device
        
        if self.fit_par:
            self.val = torch.nn.Parameter(self.val.detach().clone().to(device))
        else:
            self.val = self.val.to(device) 
            
        if self.fit_hyper:            
            self.prior_mean = torch.nn.Parameter(self.prior_mean.detach().clone().to(device))
            self.prior_var = torch.nn.Parameter(self.prior_var.detach().clone().to(device))
        else:
            self.prior_mean = self.prior_mean.to(device)
            self.prior_var = self.prior_var.to(device) 


    def randSet(self):
        '''
        This method sets the initial value using the mean and variance of the priors.
        '''
        
        if self.has_prior:
            self.var = self.prior_mean.detach() + self.prior_var.detach() * torch.randn(1)
            if self.fit_par:
                self.val = torch.nn.parameter.Parameter(self.val)
        else:
            raise ValueError("must have priors provided at par object initialization to use this method")
    
    # The below is so that the parameter object can be used in equations.
    # Additional operations may need to be added.
    
    def __pos__(self):
        if self.asLog:
            return torch.exp(self.val)
        else:
            return self.val
    
    def __neg__(self):
        if self.asLog:
            return -torch.exp(self.val)
        else:
            return -self.val
    
    def __add__(self, num):
        if self.asLog:
            return torch.exp(self.val) + num
        else:
            return self.val + num
    
    def __radd__(self, num):
        if self.asLog:
            return num + torch.exp(self.val)
        else:
            return num + self.val
    
    def __sub__(self, num):
        if self.asLog:
            return torch.exp(self.val) - num
        else:
            return self.val - num
    
    def __rsub__(self, num):
        if self.asLog:
            return num - torch.exp(self.val)
        else:
            return num - self.val
    
    def __mul__(self, num):
        if self.asLog:
            return torch.exp(self.val) * num
        else:
            return self.val * num
    
    def __rmul__(self, num):
        if self.asLog:
            return num * torch.exp(self.val)
        else:
            return num * self.val
    
    def __truediv__(self, num):
        if self.asLog:
            return torch.exp(self.val) / num
        else:
            return self.val / num
    
    def __rtruediv__(self, num):
        if self.asLog:
            return num / torch.exp(self.val)
        else:
            return num / self.val
