import torch
import numpy

class par:
    # Goals of this class:
    # This class contains a global parameter value or array of values (one per node)
    # It can also contain associated priors (mean and variance)
    # It also has attributes for whether the parameter and/or priors should be fit during training
    # It has method to have parameter as numpy value
    # It has method to set a random val using based on priors
    # It has functionality to represent val as log(val) so that during training val will be constrained to be positive

    def __init__(self, val, prior_mean = None, prior_var = None, fit_par = False, fit_hyper = False, asLog = False, isPlastic = False):
        
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
    
        self.val = torch.tensor(val, dtype=torch.float32)
        self.asLog = asLog # Store log(val) instead of val directly, so that val itself will always stay positive during training
        if asLog:
            self.val = torch.log(self.val) #TODO: It's not ideal that the attribute is called val when it might be log(val)
            
        self.prior_mean = torch.tensor(prior_mean, dtype=torch.float32)
        self.prior_var = torch.tensor(prior_var, dtype=torch.float32)
        self.fit_par = fit_par
        self.fit_hyper = fit_hyper
        
        self.isPlastic = isPlastic #Custom functionality, may not be compatible with other features
        
        if fit_par:
            self.val = torch.nn.parameter.Parameter(self.val)
        
        if fit_hyper:
            self.prior_mean = torch.nn.parameter.Parameter(self.prior_mean)
            self.prior_var = torch.nn.parameter.Parameter(self.prior_var)
    
    def value(self):
        if self.asLog:
            return torch.exp(self.val)
        else:
            return self.val
    
    def npValue(self):
        if self.asLog:
            return numpy.exp(self.val.detach().clone().numpy())
        else:
            return self.val.detach().clone().numpy()
        
    def randSet(self):
        # This method sets the initial value using the mean and variance of the priors
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
