import torch

class par:
    # Goals of this class:
    # This class contains a global parameter value or array of values (one per node)
    # It can also contain associated priors and means
    # It also has attributes for whether the parameter and/or priors should be fit during training
    # It has method to have parameter as numpy value or torch tensor value
    # It has method to add noise to randomise initial conditoins

    def __init__(self, val, prior_mean = 0, prior_var = 0, fit_par = False, fit_hyper = False):
        self.val = val
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.fit_par = fit_par
        self.fit_hyper = fit_hyper
        
    def asNumpy():
        pass
        
    def asTorch():
        pass
        
    def addNoise():
        pass
    
    # The below is so that the parameter object can be used in equations as if it 
    # is a numpy array or torch tensor. Additional operations may need to be added.
    
    def __pos__(self):
        return self.val
    
    def __neg__(self):
        return -self.val
    
    def __add__(self, num):
        return self.val + num
    
    def __radd__(self, num):
        return num + self.val
    
    def __sub__(self, num):
        return self.val - num
    
    def __rsub__(self, num):
        return num - self.val
    
    def __mul__(self, num):
        return self.val * num
    
    def __rmul__(self, num):
        return num * self.val
    
    def __truediv__(self, num):
        return self.val / num
    
    def __rtruediv__(self, num):
        return num / self.val