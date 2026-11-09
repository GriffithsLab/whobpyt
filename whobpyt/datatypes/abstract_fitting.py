import torch
import pickle

from .abstract_neural_model import AbstractNeuralModel
from .abstract_loss import AbstractLoss
from .outputs import TrainingStats

class AbstractFitting():
    """Base class for fitting paradigms.

    A fitting paradigm runs a model, compares its output with data using an
    objective function, and updates the model's learnable parameters.
    Subclasses implement :meth:`train`, :meth:`evaluate` and :meth:`simulate`.

    Parameters
    ----------
    model : AbstractNeuralModel
        The model to fit.
    cost : AbstractLoss
        The objective function.
    device : torch.device
        Device to compute on.

    Attributes
    ----------
    trainingStats : TrainingStats
        Loss and parameter values recorded during training.
    lastRec : dict of Timeseries or None
        The simulated time series from the most recent run.
    """

    def __init__(self, model: AbstractNeuralModel, cost: AbstractLoss, device = torch.device('cpu')):
        
        self.model = model
        self.cost = cost
        self.device = device
        
        self.trainingStats = TrainingStats(self.model)
        self.lastRec = None # Saves the last simulation during training
    
    
    def save(self, filename):
        """Save this object, including the model and training history, with pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
        
    def train(self, *args, **kwargs):
        """Fit the model's learnable parameters to data."""
        raise NotImplementedError
    
    
    def evaluate(self, *args, **kwargs):
        """Run the model and compare it with data, without updating parameters."""
        raise NotImplementedError
    
    
    def simulate(self, *args, **kwargs):
        """Run the model for a fixed duration."""
        raise NotImplementedError
    
    
