import torch
import pickle

from whobpyt.datatypes.AbstractNMM import AbstractNMM
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.datatypes.outputs import TrainingStats

class AbstractFitting():
    # AbstractFitting is a template class for different parameter optimization or machine learning paradigms to inherit from. 

    def __init__(self, model: AbstractNMM, cost: AbstractLoss, device = torch.device('cpu')):
        # Initializing the class
        
        self.model = model
        self.cost = cost
        self.device = device
        
        self.trainingStats = TrainingStats(self.model)
        self.lastRec = None # Saves the last simulation during training
    
    
    def save(self, filename):
        # Saving the entire class
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
        
    def train():
        # This function is for training of a model. 
        pass
    
    
    def evaluate():
        # The function is intended to calculate statistics as they are calculated during training, but not updating model parameters. 
        pass
    
    
    def simulate():
        # This function is intended for running some fixed duration of simulation.
        pass
    
    