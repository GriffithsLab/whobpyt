"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for output datatype
"""

import pickle

import numpy as np  # for numerical operations

import whobpyt.datatypes.parameter


class TrainingStats:
    '''
    This class is responsible for recording stats during training (it replaces OutputNM) including:
        - The training and validation losses over time
        - The change in model parameters over time
        - changing hyper parameters over time like learing rates TODO
        
    These are things typically recorded on a per window/epoch basis  
        
    Attributes
    ------------
    model_info : Dict
        Information about model being tracked during training. 
    track_params : List
        List of parameter names being tracked during training. 
    loss : List
        A list of loss values over training.
    connectivity : List
        A list of connectivity values over training.
    leadfield : List
        A list of leadfield matrices over training.
    fit_params : Dict
        A dictionary of lists where the key is the parameter name and the value is the list of parameter values over training. 
        
    '''

    def __init__(self, model):
        '''
        
        Parameters
        -----------
        model : AbstractNMM
            A model for which stats will be recorded during training. 
        
        '''
    
        model_info = model.info()
        self.track_params = model.track_params

        self.loss = []

        self.connectivity = []
        self.leadfield = []

        self.fit_params = {}
            
    def save(self, filename):
        '''
        Parameters
        ------------
        filename : String
            The filename to use to save the TrainingStats as a pickle object.
        
        '''
    
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def reset(self):
        '''
        Resets the attributes of the model to a pre-training state. 
        
        '''
    
        self.loss = []
        
        self.network_con = []
        self.leadfield = []

        self.fit_params = {}

    def appendLoss(self, newValue):
        """ 
        Append Trainig Loss

        Parameters
        -----------
        newValue : Float
            The loss value of objective function being tracked.
        
        """
        self.loss.append(newValue)
        
    def appendSC(self, newValue):
        """ 
        Append Network Connections 
        
        Parameters
        -----------
        newValue : Array
            Current state of the structural connectivity being tracked. 
        
        """
        self.connectivity.append(newValue)
        
    def appendLF(self, newValue):
        """ 
        Append Lead Field Loss 
        
        Parameters
        -----------
        newValue : Array
            Current state of a lead field matrix being tracked.
            
        """
        self.leadfield.append(newValue)

    def appendParam(self, newValues):
        """ 
        Append Fit Parameters
        
        Parameters
        ----------
        newValues : Dict
            Dictionary with current states of each model parameter being tracked.
        
        """
        if (self.fit_params == {}):
            for name in newValues.keys():
                self.fit_params[name] = [newValues[name]]
        else:
            for name in newValues.keys():
                self.fit_params[name].append(newValues[name])