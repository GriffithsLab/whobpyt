"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for output datatype
"""

import pickle

import numpy as np  # for numerical operations

import whobpyt.datatypes.parameter


class TrainingStats:
    """
        The class replaces OutputNM.
        This class is responsible for recording stats during training including:
            - The training and validation losses over time
            - The change in parameters over time
            - changing hyperparameters over time like learing rates
            
        These are things typically recorded on a per window/epoch basis  
    """

    def __init__(self, model):
        model_info = model.info()
        self.track_params = model.track_params

        self.loss = []

        self.connectivity = []
        self.leadfield = []

        self.fit_params = {}

        #for name in set(self.state_names + self.output_names):
        #    for m in ['train', 'test']:
        #        setattr(self, name + '_' + m, [])

        #vars_name = [a for a in dir(model.params) if not a.startswith('__') and not callable(getattr(model.params, a))]
        #for var in vars_name:
        #    if (type(getattr(model.params, var)) != whobpyt.datatypes.parameter.par):
        #        if np.any(getattr(model.params, var)[1] > 0):
        #            if var != 'std_in':
        #                setattr(self, var, np.array([]))
        #                for stat_var in ['m', 'v_inv']:
        #                    setattr(self, var + '_' + stat_var, [])
        #            else:
        #                setattr(self, var, [])
        #    
        #    else:
        #    
        #        if getattr(model.params, var).fit_par:
        #            setattr(self, var, np.array([]))
        #        
        #        if getattr(model.params, var).fit_hyper:
        #            for stat_var in ['m', 'v_inv']:
        #                setattr(self, var + '_' + stat_var, [])
            
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def reset(self):
        self.loss = []
        
        self.network_con = []
        self.leadfield = []

        self.nmm_pars = {}
        self.other_pars = {}

    def appendLoss(self, newValue):
        """ Append Trainig Loss """
        self.loss.append(newValue)
        
    def appendSC(self, newValue):
        """ Append Network Connections """
        self.connectivity.append(newValue)
        
    def appendLF(self, newValue):
        """ Append Lead Field Loss """
        self.leadfield.append(newValue)

    def appendParam(self, newValues):
        """ Append Fit Parameters """
        if (self.fit_params == {}):
            for name in newValues.keys():
                self.fit_params[name] = [newValues[name]]
        else:
            for name in newValues.keys():
                self.fit_params[name].append(newValues[name])