"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for output datatype
"""

import pickle

import numpy as np  # for numerical operations

import whobpyt.datatypes.parameter


class OutputNM:
    mode_all = ['train', 'test']
    stat_vars_all = ['m', 'v_inv']

    def __init__(self, model):
        self.loss = np.array([])
        
        model_info = model.info()
        self.state_names = model_info["state_names"]
        self.output_names = model_info["output_names"]

        for name in self.state_names + self.output_names:
            for m in self.mode_all:
                setattr(self, name + '_' + m, [])

        vars_name = [a for a in dir(model.params) if not a.startswith('__') and not callable(getattr(model.params, a))]
        for var in vars_name:
            if (type(getattr(model.params, var)) != whobpyt.datatypes.parameter.par):
                if np.any(getattr(model.params, var)[1] > 0):
                    if var != 'std_in':
                        setattr(self, var, np.array([]))
                        for stat_var in self.stat_vars_all:
                            setattr(self, var + '_' + stat_var, [])
                    else:
                        setattr(self, var, [])
            
            else:
            
                if getattr(model.params, var).fit_par:
                    setattr(self, var, np.array([]))
                
                if getattr(model.params, var).fit_hyper:
                    for stat_var in self.stat_vars_all:
                        setattr(self, var + '_' + stat_var, [])
            
        if hasattr(model, 'use_fit_gains'):
            if model.use_fit_gains:
                self.weights = []

        if hasattr(model, 'use_fit_lfm'):
            if model.use_fit_lfm:
                self.leadfield = []

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
