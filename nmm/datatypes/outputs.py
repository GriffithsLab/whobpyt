"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for output datatype
"""

import pickle

import numpy as np  # for numerical operations


class OutputNM:
    mode_all = ['train', 'test']
    stat_vars_all = ['m', 'v_inv']
    output_name = ['bold']

    def __init__(self, model_name, param, fit_weights=False, fit_lfm=False):
        state_names = ['E']
        self.loss = np.array([])
        if model_name == 'RWW':
            state_names = ['E', 'I', 'x', 'f', 'v', 'q']
            self.output_name = "bold"
        elif model_name == "JR":
            state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
            self.output_name = "eeg"
        elif model_name == 'LIN':
            state_names = ['E']
            self.output_name = "bold"

        for name in state_names + [self.output_name]:
            for m in self.mode_all:
                setattr(self, name + '_' + m, [])

        vars_name = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars_name:
            if np.any(getattr(param, var)[1] > 0):
                if var != 'std_in':
                    setattr(self, var, np.array([]))
                    for stat_var in self.stat_vars_all:
                        setattr(self, var + '_' + stat_var, [])
                else:
                    setattr(self, var, [])
        if fit_weights:
            self.weights = []

        if model_name == 'JR' and fit_lfm:
            self.leadfield = []

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
