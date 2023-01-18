"""
WhoBPyt Visualization Classes
"""

class OutputNM():
    mode_all = ['train', 'test']
    stat_vars_all = ['m', 'v']

    def __init__(self, model_name, node_size, param, fit_weights=False, fit_lfm=False):
        self.loss = np.array([])
        if model_name == 'WWD':
            state_names = ['E', 'I', 'x', 'f', 'v', 'q']
            self.output_name = "bold"
        elif model_name == "JR":
            state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
            self.output_name = "eeg"
        for name in state_names + [self.output_name]:
            for m in self.mode_all:
                setattr(self, name + '_' + m, [])

        vars = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars:
            if np.any(getattr(param, var)[1] > 0):
                if var != 'std_in':
                    setattr(self, var, np.array([]))
                    for stat_var in self.stat_vars_all:
                        setattr(self, var + '_' + stat_var, [])
                else:
                    setattr(self, var, [])
        if fit_weights == True:
            self.weights = []
        if model_name == 'JR' and fit_lfm == True:
            self.leadfield = []

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
