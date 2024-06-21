from whobpyt.datatypes import AbstractMeasurementModel, AbstractNeuralModel, AbstractParams
from whobpyt.datatypes import Parameter as par, Timeseries

class EEG_Params(AbstractParams):
    def __init__(self, Lead_Field):
        
        #############################################
        ## EEG Lead Field
        #############################################
        
        self.LF = Lead_Field # This should be [num_regions, num_channels]
        
    def to(self, device):
        # Moves all parameters between CPU and GPU
        
        vars_names = [a for a in dir(self) if not a.startswith('__')]
        for var_name in vars_names:
            var = getattr(self, var_name)
            if (type(var) == par):
                var.to(device)
        