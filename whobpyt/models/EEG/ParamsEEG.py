import numpy as np
from whobpyt.datatypes import AbstractParams, par

class EEG_Params(AbstractParams):
    def __init__(self, Lead_Field):
        
        #############################################
        ## EEG Lead Field
        #############################################
        
        self.LF = Lead_Field # This should be [num_regions, num_channels]
        