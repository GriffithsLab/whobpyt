import numpy as np
from whobpyt.datatypes import AbstractParams, par

class ParamsEEG(AbstractParams):
    
    def __init__(self, **kwargs):
        """
        Initializes the ParamsCB object.

        Args:
            **kwargs: Keyword arguments for the model parameters.

        Returns:
            None
        """
        super(ParamsEEG, self).__init__(**kwargs)
        params = {
            
            "lm": par(1)
        }

        for var in params:
            if var not in self.params:
                self.params[var] = params[var]
        
        self.setParamsAsattr()
        