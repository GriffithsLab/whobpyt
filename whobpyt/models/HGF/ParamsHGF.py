import torch
from whobpyt.datatypes import AbstractParams, par

class ParamsHGF(AbstractParams):
    def __init__(self, **kwargs):

        super(ParamsHGF, self).__init__(**kwargs)

        params = {

            "omega_3": par(0.03),  # standard deviation of the Gaussian noise
            "omega_2": par(0.02),  # standard deviation of the Gaussian noise

            "kappa": par(1.),  # scale of the external input
            "x2mean" : par(1),
            "deca2" : par(1),
            "deca3" : par(1),
            "g_x2_x3" : par(1),
            "g_x3_x2" : par(1),
            "c" : par(1)
            }

        for var in params:
            if var not in self.params:
                self.params[var] = params[var]

        self.setParamsAsattr()