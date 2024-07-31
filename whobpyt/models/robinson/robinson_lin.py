"""

WhoBPyt Robinson Linear model classes
--------------------------------------

Authors: Zheng Wang, John Griffiths, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Shreyas Harita, Minarose Ismail

Neural Mass Model fitting module for Linear Robinson for M/EEG

"""

"""
Importage
---------
"""

from torch.nn.parameter import Parameter as ptParameter
from torch.nn import ReLU as ptReLU
from torch.linalg import norm as ptnorm
from torch import (tensor as pttensor, float32 as ptfloat32, sum as ptsum, exp as ptexp, diag as ptdiag, 
                   transpose as pttranspose, zeros_like as ptzeros_like, int64 as ptint64, randn as ptrandn, 
                   matmul as ptmatmul, tanh as pttanh, matmul as ptmatmul, reshape as ptreshape, sqrt as ptsqrt,
                   ones as ptones, cat as ptcat)

# Numpy stuff
from numpy.random import uniform 
from numpy import ones,zeros

# WhoBPyT stuff
from ...datatypes import AbstractNeuralModel, AbstractParams, Parameter as par
from ...functions.arg_type_check import method_arg_type_check

"""
Robinson Linear params class
---------------
"""

class RobinsonLinParams(AbstractParams):
    """
    A class for setting the parameters of a neural mass model for M/EEG data fitting.

    Attributes:
      Gee (par): Gain of excitatory-excitatory
      Gei (par): Gain of excitatory-inhibitory
      Gese (par): Gain of excitatory-relay nuclei-excitatory
      Gesre (par): Gain of excitatory-relay nuclei-reticular nucleus-excitatory
      Gsrs (par): Gain of relay nuclei-reticular nucleus-relay nuclei
      alpha (par): Inverse decay time
      beta (par): Inverse rise time 
      t0: Corticothalamic delay
      EMG: Artifacts

      Not fitted params
      k0 (par): 10;  Volume conduction parameter 
      phin (par): 1e-5; Input
      kmax (par) 4;
    """
    def __init__(self, **kwargs):
        """
        Initializes the RobinsonLinParams object.

        Args:
            **kwargs: Keyword arguments for the model parameters.

        Returns:
            None
        """
        # Initializing EC Abeysuriya 2015
        param = {
            "Gee": par(2.07),
            "Gei": par(-4.11),            
            "Gese": par(0.77*7.77),
            "Gesre": par(-3.30*0.77*0.66),
            "Gsrs": par(-3.30*0.20),
            "alpha": par(83),
            "beta": par(769),
            "t0": par(0.085),
            "EMG": par(0),   
            "k0": par(10),
            "phin": par(1e-5)
            "kmax": par(4)
         }
        
         for var in param:
            setattr(self, var, param[var])

         for var in kwargs:
            setattr(self, var, kwargs[var])

"""
RobinsonLin model class
--------------
"""

class RobinsonLinModel(AbstractNeuralModel):
    """
    A module for forward model (Robinson) to simulate M/EEG power spectra
    
    Attibutes
    ---------
    state_size : int
        Equal to 1 for phie: Number of states in the RobinsonLin model
        
    output_size : int
        Equal to 1: Power spectra of phie

    hidden_size: int
        Number of step_size for each sampling step

    step_size: float
        Integration step for forward model

    use_fit_gains: bool
        Flag for fitting gains. 1: fit, 0: not fit

    params: RobinsonLinParams
        Model parameters object.

    Methods
    -------
    createIC(self, ver):
        Creates the initial conditions for the model.

    setModelParameters(self):    
        Sets the parameters of the model.
    
    forward(params)
        Forward pass for generating power spectra with current model parameters
    """

    def __init__(self, 
                 params: RobinsonLinParams, 
                 step_size=0.0001, 
                 output_size=1, 
                 use_fit_gains=True
                 ):               
        """
        Parameters
        ----------
        step_size: float
            Integration step for forward model
        output_size : int
            Output of phie
        use_fit_gains: bool
            Flag for fitting gains. 1: fit, 0: not fit
        params: RobinsonLinParams
            Model parameters object.
        """
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types
        
        super(RobinsonLinModel, self).__init__(params)
        self.state_names = ['phie']
        self.output_names = ["power_spectra"]
        self.track_params = [] #Is populated during setModelParameters()
        
        self.model_name = "RobinsonLin"
        self.state_size = 1 # Power spectra phie
        self.step_size = pttensor(step_size, dtype=ptfloat32)  # integration step 0.1 ms
        self.output_size = output_size  # number of power spectra
        self.use_fit_gains = use_fit_gains  # flag for fitting gainsfm
        self.params = params
        
        self.setModelParameters()
        self.setModelSCParameters()  

  
    def createIC(self, ver, state_lb = -0.5, state_ub = 0.5):
        """
        Creates the initial conditions for the model.

        Parameters
        ----------
        ver : int # TODO: ADD MORE DESCRIPTION
            Initial condition version. (in the Robinson model, the version is not used. It is just for consistency with other models)

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, state_size) with random values between `state_lb` and `state_ub`.
        """
        n_states = self.state_size
        init_conds = uniform(state_lb, state_ub, (n_nodes, n_states))
        ptinit_conds = pttensor(init_conds, dtype=ptfloat32)
                             
        return ptinit_conds

  
    def forward(self, external, hx, hE):   
      # STILL TO DO 
