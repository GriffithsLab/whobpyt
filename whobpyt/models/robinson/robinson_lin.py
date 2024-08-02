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
      Ges (par): Gain of excitatory-relay nuclei
      Gse (par): Gain of relay nuclei-excitatory
      Gsr (par): Gain of relay nuclei-reticular nucleus
      Grs (par): Gain of reticular nucleus-relay nulei
      Gre (par): Gain of reticular nucleus-excitatory
      Gsn (par): Gain of relay nuclei-input
      Gese (par): Gain of excitatory-relay nuclei-excitatory
      Gesre (par): Gain of excitatory-relay nuclei-reticular nucleus-excitatory
      Gsrs (par): Gain of relay nuclei-reticular nucleus-relay nuclei
      alpha (par): Inverse decay time
      beta (par): Inverse rise time 
      t0: Corticothalamic delay
      EMG: Artifacts

      Not fitted params
      gammae: 116
      k0 (par): 10;  Volume conduction parameter 
      re (par):
      phin (par): 1e-5; Input
      kmax (par) 4;
      re (par): 
      Lx (par): 
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
            "Ges": par(0.77),
            "Gse": par(7.77),
            "Gsr": par(-3.30),
            "Grs": par(0.20),
            "Gre": par(0.66),
            "Gsn": par(8.10),
            "Gese": par(0.77*7.77),
            "Gesre": par(-3.30*0.77*0.66),
            "Gsrs": par(-3.30*0.20),
            "G_esn": par(0.77*8.10), 
            "alpha": par(83),
            "beta": par(769),
            "gammae": par(116),
            "t0": par(0.085),
            "re": par(0.086),
            "EMG": par(0),   
            "k0": par(10),
            "phin": par(1e-5),
            "kmax": par(4),
            "Lx": par(0.5)
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

  
    def forward(self, external, w1):   
      # STILL TO DO 
      G_ei = self.params.Gei.value()
      G_ee = self.params.Gee.value()
      G_es = self.params.Ges.value()
      G_sn = self.params.Gsn.value()
      G_se = self.params.Gse.value()
      G_sr = self.params.Gsr.value()
      G_rs = self.params.Grs.value()
      G_re = self.params.Gre.value()
      alpha = 83
      beta = 769
      G_esn = G_es*G_sn
      G_srs = G_sr*G_rs
      G_esre = G_es*G_sr*G_re
      G_ese = G_es*G_se
      gamma_e = self.params.gammae.value()
      t0 = self.params.t0.value()
      r_e = self.params.re.value()
      phin = self.params.phin.value()
      EMG = self.params.EMG.value()
      kmax = self.params.kmax.value()
      k0 = self.params.k0.value()
      Lx = self.params.Lx.value()

      dk = 2*np.pi/Lx
      m_rows = list(range(-kmax, kmax + 1))
      n_cols = list(range(-kmax, kmax + 1))
      [kxa,kya] = np.meshgrid(dk*np.array(m_rows),dk*np.array(n_cols))
      k2 = kxa**(2)+kya**(2)
      k2u = np.unique(k2[:])
      counts, _ = np.histogram(k2, bins=np.append(k2u, k2u[-1] + 1))
      k2u = np.column_stack((k2u, counts))
      k2_volconduct = np.exp(-k2u[:,0]/k0**2);
      emg_f = 40;
      emg = ((w1/(2*np.pi*emg_f))**2)/((1+(w1/(2*np.pi*emg_f))**2)**2);
      if EMG==0:
        emg = 0;
    
      L = (1-(w1*1j/alpha))**(-1)*(1-(w1*1j/beta))**(-1)
      phin = 1e-5
      Gei_oneminus = 1-L*G_ei
      Gsrs_oneminus = 1-G_srs*(L**(2))
      re2 = r_e**2
      q2re2 = (1-(w1*1j/gamma_e))**(2) - (1/(1-G_ei*L))*(G_ee*L + ((G_ese*L**(2) + G_esre*L**(3))*np.exp(w1*1j*t0))/(1-G_srs*L**(2)))
      T_prefactor = (L**(2)*phin)/(Gei_oneminus*Gsrs_oneminus);
    
      P = np.zeros_like(w1)
      for j in range(k2u.shape[0]):
          contribution = k2u[j, 1] * np.abs(T_prefactor / (k2u[j, 0] * re2 + q2re2))**2 * k2_volconduct[j]
          P += contribution
    
      P = P + 1e-12*0.1*emg;
      return P
