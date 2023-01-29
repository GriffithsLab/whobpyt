"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for wong-wang model
"""

import torch
from whobpyt.datatypes.AbstractParams import AbstractParams
from whobpyt.datatypes.AbstractNMM import AbstractNMM
from whobpyt.functions.pytorch_funs import setModelParameters
from whobpyt.functions.pytorch_funs import integration_forward


class ParamsRWW(AbstractParams):
    
    def __init__(self, **kwargs):
       
        param = {

            "std_in": [0.02, 0],  # standard deviation of the Gaussian noise
            "std_out": [0.02, 0],  # standard deviation of the Gaussian noise
            # Parameters for the ODEs
            # Excitatory population
            "W_E": [1., 0],  # scale of the external input
            "tau_E": [100., 0],  # decay time
            "gamma_E": [0.641 / 1000., 0],  # other dynamic parameter (?)

            # Inhibitory population
            "W_I": [0.7, 0],  # scale of the external input
            "tau_I": [10., 0],  # decay time
            "gamma_I": [1. / 1000., 0],  # other dynamic parameter (?)

            # External input
            "I_0": [0.32, 0],  # external input
            "I_external": [0., 0],  # external stimulation

            # Coupling parameters
            "g": [20., 0],  # global coupling (from all nodes E_j to single node E_i)
            "g_EE": [.1, 0],  # local self excitatory feedback (from E_i to E_i)
            "g_IE": [.1, 0],  # local inhibitory coupling (from I_i to E_i)
            "g_EI": [0.1, 0],  # local excitatory coupling (from E_i to I_i)

            "aE": [310, 0],
            "bE": [125, 0],
            "dE": [0.16, 0],
            "aI": [615, 0],
            "bI": [177, 0],
            "dI": [0.087, 0],

            # Output (BOLD signal)

            "alpha": [0.32, 0],
            "rho": [0.34, 0],
            "k1": [2.38, 0],
            "k2": [2.0, 0],
            "k3": [0.48, 0],  # adjust this number from 0.48 for BOLD fluctruate around zero
            "V": [.02, 0],
            "E0": [0.34, 0],
            "tau_s": [1 / 0.65, 0],
            "tau_f": [1 / 0.41, 0],
            "tau_0": [0.98, 0],
            "mu": [0.5, 0]

        }

        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])


class RNNRWW(AbstractNMM):
    """
    A module for forward model (WWD) to simulate a window of BOLD signals
    Attibutes
    ---------
    state_size : int
        the number of states in the WWD model
    input_size : int
        the number of states with noise as input
    tr : float
        tr of fMRI image
    step_size: float
        Integration step for forward model
    steps_per_TR: int
        the number of step_size in a tr
    TRs_per_window: int
        the number of BOLD signals to simulate
    node_size: int
        the number of ROIs
    sc: float node_size x node_size array
        structural connectivity
    fit_gains: bool
        flag for fitting gains 1: fit 0: not fit
    g, g_EE, gIE, gEI: tensor with gradient on
        model parameters to be fit
    gains_con: tensor with node_size x node_size (grad on depends on fit_gains)
        connection gains exp(gains_con)*sc
    std_in std_out: tensor with gradient on
        std for state noise and output noise
    g_m g_v f_EE_m g_EE_v sup_ca sup_cb sup_cc: tensor with gradient on
        hyper parameters for prior distribution of g gEE gIE and gEI
    Methods
    -------
    forward(input, external, hx, hE)
        forward model (WWD) for generating a number of BOLD signals with current model parameters
    """
    state_names = ['E', 'I', 'x', 'f', 'v', 'q']
    model_name = "RWW"
    use_fit_lfm = False
    input_size = 2

    def __init__(self, node_size: int,
                 TRs_per_window: int, step_size: float, sampling_size: float, tr: float, sc: float, use_fit_gains: bool,
                 param: ParamsRWW, use_Bifurcation=True, use_Gaussian_EI=False, use_Laplacian=True,
                 use_dynamic_boundary=True) -> None:
        """
        Parameters
        ----------

        tr : float
            tr of fMRI image
        step_size: float
            Integration step for forward model
        TRs_per_window: int
            the number of BOLD signals to simulate
        node_size: int
            the number of ROIs
        sc: float node_size x node_size array
            structural connectivity
        use_fit_gains: bool
            flag for fitting gains 1: fit 0: not fit
        use_Laplacian: bool
            using Laplacian or not
        param: ParamsModel
            define model parameters(var:0 constant var:non-zero Parameter)
        """
        super(RNNRWW, self).__init__()
        self.state_size = 6  # 6 states WWD model
        # self.input_size = input_size  # 1 or 2
        self.tr = tr  # tr fMRI image
        self.step_size = step_size  # integration step 0.05
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.sampling_size = sampling_size
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.sc_fitted = torch.tensor(sc, dtype=torch.float32)  # placeholder
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        self.use_Laplacian = use_Laplacian
        self.use_Bifurcation = use_Bifurcation
        self.use_Gaussian_EI = use_Gaussian_EI
        self.use_dynamic_boundary = use_dynamic_boundary
        self.param = param

        self.output_size = node_size  # number of EEG channels

    def setModelParameters(self):
        # set states E I f v mean and 1/sqrt(variance)
        return setModelParameters(self)

    def forward(self, external, hx, hE):
        return integration_forward(self, external, hx, hE)
