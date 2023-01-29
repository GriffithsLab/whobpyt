"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for linear model
"""

import torch
from whobpyt.datatypes.AbstractParams import AbstractParams
from whobpyt.datatypes.AbstractNMM import AbstractNMM
from whobpyt.functions.pytorch_funs import integration_forward
from whobpyt.functions.pytorch_funs import setModelParameters


class ParamsLIN(AbstractParams):

    def __init__(self, model_name, **kwargs):
        param = {'g': [100, 0]}

        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])


class RNNLIN(AbstractNMM):
    """
    A module for forward model (Linear Model with 1 population) to simulate a window of BOLD signals
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
    g: tensor with gradient on
        model parameters to be fit
    gains_con: tensor with node_size x node_size (grad on depends on fit_gains)
        connection gains exp(gains_con)*sc
    std_in std_out: tensor with gradient on
        std for state noise and output noise

    Methods
    -------
    forward(input, external, hx, hE)
        forward model (WWD) for generating a number of BOLD signals with current model parameters
    """
    state_names = ['E', 'x', 'f', 'v', 'q']
    model_name = "LIN"
    use_fit_lfm = False
    input_size = 1

    def __init__(self, node_size: int,
                 TRs_per_window: int, step_size: float, sampling_size: float, tr: float, sc: float, use_fit_gains: bool,
                 param: ParamsLIN, use_Laplacian=True) -> None:
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
        super(RNNLIN, self).__init__()
        self.state_size = 5  # 6 states WWD model
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

        self.param = param

        self.output_size = node_size  # number of EEG channels

    def setModelParameters(self):
        return setModelParameters(self)

    def forward(self, external, hx, hE):
        return integration_forward(self, external, hx, hE)
