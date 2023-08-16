"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting
module for linear model
"""

import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes.AbstractParams import AbstractParams
from whobpyt.datatypes.AbstractNMM import AbstractNMM
from whobpyt.functions.arg_type_check import method_arg_type_check
import numpy as np  # for numerical operations

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
                 param: ParamsLIN, use_Laplacian: bool = True) -> None:
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
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types       
        
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

    def info(self):
        return {"state_names": ['E'], "output_name": "bold"}

    def createIC(self, ver):
        # initial state
        return torch.tensor(0.2 * np.random.randn(self.node_size, self.state_size) + np.array(
                [0, 0.5, 1.0, 1.0, 1.0]), dtype=torch.float32)

    def setModelParameters(self):
        return setModelParameters(self)

    def forward(self, external, hx, hE):
        return integration_forward(self, external, hx, hE)

def setModelParameters(model):
    param_reg = []
    param_hyper = []
    # set gains_con as Parameter if fit_gain is True
    if model.use_fit_gains:
        model.gains_con = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                                 dtype=torch.float32))  # connenction gain to modify empirical sc
        param_reg.append(model.gains_con)
    else:
        model.gains_con = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)

    vars_name = [a for a in dir(model.param) if not a.startswith('__') and not callable(getattr(model.param, a))]
    for var in vars_name:
        if np.any(getattr(model.param, var)[1] > 0):
            setattr(model, var, Parameter(
                torch.tensor(getattr(model.param, var)[0] + getattr(model.param, var)[1] * np.random.randn(1, )[0],
                             dtype=torch.float32)))
            param_reg.append(getattr(model, var))

            if var not in ['std_in']:
                dict_nv = {'m': getattr(model.param, var)[0], 'v': 1 / (getattr(model.param, var)[1]) ** 2}

                dict_np = {'m': var + '_m', 'v': var + '_v_inv'}

                for key in dict_nv:
                    setattr(model, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
                    param_hyper.append(getattr(model, dict_np[key]))
        else:
            setattr(model, var, torch.tensor(getattr(model.param, var)[0], dtype=torch.float32))
    model.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}

            
def integration_forward(model, external, hx, hE):

    """
    Forward step in simulating the BOLD signal.
    Parameters
    ----------
    external: tensor with node_size x steps_per_TR x TRs_per_window x input_size
        noise for states

    hx: tensor with node_size x state_size
        states of WWD model
    Outputs
    -------
    next_state: dictionary with keys:
    'current_state''bold_window''E_window''I_window''x_window''f_window''v_window''q_window'
        record new states and BOLD
    """
    next_state = {}

    # hx is current state (6) 0: E 1:I (neural activities) 2:x 3:f 4:v 5:f (BOLD)

    x = hx[:, 1:2]
    f = hx[:, 2:3]
    v = hx[:, 3:4]
    q = hx[:, 4:5]

    dt = torch.tensor(model.step_size, dtype=torch.float32)

    # Generate the ReLU module for model parameters gEE gEI and gIE
    m = torch.nn.ReLU()

    # Update the Laplacian based on the updated connection gains gains_con.
    if model.sc.shape[0] > 1:

        # Update the Laplacian based on the updated connection gains gains_con.
        sc_mod = torch.exp(model.gains_con) * torch.tensor(model.sc, dtype=torch.float32)
        sc_mod_normalized = (0.5 * (sc_mod + torch.transpose(sc_mod, 0, 1))) / torch.linalg.norm(
            0.5 * (sc_mod + torch.transpose(sc_mod, 0, 1)))
        model.sc_fitted = sc_mod_normalized

        if model.use_Laplacian:
            lap_adj = -torch.diag(sc_mod_normalized.sum(1)) + sc_mod_normalized
        else:
            lap_adj = sc_mod_normalized

    else:
        lap_adj = torch.tensor(np.zeros((1, 1)), dtype=torch.float32)

    # placeholder for the updated current state
    current_state = torch.zeros_like(hx)

    # placeholders for output BOLD, history of E I x f v and q
    # placeholders for output BOLD, history of E I x f v and q
    bold_window = torch.zeros((model.node_size, model.TRs_per_window))
    E_window = torch.zeros((model.node_size, model.TRs_per_window))
    # I_window = torch.zeros((model.node_size,model.TRs_per_window))

    x_window = torch.zeros((model.node_size, model.TRs_per_window))
    f_window = torch.zeros((model.node_size, model.TRs_per_window))
    v_window = torch.zeros((model.node_size, model.TRs_per_window))
    q_window = torch.zeros((model.node_size, model.TRs_per_window))

    # E_hist = torch.zeros((model.node_size, model.TRs_per_window, model.steps_per_TR))

    E = hx[:, 0:1]

    # print(E_m.shape)
    # Use the forward model to get neural activity at ith element in the window.

    for TR_i in range(model.TRs_per_window):

        # print(E.shape)

        # Since tr is about second we need to use a small step size like 0.05 to integrate the model states.
        for step_i in range(model.steps_per_TR):
            # Calculate the input recurrents.
            IE = model.g * torch.matmul(lap_adj, E)  # input currents for E

            E_next = E + dt * (-E + torch.tanh(IE)) \
                     + torch.sqrt(dt) * torch.randn(model.node_size, 1) * (0.1 + m(
                model.std_in))  ### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)

            x_next = x + 1 * dt * (E - torch.reciprocal(model.tau_s) * x - torch.reciprocal(model.tau_f) * (f - 1))
            f_next = f + 1 * dt * x
            v_next = v + 1 * dt * (f - torch.pow(v, torch.reciprocal(model.alpha))) * torch.reciprocal(model.tau_0)
            q_next = q + 1 * dt * (
                    f * (1 - torch.pow(1 - model.rho, torch.reciprocal(f))) * torch.reciprocal(model.rho)
                    - q * torch.pow(v, torch.reciprocal(model.alpha)) * torch.reciprocal(v)) \
                     * torch.reciprocal(model.tau_0)

            # f_next[f_next < 0.001] = 0.001
            # v_next[v_next < 0.001] = 0.001
            # q_next[q_next < 0.001] = 0.001

            E = E_next  # torch.tanh(0.00001+1.0*E_next)
            x = x_next  # torch.tanh(x_next)
            f = (1 + torch.tanh(f_next - 1))
            v = (1 + torch.tanh(v_next - 1))
            q = (1 + torch.tanh(q_next - 1))
            # Put x f v q from each tr to the placeholders for checking them visually.
        E_window[:, TR_i] = E[:, 0]
        x_window[:, TR_i] = x[:, 0]
        f_window[:, TR_i] = f[:, 0]
        v_window[:, TR_i] = v[:, 0]
        q_window[:, TR_i] = q[:, 0]
        # Put the BOLD signal each tr to the placeholder being used in the cost calculation.

        bold_window[:, TR_i] = ((0.001 + m(model.std_out)) * torch.randn(model.node_size, 1) +
                                100.0 * model.V * torch.reciprocal(model.E0) * (model.k1 * (1 - q)
                                                                                + model.k2 * (
                                                                                        1 - q * torch.reciprocal(
                                                                                    v)) + model.k3 * (1 - v)))[:, 0]

    # Update the current state.
    # print(E_m.shape)
    current_state = torch.cat([E, x, f, v, q], dim=1)
    next_state['current_state'] = current_state
    next_state['bold_window'] = bold_window  # E_window#
    next_state['E_window'] = E_window
    next_state['x_window'] = x_window
    next_state['f_window'] = f_window
    next_state['v_window'] = v_window
    next_state['q_window'] = q_window

    return next_state, hE