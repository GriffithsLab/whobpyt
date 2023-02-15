"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for JR with forward backward and lateral connection for EEG
"""

# @title new function PyTepFit

# Pytorch stuff


"""
Importage
"""
import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes.AbstractParams import AbstractParams
from whobpyt.datatypes.AbstractNMM import AbstractNMM
import numpy as np  # for numerical operations

class ParamsJR(AbstractParams):

    def __init__(self, **kwargs):

        param = {
            "A ": [3.25, 0], 
            "a": [100, 0.], 
            "B": [22, 0], 
            "b": [50, 0], 
            "g": [1000, 0],
            
            "c1": [135, 0.], 
            "c2": [135 * 0.8, 0.], 
            "c3 ": [135 * 0.25, 0.], 
            "c4": [135 * 0.25, 0.],
            
            "std_in": [100, 0], 
            "vmax": [5, 0], 
            "v0": [6, 0], 
            "r": [0.56, 0], 
            "y0": [2, 0],
            
            "mu": [.5, 0], 
            "k": [5, 0], 
            "cy0": [5, 0], 
            "ki": [1, 0]
        }
        
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])


class RNNJANSEN(AbstractNMM):
    """
    A module for forward model (JansenRit) to simulate a batch of EEG signals
    Attibutes
    ---------
    state_size : int
        the number of states in the JansenRit model
    input_size : int
        the number of states with noise as input
    tr : float
        tr of image
    step_size: float
        Integration step for forward model
    hidden_size: int
        the number of step_size in a tr
    TRs_per_window: int
        the number of EEG signals to simulate
    node_size: int
        the number of ROIs
    sc: float node_size x node_size array
        structural connectivity
    fit_gains: bool
        flag for fitting gains 1: fit 0: not fit
    g, c1, c2, c3,c4: tensor with gradient on
        model parameters to be fit
    w_bb: tensor with node_size x node_size (grad on depends on fit_gains)
        connection gains
    std_in std_out: tensor with gradient on
        std for state noise and output noise
    hyper parameters for prior distribution of model parameters
    Methods
    -------
    forward(input, noise_out, hx)
        forward model (JansenRit) for generating a number of EEG signals with current model parameters
    """
    state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
    model_name = "JR"

    def __init__(self, node_size: int,
                 TRs_per_window: int, step_size: float, output_size: int, tr: float, sc: float, lm: float, dist: float,
                 use_fit_gains: bool, use_fit_lfm: bool, param: ParamsJR) -> None:
        """
        Parameters
        ----------

        tr : float
            tr of image
        step_size: float
            Integration step for forward model
        
        TRs_per_window: int
            the number of EEG signals to simulate
        node_size: int
            the number of ROIs
        output_size: int
            the number of channels EEG
        sc: float node_size x node_size array
            structural connectivity
        use_fit_gains: bool
            flag for fitting gains 1: fit 0: not fit
        use_fit_lfm: bool
            flag for fitting gains 1: fit 0: not fit
        param from ParamJR
        """
        super(RNNJANSEN, self).__init__()
        self.state_size = 6  # 6 states JR model
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = torch.tensor(step_size, dtype=torch.float32)  # integration step 0.1 ms
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.output_size = output_size  # num of EEG channels
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.dist = torch.tensor(dist, dtype=torch.float32)
        self.lm = lm
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        self.use_fit_lfm = use_fit_lfm
        self.param = param

        self.output_size = lm.shape[0]  # number of EEG channels

    def info(self):
        return {"state_names": ['E', 'Ev', 'I', 'Iv', 'P', 'Pv'], "output_name": "eeg"}
    
    def createIC(self, ver):
        # initial state
        if (ver == 0):
            state_lb = 0.5
            state_ub = 2
        if (ver == 1):
            state_lb = 0
            state_ub = 5
        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size, self.state_size)),
                             dtype=torch.float32)
    
    def setModelParameters(self):
        # set states E I f v mean and 1/sqrt(variance)
        return setModelParameters(self)

    def forward(self, external, hx, hE):
        return integration_forward(self, external, hx, hE)

def sigmoid(x, vmax, v0, r):
    return vmax / (1 + torch.exp(r * (v0 - x)))
    
def sys2nd(A, a, u, x, v):
    return A * a * u - 2 * a * v - a ** 2 * x

def setModelParameters(model):
    param_reg = []
    param_hyper = []
    # set model parameters (variables: need to calculate gradient) as Parameter others : tensor
    # set w_bb as Parameter if fit_gain is True
    if model.use_fit_gains:
        model.w_bb = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                            dtype=torch.float32))  # connenction gain to modify empirical sc
        model.w_ff = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                            dtype=torch.float32))
        model.w_ll = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                            dtype=torch.float32))
        param_reg.append(model.w_ll)
        param_reg.append(model.w_ff)
        param_reg.append(model.w_bb)
    else:
        model.w_bb = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)
        model.w_ff = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)
        model.w_ll = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)

    if model.use_fit_lfm:
        model.lm = Parameter(torch.tensor(model.lm, dtype=torch.float32))  # leadfield matrix from sourced data to eeg
        param_reg.append(model.lm)
    else:
        model.lm = torch.tensor(model.lm, dtype=torch.float32)  # leadfield matrix from sourced data to eeg

    vars_name = [a for a in dir(model.param) if not a.startswith('__') and not callable(getattr(model.param, a))]
    for var in vars_name:
        if np.any(getattr(model.param, var)[1] > 0):
            # print(type(getattr(param, var)[1]))
            if type(getattr(model.param, var)[1]) is np.ndarray:
                if var == 'lm':
                    size = getattr(model.param, var)[1].shape
                    setattr(model, var, Parameter(
                        torch.tensor(getattr(model.param, var)[0] - 1 * np.ones((size[0], size[1])),
                                     dtype=torch.float32)))
                    param_reg.append(getattr(model, var))
                    print(getattr(model, var))
                else:
                    size = getattr(model.param, var)[1].shape
                    setattr(model, var, Parameter(
                        torch.tensor(
                            getattr(model.param, var)[0] + getattr(model.param, var)[1] * np.random.randn(size[0], size[1]),
                            dtype=torch.float32)))
                    param_reg.append(getattr(model, var))
                    # print(getattr(self, var))
            else:
                setattr(model, var, Parameter(
                    torch.tensor(getattr(model.param, var)[0] + getattr(model.param, var)[1] * np.random.randn(1, )[0],
                                 dtype=torch.float32)))
                param_reg.append(getattr(model, var))
            if var != 'std_in':
                dict_nv = {'m': getattr(model.param, var)[0], 'v': 1 / (getattr(model.param, var)[1]) ** 2}

                dict_np = {'m': var + '_m', 'v': var + '_v_inv'}

                for key in dict_nv:
                    setattr(model, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
                    param_hyper.append(getattr(model, dict_np[key]))
        else:
            setattr(model, var, torch.tensor(getattr(model.param, var)[0], dtype=torch.float32))
    model.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}


def integration_forward(model, external, hx, hE):

    # define some constants
    conduct_lb = 1.5  # lower bound for conduct velocity
    u_2ndsys_ub = 500  # the bound of the input for second order system
    noise_std_lb = 150  # lower bound of std of noise
    lb = 0.01  # lower bound of local gains
    s2o_coef = 0.0001  # coefficient from states (source EEG) to EEG
    k_lb = 0.5  # lower bound of coefficient of external inputs

    next_state = {}

    M = hx[:, 0:1]  # current of main population
    E = hx[:, 1:2]  # current of excitory population
    I = hx[:, 2:3]  # current of inhibitory population

    Mv = hx[:, 3:4]  # voltage of main population
    Ev = hx[:, 4:5]  # voltage of exictory population
    Iv = hx[:, 5:6]  # voltage of inhibitory population

    dt = model.step_size
    # Generate the ReLU module for model parameters gEE gEI and gIE

    m = torch.nn.ReLU()

    # define constant 1 tensor
    con_1 = torch.tensor(1.0, dtype=torch.float32)
    if model.sc.shape[0] > 1:

        # Update the Laplacian based on the updated connection gains w_bb.
        w_b = torch.exp(model.w_bb) * torch.tensor(model.sc, dtype=torch.float32)
        w_n_b = w_b / torch.linalg.norm(w_b)

        model.sc_m_b = w_n_b
        dg_b = -torch.diag(torch.sum(w_n_b, dim=1))
        # Update the Laplacian based on the updated connection gains w_bb.
        w_f = torch.exp(model.w_ff) * torch.tensor(model.sc, dtype=torch.float32)
        w_n_f = w_f / torch.linalg.norm(w_f)

        model.sc_m_f = w_n_f
        dg_f = -torch.diag(torch.sum(w_n_f, dim=1))
        # Update the Laplacian based on the updated connection gains w_bb.
        w = torch.exp(model.w_ll) * torch.tensor(model.sc, dtype=torch.float32)
        w_n_l = (0.5 * (w + torch.transpose(w, 0, 1))) / torch.linalg.norm(
            0.5 * (w + torch.transpose(w, 0, 1)))

        model.sc_fitted = w_n_l
        dg_l = -torch.diag(torch.sum(w_n_l, dim=1))
    else:
        l_s = torch.tensor(np.zeros((1, 1)), dtype=torch.float32)
        dg_l = 0
        dg_b = 0
        dg_f = 0
        w_n_l = 0
        w_n_b = 0
        w_n_f = 0

    model.delays = (model.dist / (conduct_lb * con_1 + m(model.mu))).type(torch.int64)
    # print(torch.max(model.delays), model.delays.shape)

    # placeholder for the updated current state
    current_state = torch.zeros_like(hx)

    # placeholders for output BOLD, history of E I x f v and q
    eeg_window = []
    E_window = []
    I_window = []
    M_window = []
    Ev_window = []
    Iv_window = []
    Mv_window = []

    # Use the forward model to get EEG signal at ith element in the window.
    for i_window in range(model.TRs_per_window):

        for step_i in range(model.steps_per_TR):
            Ed = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)  # delayed E

            """for ind in range(model.node_size):
                #print(ind, hE[ind,:].shape, model.delays[ind,:].shape)
                Ed[ind] = torch.index_select(hE[ind,:], 0, model.delays[ind,:])"""
            hE_new = hE.clone()
            Ed = hE_new.gather(1, model.delays)  # delayed E

            LEd_b = torch.reshape(torch.sum(w_n_b * torch.transpose(Ed, 0, 1), 1),
                                  (model.node_size, 1))  # weights on delayed E
            LEd_f = torch.reshape(torch.sum(w_n_f * torch.transpose(Ed, 0, 1), 1),
                                  (model.node_size, 1))  # weights on delayed E
            LEd_l = torch.reshape(torch.sum(w_n_l * torch.transpose(Ed, 0, 1), 1),
                                  (model.node_size, 1))  # weights on delayed E
            # Input noise for M.

            u_tms = external[:, step_i:step_i + 1, i_window]
            #u_aud = external[:, i_hidden:i_hidden + 1, i_window, 1]
            #u_0 = external[:, i_hidden:i_hidden + 1, i_window, 2]

            # LEd+torch.matmul(dg,E): Laplacian on delayed E

            rM = (k_lb * con_1 + m(model.k)) * model.ki * u_tms + \
                 (0.1 * con_1 + m(model.std_in)) * torch.randn(model.node_size, 1) + \
                 1 * (lb * con_1 + m(model.g)) * (
                         LEd_l + 1 * torch.matmul(dg_l, M)) + \
                 sigmoid(E - I, model.vmax, model.v0, model.r)  # firing rate for Main population
            rE = (.1 * con_1 + m(model.std_in)) * torch.randn(model.node_size, 1) + \
                 1 * (lb * con_1 + m(model.g_f)) * (LEd_f + 1 * torch.matmul(dg_f, E - I)) + \
                 (lb * con_1 + m(model.c2)) * sigmoid((lb * con_1 + m(model.c1)) * M, model.vmax, model.v0,
                                                      model.r)  # firing rate for Excitory population
            rI = (0.1 * con_1 + m(model.std_in)) * torch.randn(model.node_size, 1) + \
                 1 * (lb * con_1 + m(model.g_b)) * (-LEd_b - 1 * torch.matmul(dg_b, E - I)) + \
                 (lb * con_1 + m(model.c4)) * sigmoid((lb * con_1 + m(model.c3)) * M, model.vmax, model.v0,
                                                      model.r)  # firing rate for Inhibitory population

            # Update the states by step-size.
            ddM = M + dt * Mv
            ddE = E + dt * Ev
            ddI = I + dt * Iv
            ddMv = Mv + dt * sys2nd(0 * con_1 + m(model.A), 1 * con_1 +
                                    m(model.a),
                                    u_2ndsys_ub * torch.tanh(rM / u_2ndsys_ub), M, Mv)

            ddEv = Ev + dt * sys2nd(0 * con_1 + m(model.A), 1 * con_1 +
                                    m(model.a),
                                    u_2ndsys_ub * torch.tanh(rE / u_2ndsys_ub), E, Ev)

            ddIv = Iv + dt * sys2nd(0 * con_1 + m(model.B), 1 * con_1 + m(model.b),
                                    u_2ndsys_ub * torch.tanh(rI / u_2ndsys_ub), I, Iv)

            # Calculate the saturation for model states (for stability and gradient calculation).
            E = ddE  # 1000*torch.tanh(ddE/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddE))
            I = ddI  # 1000*torch.tanh(ddI/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddI))
            M = ddM  # 1000*torch.tanh(ddM/1000)
            Ev = ddEv  # 1000*torch.tanh(ddEv/1000)#(con_1 + torch.tanh(df - con_1))
            Iv = ddIv  # 1000*torch.tanh(ddIv/1000)#(con_1 + torch.tanh(dv - con_1))
            Mv = ddMv  # 1000*torch.tanh(ddMv/1000)#(con_1 + torch.tanh(dq - con_1))

            # update placeholders for E buffer
            hE[:, 0] = M[:, 0]
            # hE = torch.cat([M, hE[:, :-1]], axis=1)

        # Put M E I Mv Ev and Iv at every tr to the placeholders for checking them visually.
        M_window.append(M)
        I_window.append(I)
        E_window.append(E)
        Mv_window.append(Mv)
        Iv_window.append(Iv)
        Ev_window.append(Ev)
        hE = torch.cat([M, hE[:, :-1]], dim=1)  # update placeholders for E buffer

        # Put the EEG signal each tr to the placeholder being used in the cost calculation.
        lm_t = (model.lm.T / torch.sqrt(model.lm ** 2).sum(1)).T

        model.lm_t = (lm_t - 1 / model.output_size * torch.matmul(torch.ones((1, model.output_size)),
                                                                  lm_t))  # s2o_coef *
        temp = model.cy0 * torch.matmul(model.lm_t, M[:200, :]) - 1 * model.y0
        eeg_window.append(temp)  # torch.abs(E) - torch.abs(I) + 0.0*noiseEEG)

    # Update the current state.
    current_state = torch.cat([M, E, I, Mv, Ev, Iv], dim=1)
    next_state['current_state'] = current_state
    next_state['eeg_window'] = torch.cat(eeg_window, dim=1)
    next_state['E_window'] = torch.cat(E_window, dim=1)
    next_state['I_window'] = torch.cat(I_window, dim=1)
    next_state['P_window'] = torch.cat(M_window, dim=1)
    next_state['Ev_window'] = torch.cat(Ev_window, dim=1)
    next_state['Iv_window'] = torch.cat(Iv_window, dim=1)
    next_state['Pv_window'] = torch.cat(Mv_window, dim=1)

    return next_state, hE