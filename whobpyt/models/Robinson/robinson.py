"""
Authors: Zheng Wang, John Griffiths, Davide Momi, Kevin Kadak, Parsa Oveisi, Taha Morshedzadeh, Sorenza Bastiaens
Neural Mass Model fitting
module for Robinson with forward backward and lateral connection for EEG
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

class ParamsCT(AbstractParams):

    def __init__(self, **kwargs):

        param = {
            "Q_max": [250, 0], 
            "sig_theta": [15/1000, 0.], 
            "sigma": [3.3/1000, 0], 
            "gamma": [100, 0], 
            "beta": [200, 0],
            "alpha": [200/4, 0.], 
            "t0": [0.08, 0.],
            "g": [100, 0], 
            "nu_ee ": [0.0528/1000, 0.], 
            "nu_ii": [0.0528/1000, 0.],
            "nu_ie": [0.02/1000, 0], 
            "nu_es": [1.2/1000, 0], 
            "nu_is": [1.2/1000, 0], 
            "nu_se": [1.2/1000, 0], 
            "nu_si": [0.0, 0], 
            "nu_ei": [0.4/1000, 0], 
            "nu_sr": [0.01/1000, 0], 
            "nu_sn": [0.0, 0], 
            "nu_re": [0.1/1000, 0], 
            "nu_ri": [0.0, 0], 
            "nu_rs": [0.1/1000, 0], 
            "nu_ss": [0.0, 0], 
            "nu_rr": [0.0, 0], 
            "nu_rn": [0.0, 0], 
            "mu": [5, 0], 
            "cy0": [5, 0], 
            "y0": [2, 0]
        }
        
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])


class RNNROBINSON(AbstractNMM):
    """
    A module for forward model (Robinson) to simulate a batch of EEG signals
    Attibutes
    ---------
    state_size : int
        the number of states in the Robinson model
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
    g: float
        global gain parameter
    w_bb: tensor with node_size x node_size (grad on depends on fit_gains)
        connection gains
    std_in std_out: tensor with gradient on
        std for state noise and output noise
    hyper parameters for prior distribution of model parameters
    Methods
    -------
    forward(input, noise_out, hx)
        forward model (Robinson) for generating a number of EEG signals with current model parameters
    """
    state_names = ['V_e', 'V_e_dot', 'phi_e', 'phi_e_dot', 'V_i', 'V_i_dot', 'phi_i', 'phi_i_dot']
    model_name = "CT"

    def __init__(self, node_size: int,
                 TRs_per_window: int, step_size: float, output_size: int, tr: float, sc: float, lm: float, dist: float,
                 use_fit_gains: bool, use_fit_lfm: bool, param: ParamsCT) -> None:
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
        param from ParamCT
        """
        super(RNNROBINSON, self).__init__()
        self.state_size = 8  # 8 states CT model
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
        return {"state_names": ['V_e', 'V_e_dot', 'phi_e', 'phi_e_dot', 'V_i', 'V_i_dot', 'phi_i', 'phi_i_dot'], "output_name": "eeg"}
    
    def createIC(self, ver):
        # initial state
        if (ver == 0):
            state_lb = 0.5
            state_ub = 2
            return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size+1, self.state_size)),
                             dtype=torch.float32)
        if (ver == 1):
            state_lb = 0
            state_ub = 5
            return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size+1, self.state_size)),
                             dtype=torch.float32)
        if (ver == 2): # for testing the robinson corticothalamic model
            state_lb = -1.5*1e-4
            state_ub = 1.5*1e-4
            return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size+1, self.state_size)),
                             dtype=torch.float32)        
    
    def setModelParameters(self):
        # set states E I f v mean and 1/sqrt(variance)
        return setModelParameters(self)

    def forward(self, external, hx, hE):
        return integration_forward(self, external, hx, hE)

def sigmoid(x, Q_max, sig_theta, sigma):
    return Q_max / (1 + torch.exp(-(x-sig_theta) / sigma))

    
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
                        torch.tensor(getattr(model.param, var)[0] + 1 * np.ones((size[0], size[1])),
                                     dtype=torch.float32)))
                    param_reg.append(getattr(model, var))
                    # print(getattr(model, var))
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

    V_e = hx[:model.node_size, 0:1]  # voltage of cortical excitatory population 
    V_e_dot = hx[:model.node_size, 1:2] # current of cortical excitatory population
    phi_e = hx[:model.node_size, 2:3]  # firing rate of excitory population
    phi_e_dot = hx[:model.node_size, 3:4] # change in firing rate of excitory population
    V_i = hx[:model.node_size, 4:5]  # voltage of cortical inhibitory population
    V_i_dot = hx[:model.node_size, 5:6] # current of cortical inhibitory population
    phi_i = hx[:model.node_size, 6:7] # firing rate of inhibitory population
    phi_i_dot = hx[:model.node_size, 7:8] # change in firing rate of inhibitory population
    
    V_s = hx[model.node_size:model.node_size+1, 0:1]  # voltage of cortical excitatory population 
    V_s_dot = hx[model.node_size:model.node_size+1, 1:2] # current of cortical excitatory population
    phi_s = hx[model.node_size:model.node_size+1, 2:3]  # firing rate of excitory population
    phi_s_dot = hx[model.node_size:model.node_size+1, 3:4] # change in firing rate of excitory population
    V_r = hx[model.node_size:model.node_size+1, 4:5]  # voltage of cortical inhibitory population
    V_r_dot = hx[model.node_size:model.node_size+1, 5:6] # current of cortical inhibitory population
    phi_r = hx[model.node_size:model.node_size+1, 6:7] # firing rate of inhibitory population
    phi_r_dot = hx[model.node_size:model.node_size+1, 7:8] # change in firing rate of inhibitory population
    
    dt = model.step_size
    # Generate the ReLU module for model parameters gEE gEI and gIE

    m = torch.nn.ReLU()

    # define constant 1 tensor
    con_1 = torch.tensor(1.0, dtype=torch.float32)
    if model.sc.shape[0] > 1:
        # Update the Laplacian based on the updated connection gains w_bb.
        w = torch.exp(model.w_ll) * torch.tensor(model.sc, dtype=torch.float32)
        w_n_l = (0.5 * (w + torch.transpose(w, 0, 1))) / torch.linalg.norm(
            0.5 * (w + torch.transpose(w, 0, 1)))

        model.sc_fitted = w_n_l
        dg_l = -torch.diag(torch.sum(w_n_l, dim=1))
    else:
        l_s = torch.tensor(np.zeros((1, 1)), dtype=torch.float32)
        dg_l = 0
        
        w_n_l = 0
        

    model.delays = (model.dist / (conduct_lb * con_1 + m(model.mu))).type(torch.int64)
    # print(torch.max(model.delays), model.delays.shape)

    # placeholder for the updated current state
    current_state = torch.zeros_like(hx)

    # placeholders for output BOLD, history of E I x f v and q
    eeg_window = []
    V_e_window = []
    V_e_dot_window = []
    phi_e_window = []
    phi_e_dot_window = []
    V_i_window = []
    V_i_dot_window = []
    phi_i_window = []
    phi_i_dot_window = []

    # Use the forward model to get EEG signal at ith element in the window.
    for i_window in range(model.TRs_per_window):

        for step_i in range(model.steps_per_TR):
            Ed = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)  # delayed E

            """for ind in range(model.node_size):
                #print(ind, hE[ind,:].shape, model.delays[ind,:].shape)
                Ed[ind] = torch.index_select(hE[ind,:], 0, model.delays[ind,:])"""
            hE_new = hE.clone()
            Ed = hE_new.gather(1, model.delays)  # delayed E

            
            LEd_l = torch.reshape(torch.sum(w_n_l * torch.transpose(Ed, 0, 1), 1),
                                  (model.node_size, 1))  # weights on delayed E
            # Input noise for M.

            u_tms = external[:, step_i:step_i + 1, i_window]
            #u_aud = external[:, i_hidden:i_hidden + 1, i_window, 1]
            #u_0 = external[:, i_hidden:i_hidden + 1, i_window, 2]

            # LEd+torch.matmul(dg,E): Laplacian on delayed E
        
            alpha = model.alpha
            beta = model.beta
            alphaxbeta = model.alpha*model.beta
            gamma = model.gamma
            gamma_rs = model.gamma*3
            nu_ee = model.nu_ee
            nu_ei = model.nu_ei
            nu_es = model.nu_es
            nu_ie = model.nu_ie
            nu_ii = model.nu_ii
            nu_is = model.nu_is
            nu_se = model.nu_se
            nu_si = model.nu_si
            nu_ss = model.nu_ss
            nu_sr = model.nu_sr
            nu_sn = model.nu_sn
            nu_re = model.nu_re
            nu_ri = model.nu_ri
            nu_rs = model.nu_rs
            Q = model.Q_max
            sig_theta = model.sig_theta
            sigma = model.sigma
            
            # Update the states by step-size.
            ddVe = V_e + dt * V_e_dot
            ddphie = phi_e + dt * phi_e_dot
            ddVi = V_i + dt * V_i_dot 
            ddphii = phi_i + dt * phi_i_dot
            ddVs = V_s + dt * V_s_dot 
            ddphis = phi_s + dt * phi_s_dot
            ddVr = V_r + dt * V_r_dot 
            ddphir = phi_r + dt * phi_r_dot

            ones_mx = torch.ones((1,1)) # 1x1 ones matrix
            noise_phi_e = torch.randn_like(phi_e) * 1 # noise for phi_e, stdev = 1, to be added to ddphiedot

            ddVedot = V_e_dot + dt *  (-(1/alpha + 1/beta) * alphaxbeta * V_e_dot -alphaxbeta*V_e + \
                                   alphaxbeta*(m(nu_ee) * phi_e - m(nu_ei) * phi_i + m(nu_es) * phi_s) +u_tms)
            ddVidot = V_i_dot + dt * (-(1/alpha + 1/beta) * alphaxbeta * V_i_dot -alphaxbeta*V_i + \
                                   alphaxbeta*(m(nu_ie) * phi_e + m(nu_ii) * phi_i + m(nu_is) * phi_s))
            ddVsdot = V_s_dot + dt * (-(1/alpha + 1/beta) * alphaxbeta * V_s_dot - alphaxbeta * V_s + \
                                   alphaxbeta*(m(nu_se) * torch.mean(phi_e, axis=0)*ones_mx + m(nu_si) * torch.mean(phi_i, axis=0)*ones_mx + \
                                               m(nu_ss) * phi_s - m(nu_sr) * phi_r + m(nu_sn) * (0.025)*ones_mx + \
                                               0.001 * torch.randn(1, 1)))
            ddVrdot = V_r_dot + dt * (-(1/alpha + 1/beta) * alphaxbeta * V_r_dot -alphaxbeta*V_r + \
                                   alphaxbeta*(m(nu_re) * torch.mean(phi_e, axis=0) * ones_mx + \
                                               m(nu_ri) * torch.mean(phi_i, axis=0) * ones_mx + \
                                               m(nu_rs) * phi_s))
            network_interactions = (lb * con_1 + m(model.g)) * LEd_l + \
                                    1 * torch.matmul(dg_l, phi_e) # is implementation of global gain & SC 
            ddphiedot = phi_e_dot + dt * (-2 * gamma * phi_e_dot - gamma**2 * phi_e + \
                                       gamma**2 * sigmoid(V_e, Q, sig_theta, sigma) + network_interactions + noise_phi_e)
            ddphiidot = phi_i_dot + dt * (-2 * gamma_rs * phi_i_dot - gamma_rs**2 * phi_i + \
                                       gamma_rs**2 * sigmoid(V_i, Q, sig_theta, sigma))
            ddphisdot = phi_s_dot + dt * (-2 * gamma_rs * phi_s_dot - gamma_rs**2 * phi_s + \
                                       gamma_rs**2 * sigmoid(V_s, Q, sig_theta, sigma))
            ddphirdot = phi_r_dot + dt * (-2 * gamma_rs * phi_r_dot - gamma_rs**2 * phi_r + \
                                       gamma_rs**2 * sigmoid(V_r, Q, sig_theta, sigma))


            # Calculate the saturation for model states (for stability and gradient calculation).
            V_e = ddVe  # 1000*torch.tanh(ddE/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddE))
            V_i = ddVi  # 1000*torch.tanh(ddI/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddI))
            V_s = ddVs  # 1000*torch.tanh(ddM/1000)
            V_r = ddVr
            phi_e = ddphie
            phi_i = ddphii
            phi_s = ddphis
            phi_r = ddphir
            
            V_e_dot = ddVedot  # 1000*torch.tanh(ddEv/1000)#(con_1 + torch.tanh(df - con_1))
            V_i_dot = ddVidot   # 1000*torch.tanh(ddIv/1000)#(con_1 + torch.tanh(dv - con_1))
            V_s_dot = ddVsdot   # 1000*torch.tanh(ddMv/1000)#(con_1 + torch.tanh(dq - con_1))
            V_r_dot = ddVrdot 
            phi_e_dot = ddphiedot 
            phi_i_dot = ddphiidot 
            phi_s_dot = ddphisdot
            phi_r_dot = ddphirdot
            
            # update placeholders for E buffer
            hE[:, 0] = phi_e[:, 0]
            # hE = torch.cat([M, hE[:, :-1]], axis=1)

        # Put M E I Mv Ev and Iv at every tr to the placeholders for checking them visually.
        V_e_window.append(torch.cat([V_e, V_s], axis = 0))
        V_i_window.append(torch.cat([V_i, V_r], axis = 0))
        phi_e_window.append(torch.cat([phi_e, phi_s], axis = 0))
        phi_i_window.append(torch.cat([phi_i, phi_r], axis = 0))
        V_e_dot_window.append(torch.cat([V_e_dot, V_s_dot], axis = 0))
        V_i_dot_window.append(torch.cat([V_i_dot, V_r_dot], axis = 0))
        phi_e_dot_window.append(torch.cat([phi_e_dot, phi_s_dot], axis = 0))
        phi_i_dot_window.append(torch.cat([phi_i_dot, phi_r_dot], axis = 0))                     
        hE = torch.cat([phi_e, hE[:, :-1]], dim=1)  # update placeholders for E buffer

        # Put the EEG signal each tr to the placeholder being used in the cost calculation.
        lm_t = (model.lm.T / torch.sqrt(model.lm ** 2).sum(1)).T

        model.lm_t = (lm_t - 1 / model.output_size * torch.matmul(torch.ones((1, model.output_size)),
                                                                  lm_t))  # s2o_coef *
        temp = model.cy0 * torch.matmul(lm_t, phi_e[:model.node_size, :]) - 1 * model.y0
        eeg_window.append(temp)  # torch.abs(E) - torch.abs(I) + 0.0*noiseEEG)

    # Update the current state.
    current_state = torch.cat([torch.cat([V_e, V_s], axis = 0), \
                               torch.cat([V_e_dot, V_s_dot], axis = 0), \
                               torch.cat([phi_e, phi_s], axis = 0), \
                               torch.cat([phi_e_dot, phi_s_dot], axis = 0), \
                               torch.cat([V_i, V_r], axis = 0), \
                               torch.cat([V_i_dot, V_r_dot], axis = 0), \
                               torch.cat([phi_i, phi_r], axis = 0), \
                               torch.cat([phi_i_dot, phi_r_dot], axis = 0)], dim=1)
    next_state['current_state'] = current_state
    next_state['eeg_window'] = torch.cat(eeg_window, dim=1)
    next_state['V_e_window'] = torch.cat(V_e_window, dim=1)
    next_state['V_i_window'] = torch.cat(V_i_window, dim=1)
    next_state['phi_e_window'] = torch.cat(phi_e_window, dim=1)
    next_state['phi_i_window'] = torch.cat(phi_i_window, dim=1)
    next_state['V_e_dot_window'] = torch.cat(V_e_dot_window, dim=1)
    next_state['V_i_dot_window'] = torch.cat(V_i_dot_window, dim=1)
    next_state['phi_e_dot_window'] = torch.cat(phi_e_dot_window, dim=1)                            
    next_state['phi_i_dot_window'] = torch.cat(phi_i_dot_window, dim=1)

    return next_state, hE