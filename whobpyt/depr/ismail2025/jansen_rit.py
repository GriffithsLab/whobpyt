import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class ParamsModel:

    def __init__(self, model_name, **kwargs):
        param = {'g': [100, 0]}

        if model_name == 'RWW':
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
        elif model_name == "JR":
            param = {
                "A ": [3.25, 0], "a": [100, 0.], "B": [22, 0], "b": [50, 0], "g": [1000, 0],
                "c1": [135, 0.], "c2": [135 * 0.8, 0.], "c3 ": [135 * 0.25, 0.], "c4": [135 * 0.25, 0.],
                "std_in": [100, 0], "vmax": [5, 0], "v0": [6, 0], "r": [0.56, 0], "y0": [2, 0],
                "mu": [.5, 0], "k": [5, 0], "cy0": [5, 0], "ki": [1, 0]
            }
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])


class OutputNM:
    mode_all = ['train', 'test']
    stat_vars_all = ['m', 'v_inv']
    output_name = ['bold']

    def __init__(self, model_name, param, fit_weights=False, fit_lfm=False):
        state_names = ['E']
        self.loss = np.array([])
        if model_name == 'RWW':
            state_names = ['E', 'I', 'x', 'f', 'v', 'q']
            self.output_name = "bold"
        elif model_name == "JR":
            state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
            self.output_name = "eeg"
        elif model_name == 'LIN':
            state_names = ['E']
            self.output_name = "bold"

        for name in state_names + [self.output_name]:
            for m in self.mode_all:
                setattr(self, name + '_' + m, [])

        vars_name = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars_name:
            if np.any(getattr(param, var)[1] > 0):
                if var != 'std_in':
                    setattr(self, var, np.array([]))
                    for stat_var in self.stat_vars_all:
                        setattr(self, var + '_' + stat_var, [])
                else:
                    setattr(self, var, [])
        if fit_weights:
            self.weights = []

        if model_name == 'JR' and fit_lfm:
            self.leadfield = []

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def dataloader(emp, epoch_size, TRperwindow):
    window_size = int(emp.shape[0] / TRperwindow)
    data_out = 0
    if len(emp.shape) == 2:
        node_size = emp.shape[1]
        length_ts = emp.shape[0]
        window_size = int(length_ts / TRperwindow)
        data_out = np.zeros((epoch_size, window_size, node_size, TRperwindow))
        for i_epoch in range(epoch_size):
            for i_win in range(window_size):
                data_out[i_epoch, i_win, :, :] = emp.T[:, i_win * TRperwindow:(i_win + 1) * TRperwindow]
    if len(emp.shape) == 3:
        node_size = emp.shape[2]
        length_ts = emp.shape[1]
        data_size = emp.shape[0]
        window_size = int(length_ts / TRperwindow)
        data_out = np.zeros((epoch_size, window_size, node_size, TRperwindow))
        for i_epoch in range(epoch_size):
            for i_win in range(window_size):
                data_out[i_epoch, i_win, :, :] = \
                    emp[i_epoch % data_size, i_win * TRperwindow:(i_win + 1) * TRperwindow, :].T
    return data_out

def sys2nd(A, a, u, x, v):
    return A * a * u - 2 * a * v - a ** 2 * x


def sigmoid(x, vmax, v0, r):
    return vmax / (1 + torch.exp(r * (v0 - x)))


def h_tf(a, b, d, z):
    """
    Neuronal input-output functions of excitatory pools and inhibitory pools.
    Take the variables a, x, and b and convert them to a linear equation (a*x - b) while adding a small
    amount of noise 0.00001 while dividing that term to an exponential of the linear equation multiplied by the
    d constant for the appropriate dimensions.
    """
    num = 0.00001 + torch.abs(a * z - b)
    den = 0.00001 * d + torch.abs(1.0000 - torch.exp(-d * (a * z - b)))
    return torch.divide(num, den)



def setModelParameters(model):
    if model.model_name == 'RWW':
        if model.use_Gaussian_EI:
            model.E_m = Parameter(torch.tensor(0.16, dtype=torch.float32))
            model.I_m = Parameter(torch.tensor(0.1, dtype=torch.float32))
            # model.f_m = Parameter(torch.tensor(1.0, dtype=torch.float32))
            model.v_m = Parameter(torch.tensor(1.0, dtype=torch.float32))
            # model.x_m = Parameter(torch.tensor(0.16, dtype=torch.float32))
            model.q_m = Parameter(torch.tensor(1.0, dtype=torch.float32))

            model.E_v_inv = Parameter(torch.tensor(2500, dtype=torch.float32))
            model.I_v_inv = Parameter(torch.tensor(2500, dtype=torch.float32))
            # model.f_v = Parameter(torch.tensor(100, dtype=torch.float32))
            model.v_v_inv = Parameter(torch.tensor(100, dtype=torch.float32))
            # model.x_v = Parameter(torch.tensor(100, dtype=torch.float32))
            model.q_v_inv = Parameter(torch.tensor(100, dtype=torch.float32))

        # hyper parameters (variables: need to calculate gradient) to fit density
        # of gEI and gIE (the shape from the bifurcation analysis on an isolated node)
        if model.use_Bifurcation:
            model.sup_ca = Parameter(torch.tensor(0.5, dtype=torch.float32))
            model.sup_cb = Parameter(torch.tensor(20, dtype=torch.float32))
            model.sup_cc = Parameter(torch.tensor(10, dtype=torch.float32))

        # set gains_con as Parameter if fit_gain is True
        if model.use_fit_gains:
            model.gains_con = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                                     dtype=torch.float32))  # connenction gain to modify empirical sc
        else:
            model.gains_con = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)

        vars_name = [a for a in dir(model.param) if not a.startswith('__') and not callable(getattr(model.param, a))]
        for var in vars_name:
            if np.any(getattr(model.param, var)[1] > 0):
                setattr(model, var, Parameter(
                    torch.tensor(getattr(model.param, var)[0] + getattr(model.param, var)[1] * np.random.randn(1, )[0],
                                 dtype=torch.float32)))
                if model.use_Bifurcation:
                    if var not in ['std_in', 'g_IE', 'g_EI']:
                        dict_nv = {'m': getattr(model.param, var)[0], 'v': 1 / (getattr(model.param, var)[1]) ** 2}

                        dict_np = {'m': var + '_m', 'v': var + '_v_inv'}

                        for key in dict_nv:
                            setattr(model, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
                else:
                    if var not in ['std_in']:
                        dict_nv = {'m': getattr(model.param, var)[0], 'v': 1 / (getattr(model.param, var)[1]) ** 2}

                        dict_np = {'m': var + '_m', 'v': var + '_v_inv'}

                        for key in dict_nv:
                            setattr(model, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
            else:
                setattr(model, var, torch.tensor(getattr(model.param, var)[0], dtype=torch.float32))

    if model.model_name == 'JR':
        # set model parameters (variables: need to calculate gradient) as Parameter others : tensor
        # set w_bb as Parameter if fit_gain is True
        if model.use_fit_gains:
            model.w_bb = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                                dtype=torch.float32))  # connenction gain to modify empirical sc
            model.w_ff = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                                dtype=torch.float32))
            model.w_ll = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                                dtype=torch.float32))
        else:
            model.w_bb = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)
            model.w_ff = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)
            model.w_ll = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)

        if model.use_fit_lfm:
            model.lm = Parameter(torch.tensor(model.lm, dtype=torch.float32))  # leadfield matrix from sourced data to m/eeg
        else:
            model.lm = torch.tensor(model.lm, dtype=torch.float32)  # leadfield matrix from sourced data to m/eeg

        vars_name = [a for a in dir(model.param) if not a.startswith('__') and not callable(getattr(model.param, a))]
        for var in vars_name:
            if np.any(getattr(model.param, var)[1] > 0):
                # print(type(getattr(param, var)[1]))
                if type(getattr(model.param, var)[1]) is np.ndarray:
                    if var == 'lm':
                        size = getattr(model.param, var)[1].shape
                        setattr(model, var, Parameter(
                            torch.tensor(getattr(model.param, var)[0] -0*np.ones((size[0], size[1])),
                                dtype=torch.float32)))
                        print(getattr(model, var))
                    else:
                        size = getattr(model.param, var)[1].shape
                        setattr(model, var, Parameter(
                            torch.tensor(
                                getattr(model.param, var)[0] + getattr(model.param, var)[1] * np.random.randn(size[0], size[1]),
                                dtype=torch.float32)))
                        # print(getattr(self, var))
                else:
                    setattr(model, var, Parameter(
                        torch.tensor(getattr(model.param, var)[0] + getattr(model.param, var)[1] * np.random.randn(1, )[0],
                                     dtype=torch.float32)))
                if var != 'std_in':
                    dict_nv = {'m': getattr(model.param, var)[0], 'v': 1 / (getattr(model.param, var)[1]) ** 2}

                    dict_np = {'m': var + '_m', 'v': var + '_v_inv'}

                    for key in dict_nv:
                        setattr(model, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
            else:
                setattr(model, var, torch.tensor(getattr(model.param, var)[0], dtype=torch.float32))

    if model.model_name == 'LIN':
        # set gains_con as Parameter if fit_gain is True
        if model.use_fit_gains:
            model.gains_con = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                                     dtype=torch.float32))  # connenction gain to modify empirical sc
        else:
            model.gains_con = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)

        vars_name = [a for a in dir(model.param) if not a.startswith('__') and not callable(getattr(model.param, a))]
        for var in vars_name:
            if np.any(getattr(model.param, var)[1] > 0):
                setattr(model, var, Parameter(
                    torch.tensor(getattr(model.param, var)[0] + getattr(model.param, var)[1] * np.random.randn(1, )[0],
                                 dtype=torch.float32)))

                if var not in ['std_in']:
                    dict_nv = {'m': getattr(model.param, var)[0], 'v': 1 / (getattr(model.param, var)[1]) ** 2}

                    dict_np = {'m': var + '_m', 'v': var + '_v_inv'}

                    for key in dict_nv:
                        setattr(model, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
            else:
                setattr(model, var, torch.tensor(getattr(model.param, var)[0], dtype=torch.float32))


def integration_forward(model, external, hx, hE):
    if model.model_name == 'RWW':
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

        x = hx[:, 2:3]
        f = hx[:, 3:4]
        v = hx[:, 4:5]
        q = hx[:, 5:6]

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
        # E_window = torch.zeros((model.node_size,model.TRs_per_window))
        # I_window = torch.zeros((model.node_size,model.TRs_per_window))

        x_window = torch.zeros((model.node_size, model.TRs_per_window))
        f_window = torch.zeros((model.node_size, model.TRs_per_window))
        v_window = torch.zeros((model.node_size, model.TRs_per_window))
        q_window = torch.zeros((model.node_size, model.TRs_per_window))

        E_hist = torch.zeros((model.node_size, model.TRs_per_window, model.steps_per_TR))
        I_hist = torch.zeros((model.node_size, model.TRs_per_window, model.steps_per_TR))
        E_mean = hx[:, 0:1]
        I_mean = hx[:, 1:2]
        # print(E_m.shape)
        # Use the forward model to get neural activity at ith element in the window.
        if model.use_dynamic_boundary:
            for TR_i in range(model.TRs_per_window):

                # print(E.shape)

                # Since tr is about second we need to use a small step size like 0.05 to integrate the model states.
                for step_i in range(model.steps_per_TR):
                    E = torch.zeros((model.node_size, model.sampling_size))
                    I = torch.zeros((model.node_size, model.sampling_size))
                    for sample_i in range(model.sampling_size):
                        E[:, sample_i] = E_mean[:, 0] + 0.02 * torch.randn(model.node_size)
                        I[:, sample_i] = I_mean[:, 0] + 0.001 * torch.randn(model.node_size)

                    # Calculate the input recurrent.
                    IE = torch.tanh(m(model.W_E * model.I_0 + (0.001 + m(model.g_EE)) * E
                                      + model.g * torch.matmul(lap_adj, E) - (
                                              0.001 + m(model.g_IE)) * I))  # input currents for E
                    II = torch.tanh(m(model.W_I * model.I_0 + (0.001 + m(model.g_EI)) * E - I))  # input currents for I

                    # Calculate the firing rates.
                    rE = h_tf(model.aE, model.bE, model.dE, IE)  # firing rate for E
                    rI = h_tf(model.aI, model.bI, model.dI, II)  # firing rate for I
                    # Update the states by step-size 0.05.
                    E_next = E + dt * (-E * torch.reciprocal(model.tau_E) + model.gamma_E * (1. - E) * rE) \
                             + torch.sqrt(dt) * torch.randn(model.node_size, model.sampling_size) * (0.02 + m(
                        model.std_in))  ### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)
                    I_next = I + dt * (-I * torch.reciprocal(model.tau_I) + model.gamma_I * rI) \
                             + torch.sqrt(dt) * torch.randn(model.node_size, model.sampling_size) * (
                                     0.02 + m(model.std_in))

                    # Calculate the saturation for model states (for stability and gradient calculation).

                    # E_next[E_next>=0.9] = torch.tanh(1.6358*E_next[E_next>=0.9])
                    E = torch.tanh(0.0000 + m(1.0 * E_next))
                    I = torch.tanh(0.0000 + m(1.0 * I_next))

                    I_mean = I.mean(1)[:, np.newaxis]
                    E_mean = E.mean(1)[:, np.newaxis]
                    I_mean[I_mean < 0.00001] = 0.00001
                    E_mean[E_mean < 0.00001] = 0.00001

                    E_hist[:, TR_i, step_i] = E_mean[:, 0]
                    I_hist[:, TR_i, step_i] = I_mean[:, 0]

            for TR_i in range(model.TRs_per_window):

                for step_i in range(model.steps_per_TR):
                    x_next = x + 1 * dt * (1 * E_hist[:, TR_i, step_i][:, np.newaxis] - torch.reciprocal(
                        model.tau_s) * x - torch.reciprocal(model.tau_f) * (f - 1))
                    f_next = f + 1 * dt * x
                    v_next = v + 1 * dt * (f - torch.pow(v, torch.reciprocal(model.alpha))) * torch.reciprocal(
                        model.tau_0)
                    q_next = q + 1 * dt * (
                            f * (1 - torch.pow(1 - model.rho, torch.reciprocal(f))) * torch.reciprocal(
                        model.rho) - q * torch.pow(v, torch.reciprocal(model.alpha)) * torch.reciprocal(v)) \
                             * torch.reciprocal(model.tau_0)

                    x = torch.tanh(x_next)
                    f = (1 + torch.tanh(f_next - 1))
                    v = (1 + torch.tanh(v_next - 1))
                    q = (1 + torch.tanh(q_next - 1))
                    # Put x f v q from each tr to the placeholders for checking them visually.
                x_window[:, TR_i] = x[:, 0]
                f_window[:, TR_i] = f[:, 0]
                v_window[:, TR_i] = v[:, 0]
                q_window[:, TR_i] = q[:, 0]

                # Put the BOLD signal each tr to the placeholder being used in the cost calculation.

                bold_window[:, TR_i] = ((0.00 + m(model.std_out)) * torch.randn(model.node_size, 1) +
                                        100.0 * model.V * torch.reciprocal(model.E0) *
                                        (model.k1 * (1 - q) + model.k2 * (1 - q * torch.reciprocal(v)) + model.k3 * (
                                                1 - v)))[:, 0]
        else:

            for TR_i in range(model.TRs_per_window):

                # print(E.shape)

                # Since tr is about second we need to use a small step size like 0.05 to integrate the model states.
                for step_i in range(model.steps_per_TR):
                    E = torch.zeros((model.node_size, model.sampling_size))
                    I = torch.zeros((model.node_size, model.sampling_size))
                    for sample_i in range(model.sampling_size):
                        E[:, sample_i] = E_mean[:, 0] + 0.001 * torch.randn(model.node_size)
                        I[:, sample_i] = I_mean[:, 0] + 0.001 * torch.randn(model.node_size)

                    # Calculate the input recurrent.
                    IE = 1 * torch.tanh(m(model.W_E * model.I_0 + (0.001 + m(model.g_EE)) * E \
                                          + model.g * torch.matmul(lap_adj, E) - (
                                                  0.001 + m(model.g_IE)) * I))  # input currents for E
                    II = 1 * torch.tanh(
                        m(model.W_I * model.I_0 + (0.001 + m(model.g_EI)) * E - I))  # input currents for I

                    # Calculate the firing rates.
                    rE = h_tf(model.aE, model.bE, model.dE, IE)  # firing rate for E
                    rI = h_tf(model.aI, model.bI, model.dI, II)  # firing rate for I
                    # Update the states by step-size 0.05.
                    E_next = E + dt * (-E * torch.reciprocal(model.tau_E) + model.gamma_E * (1. - E) * rE) \
                             + torch.sqrt(dt) * torch.randn(model.node_size, model.sampling_size) * (0.02 + m(
                        model.std_in))  ### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)
                    I_next = I + dt * (-I * torch.reciprocal(model.tau_I) + model.gamma_I * rI) \
                             + torch.sqrt(dt) * torch.randn(model.node_size, model.sampling_size) * (
                                     0.02 + m(model.std_in))

                    # Calculate the saturation for model states (for stability and gradient calculation).
                    E_next[E_next < 0.00001] = 0.00001
                    I_next[I_next < 0.00001] = 0.00001
                    # E_next[E_next>=0.9] = torch.tanh(1.6358*E_next[E_next>=0.9])
                    E = E_next  # torch.tanh(0.00001+m(1.0*E_next))
                    I = I_next  # torch.tanh(0.00001+m(1.0*I_next))

                    I_mean = I.mean(1)[:, np.newaxis]
                    E_mean = E.mean(1)[:, np.newaxis]
                    E_hist[:, TR_i, step_i] = torch.tanh(E_mean)[:, 0]
                    I_hist[:, TR_i, step_i] = torch.tanh(I_mean)[:, 0]

                # E_window[:,TR_i]=E_mean[:,0]
                # I_window[:,TR_i]=I_mean[:,0]

            for TR_i in range(model.TRs_per_window):

                for step_i in range(model.steps_per_TR):
                    x_next = x + 1 * dt * (1 * E_hist[:, TR_i, step_i][:, np.newaxis] - torch.reciprocal(
                        model.tau_s) * x - torch.reciprocal(model.tau_f) * (f - 1))
                    f_next = f + 1 * dt * x
                    v_next = v + 1 * dt * (f - torch.pow(v, torch.reciprocal(model.alpha))) * torch.reciprocal(
                        model.tau_0)
                    q_next = q + 1 * dt * (
                            f * (1 - torch.pow(1 - model.rho, torch.reciprocal(f))) * torch.reciprocal(
                        model.rho) - q * torch.pow(v, torch.reciprocal(model.alpha)) * torch.reciprocal(v)) \
                             * torch.reciprocal(model.tau_0)

                    f_next[f_next < 0.001] = 0.001
                    v_next[v_next < 0.001] = 0.001
                    q_next[q_next < 0.001] = 0.001
                    x = x_next  # torch.tanh(x_next)
                    f = f_next  # (1 + torch.tanh(f_next - 1))
                    v = v_next  # (1 + torch.tanh(v_next - 1))
                    q = q_next  # (1 + torch.tanh(q_next - 1))
                # Put x f v q from each tr to the placeholders for checking them visually.
                x_window[:, TR_i] = x[:, 0]
                f_window[:, TR_i] = f[:, 0]
                v_window[:, TR_i] = v[:, 0]
                q_window[:, TR_i] = q[:, 0]
                # Put the BOLD signal each tr to the placeholder being used in the cost calculation.

                bold_window[:, TR_i] = ((0.00 + m(model.std_out)) * torch.randn(model.node_size, 1) +
                                        100.0 * model.V * torch.reciprocal(
                            model.E0) * (model.k1 * (1 - q) + model.k2 * (
                                1 - q * torch.reciprocal(v)) + model.k3 * (1 - v)))[:, 0]

        # Update the current state.
        # print(E_m.shape)
        current_state = torch.cat([E_mean, I_mean, x, f, v, q], dim=1)
        next_state['current_state'] = current_state
        next_state['bold_window'] = bold_window
        next_state['E_window'] = E_hist.reshape((model.node_size, -1))
        next_state['I_window'] = I_hist.reshape((model.node_size, -1))
        next_state['x_window'] = x_window
        next_state['f_window'] = f_window
        next_state['v_window'] = v_window
        next_state['q_window'] = q_window

        return next_state, hE

    if model.model_name == 'JR':

        # define some constants
        conduct_lb = 1.5  # lower bound for conduct velocity
        u_2ndsys_ub = 500  # the bound of the input for second order system
        noise_std_lb = 150  # lower bound of std of noise
        lb = 0.01  # lower bound of local gains
        s2o_coef = 0.0001  # coefficient from states (source M/EEG) to M/EEG
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

        # Use the forward model to get M/EEG signal at ith element in the window.
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

                rM =(k_lb * con_1 + m(model.k)) * m(model.ki)* u_tms + \
                     (5 * con_1 + torch.exp(model.std_in)) * torch.randn(model.node_size, 1) + \
                     1 * (lb * con_1 + m(model.g)) * (
                             LEd_l + 1 * torch.matmul(dg_l, M)) + \
                     sigmoid(E - I, model.vmax, model.v0, model.r)  # firing rate for Main population
                rE =  (0.0+m(model.kE))+ (5 * con_1 + torch.exp(model.std_in)) * torch.randn(model.node_size, 1) + \
                     1 * (lb * con_1 + m(model.g_f)) * (LEd_f + 1 * torch.matmul(dg_f, E - I)) + \
                     (lb * con_1 + m(model.c2)) * sigmoid((lb * con_1 + m(model.c1)) * M, model.vmax, model.v0,
                                                          model.r)  # firing rate for Excitory population
                rI = (0.0+m(model.kI))+(5* con_1 + torch.exp(model.std_in)) * torch.randn(model.node_size, 1) + \
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
                E = 1000*torch.tanh(ddE/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddE))
                I = 1000*torch.tanh(ddI/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddI))
                M = 1000*torch.tanh(ddM/1000)
                Ev = 1000*torch.tanh(ddEv/1000)#(con_1 + torch.tanh(df - con_1))
                Iv = 1000*torch.tanh(ddIv/1000)#(con_1 + torch.tanh(dv - con_1))
                Mv = 1000*torch.tanh(ddMv/1000)#(con_1 + torch.tanh(dq - con_1))

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

            # Put the M/EEG signal each tr to the placeholder being used in the cost calculation.
            lm_t = (model.lm.T / torch.sqrt(model.lm ** 2).sum(1)).T

            model.lm_t = (lm_t - 1 / model.output_size * torch.matmul(torch.ones((1, model.output_size)),
                                                                      lm_t))  # s2o_coef *
            temp = model.cy0 * torch.matmul(model.lm_t, E-I) - 1 * model.y0
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

    if model.model_name == 'LIN':
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

class RNNJANSEN(torch.nn.Module):
    """
    A module for forward model (JansenRit) to simulate a batch of M/EEG signals
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
                 use_fit_gains: bool, use_fit_lfm: bool, param: ParamsModel) -> None:
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
        self.output_size = output_size  # num of M/EEG channels
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.dist = torch.tensor(dist, dtype=torch.float32)
        self.lm = lm
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        self.use_fit_lfm = use_fit_lfm
        self.param = param

        self.output_size = lm.shape[0]  # number of M/EEG channels

    def setModelParameters(self):
        # set states E I f v mean and 1/sqrt(variance)
        return setModelParameters(self)

    def forward(self, external, hx, hE):
        return integration_forward(self, external, hx, hE)

class Costs:
    def __init__(self, method):
        self.method = method

    def cost_dist(self, sim, emp):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, the probability and negative log-likelihood.
        Parameters
        ----------
        sim: tensor with node_size X datapoint
            simulated EEG
        emp: tensor with node_size X datapoint
            empirical EEG
        """

        losses = torch.sqrt(torch.mean((sim - emp) ** 2))  #
        return losses

    def cost_r(self, logits_series_tf, labels_series_tf):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, the probability and negative log-likelihood.
        Parameters
        ----------
        logits_series_tf: tensor with node_size X datapoint
            simulated BOLD
        labels_series_tf: tensor with node_size X datapoint
            empirical BOLD
        """
        # get node_size() and TRs_per_window()
        node_size = logits_series_tf.shape[0]
        truncated_backprop_length = logits_series_tf.shape[1]

        # remove mean across time
        labels_series_tf_n = labels_series_tf - torch.reshape(torch.mean(labels_series_tf, 1),
                                                              [node_size, 1])  # - torch.matmul(

        logits_series_tf_n = logits_series_tf - torch.reshape(torch.mean(logits_series_tf, 1),
                                                              [node_size, 1])  # - torch.matmul(

        # correlation
        cov_sim = torch.matmul(logits_series_tf_n, torch.transpose(logits_series_tf_n, 0, 1))
        cov_def = torch.matmul(labels_series_tf_n, torch.transpose(labels_series_tf_n, 0, 1))

        # fc for sim and empirical BOLDs
        FC_sim_T = torch.matmul(torch.matmul(torch.diag(torch.reciprocal(torch.sqrt(
            torch.diag(cov_sim)))), cov_sim),
            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_sim)))))
        FC_T = torch.matmul(torch.matmul(torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_def)))), cov_def),
                            torch.diag(torch.reciprocal(torch.sqrt(torch.diag(cov_def)))))

        # mask for lower triangle without diagonal
        ones_tri = torch.tril(torch.ones_like(FC_T), -1)
        zeros = torch.zeros_like(FC_T)  # create a tensor all ones
        mask = torch.greater(ones_tri, zeros)  # boolean tensor, mask[i] = True iff x[i] > 1

        # mask out fc to vector with elements of the lower triangle
        FC_tri_v = torch.masked_select(FC_T, mask)
        FC_sim_tri_v = torch.masked_select(FC_sim_T, mask)

        # remove the mean across the elements
        FC_v = FC_tri_v - torch.mean(FC_tri_v)
        FC_sim_v = FC_sim_tri_v - torch.mean(FC_sim_tri_v)

        # corr_coef
        corr_FC = torch.sum(torch.multiply(FC_v, FC_sim_v)) \
                  * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_v, FC_v)))) \
                  * torch.reciprocal(torch.sqrt(torch.sum(torch.multiply(FC_sim_v, FC_sim_v))))

        # use surprise: corr to calculate probability and -log
        losses_corr = -torch.log(0.5000 + 0.5 * corr_FC)  # torch.mean((FC_v -FC_sim_v)**2)#
        return losses_corr

    def cost_eff(self, sim, emp, model: torch.nn.Module, next_window):
        # define some constants
        lb = 0.001

        w_cost = 10

        # define the relu function
        m = torch.nn.ReLU()

        exclude_param = []
        if model.use_fit_gains:
            exclude_param.append('gains_con')

        if model.model_name == "JR" and model.use_fit_lfm:
            exclude_param.append('lm')

        if self.method == 0:
            loss_main = self.cost_dist(sim, emp)
        else:
            loss_main = self.cost_r(sim, emp)
        loss_EI = 0

        if model.model_name == 'RWW':
            E_window = next_window['E_window']
            I_window = next_window['I_window']
            f_window = next_window['f_window']
            v_window = next_window['v_window']
            x_window = next_window['x_window']
            q_window = next_window['q_window']
            if model.use_Gaussian_EI and model.use_Bifurcation:
                loss_EI = torch.mean(model.E_v_inv * (E_window - model.E_m) ** 2) \
                          + torch.mean(-torch.log(model.E_v_inv)) + \
                          torch.mean(model.I_v_inv * (I_window - model.I_m) ** 2) \
                          + torch.mean(-torch.log(model.I_v_inv)) + \
                          torch.mean(model.q_v_inv * (q_window - model.q_m) ** 2) \
                          + torch.mean(-torch.log(model.q_v_inv)) + \
                          torch.mean(model.v_v_inv * (v_window - model.v_m) ** 2) \
                          + torch.mean(-torch.log(model.v_v_inv)) \
                          + 5.0 * (m(model.sup_ca) * m(model.g_IE) ** 2
                                   - m(model.sup_cb) * m(model.g_IE)
                                   + m(model.sup_cc) - m(model.g_EI)) ** 2
            if model.use_Gaussian_EI and not model.use_Bifurcation:
                loss_EI = torch.mean(model.E_v_inv * (E_window - model.E_m) ** 2) \
                          + torch.mean(-torch.log(model.E_v_inv)) + \
                          torch.mean(model.I_v_inv * (I_window - model.I_m) ** 2) \
                          + torch.mean(-torch.log(model.I_v_inv)) + \
                          torch.mean(model.q_v_inv * (q_window - model.q_m) ** 2) \
                          + torch.mean(-torch.log(model.q_v_inv)) + \
                          torch.mean(model.v_v_inv * (v_window - model.v_m) ** 2) \
                          + torch.mean(-torch.log(model.v_v_inv))

            if not model.use_Gaussian_EI and model.use_Bifurcation:
                loss_EI = .1 * torch.mean(
                    torch.mean(E_window * torch.log(E_window) + (1 - E_window) * torch.log(1 - E_window) \
                               + 0.5 * I_window * torch.log(I_window) + 0.5 * (1 - I_window) * torch.log(
                        1 - I_window), dim=1)) + \
                          + 5.0 * (m(model.sup_ca) * m(model.g_IE) ** 2
                                   - m(model.sup_cb) * m(model.g_IE)
                                   + m(model.sup_cc) - m(model.g_EI)) ** 2

            if not model.use_Gaussian_EI and not model.use_Bifurcation:
                loss_EI = .1 * torch.mean(
                    torch.mean(E_window * torch.log(E_window) + (1 - E_window) * torch.log(1 - E_window) \
                               + 0.5 * I_window * torch.log(I_window) + 0.5 * (1 - I_window) * torch.log(
                        1 - I_window), dim=1))

            loss_prior = []

            variables_p = [a for a in dir(model.param) if
                           not a.startswith('__') and not callable(getattr(model.param, a))]
            # get penalty on each model parameters due to prior distribution
            for var in variables_p:
                # print(var)
                if model.use_Bifurcation:
                    if np.any(getattr(model.param, var)[1] > 0) and var not in ['std_in', 'g_EI', 'g_IE'] and \
                            var not in exclude_param:
                        # print(var)
                        dict_np = {'m': var + '_m', 'v': var + '_v_inv'}
                        loss_prior.append(torch.sum((lb + m(model.get_parameter(dict_np['v']))) * \
                                                    (m(model.get_parameter(var)) - m(
                                                        model.get_parameter(dict_np['m']))) ** 2) \
                                          + torch.sum(-torch.log(lb + m(model.get_parameter(dict_np['v'])))))
                else:
                    if np.any(getattr(model.param, var)[1] > 0) and var not in ['std_in'] and \
                            var not in exclude_param:
                        # print(var)
                        dict_np = {'m': var + '_m', 'v': var + '_v_inv'}
                        loss_prior.append(torch.sum((lb + m(model.get_parameter(dict_np['v']))) * \
                                                    (m(model.get_parameter(var)) - m(
                                                        model.get_parameter(dict_np['m']))) ** 2) \
                                          + torch.sum(-torch.log(lb + m(model.get_parameter(dict_np['v'])))))
        else:
            lose_EI = 0
            loss_prior = []

            variables_p = [a for a in dir(model.param) if
                           not a.startswith('__') and not callable(getattr(model.param, a))]

            for var in variables_p:
                if np.any(getattr(model.param, var)[1] > 0) and var not in ['std_in'] and \
                        var not in exclude_param:
                    # print(var)
                    dict_np = {'m': var + '_m', 'v': var + '_v_inv'}
                    loss_prior.append(torch.sum((lb + m(model.get_parameter(dict_np['v']))) * \
                                                (m(model.get_parameter(var)) - m(
                                                    model.get_parameter(dict_np['m']))) ** 2) \
                                      + torch.sum(-torch.log(lb + m(model.get_parameter(dict_np['v'])))))
        # total loss
        loss = 0
        if model.model_name == 'RWW':
            loss = 0.1 * w_cost * loss_main + 1 * sum(
                loss_prior) + 1 * loss_EI
        elif model.model_name == 'JR':
            loss = w_cost * loss_main + sum(loss_prior) + 1 * loss_EI
        elif model.model_name == 'LIN':
            loss = 0.1 * w_cost * loss_main + sum(loss_prior) + 1 * loss_EI
        return loss


class Model_fitting:
    """
    Using ADAM and AutoGrad to fit JansenRit to empirical EEG
    Attributes
    ----------
    model: instance of class RNNJANSEN
        forward model JansenRit
    ts: array with num_tr x node_size
        empirical EEG time-series
    num_epoches: int
        the times for repeating trainning
    cost: choice of the cost function
    """
    u = 0  # external input

    # from sklearn.metrics.pairwise import cosine_similarity
    def __init__(self, model, ts, num_epoches, cost):
        """
        Parameters
        ----------
        model: instance of class RNNJANSEN
            forward model JansenRit
        ts: array with num_tr x node_size
            empirical EEG time-series
        num_epoches: int
            the times for repeating trainning
        """
        self.model = model
        self.num_epoches = num_epoches
        # placeholder for output(EEG and histoty of model parameters and loss)
        self.output_sim = OutputNM(self.model.model_name, self.model.param,
                                   self.model.use_fit_gains, self.model.use_fit_lfm)
        # self.u = u
        """if ts.shape[1] != model.node_size:
            print('ts is a matrix with the number of datapoint X the number of node')
        else:
            self.ts = ts"""
        self.ts = ts

        self.cost = Costs(cost)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train(self, learningrate=0.05, u=0):
        """
        Parameters
        ----------
        learningrate : for machine learing speed
        u: stimulus

        """

        delays_max = 500
        state_ub = 0.01
        state_lb = -0.01

        if self.model.model_name == "RWW":
            if not self.model.use_dynamic_boundary:
                if self.model.use_fit_gains:
                    epoch_min = 10  # run minimum epoch # part of stop criteria
                    r_lb = 0.85  # lowest pearson correlation # part of stop criteria
                else:
                    epoch_min = 10  # run minimum epoch # part of stop criteria
                    r_lb = 0.85  # lowest pearson correlation # part of stop criteria
            else:
                epoch_min = 10  # run minimum epoch # part of stop criteria
                r_lb = 0.85  # lowest pearson correlation # part of stop criteria
        else:
            epoch_min = 200  # run minimum epoch # part of stop criteria
            r_lb = 0.95

        self.u = u

        # define an optimizer(ADAM)
        optimizer = optim.Adam(self.model.parameters(), lr=learningrate, eps=1e-7)

        # initial state
        X = 0
        if self.model.model_name == 'RWW':
            # initial state
            X = torch.tensor(0.2 * np.random.uniform(0, 1, (self.model.node_size, self.model.state_size)) + np.array(
                [0, 0, 0, 1.0, 1.0, 1.0]), dtype=torch.float32)
        elif self.model.model_name == 'LIN':
            # initial state
            X = torch.tensor(0.2 * np.random.randn(self.model.node_size, self.model.state_size) + np.array(
                [0, 0.5, 1.0, 1.0, 1.0]), dtype=torch.float32)
        elif self.model.model_name == 'JR':
            X = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size, self.model.state_size)),
                             dtype=torch.float32)
        # initials of history of E
        hE = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size, delays_max)),
                          dtype=torch.float32)

        # define masks for getting lower triangle matrix indices
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # placeholders for the history of model parameters
        fit_param = {}
        exclude_param = []
        fit_sc = 0
        fit_lm = 0
        loss = 0
        if self.model.use_fit_gains:
            exclude_param.append('gains_con')
            fit_sc = [self.model.sc[mask].copy()]  # sc weights history
        if self.model.model_name == "JR" and self.model.use_fit_lfm:
            exclude_param.append('lm')
            fit_lm = [self.model.lm.detach().numpy().ravel().copy()]  # leadfield matrix history

        for key, value in self.model.state_dict().items():
            if key not in exclude_param:
                fit_param[key] = [value.detach().numpy().ravel().copy()]

        loss_his = []  # loss placeholder

        # define constant 1 tensor

        # define num_windows
        num_windows = self.ts.shape[1]
        for i_epoch in range(self.num_epoches):

            # Create placeholders for the simulated states and outputs of entire time series.
            for name in self.model.state_names + [self.output_sim.output_name]:
                setattr(self.output_sim, name + '_train', [])

            # initial the external inputs
            external = torch.tensor(
                np.zeros([self.model.node_size, self.model.steps_per_TR, self.model.TRs_per_window]),
                dtype=torch.float32)

            # Perform the training in windows.

            for TR_i in range(num_windows):

                # Reset the gradient to zeros after update model parameters.
                optimizer.zero_grad()

                # if the external not empty
                if not isinstance(self.u, int):
                    external = torch.tensor(
                        (self.u[:, :, TR_i * self.model.TRs_per_window:(TR_i + 1) * self.model.TRs_per_window]),
                        dtype=torch.float32)

                # Use the model.forward() function to update next state and get simulated EEG in this batch.

                next_window, hE_new = self.model(external, X, hE)

                # Get the batch of empirical EEG signal.
                ts_window = torch.tensor(self.ts[i_epoch, TR_i, :, :], dtype=torch.float32)

                # total loss calculation
                sim = 0
                if self.model.model_name == 'RWW':
                    sim = next_window['bold_window']
                elif self.model.model_name == 'JR':
                    sim = next_window['eeg_window']
                elif self.model.model_name == 'LIN':
                    sim = next_window['bold_window']
                if TR_i in [5,6]:
                    loss = 5*self.cost.cost_eff(sim, ts_window, self.model, next_window)
                else:
                    loss = self.cost.cost_eff(sim, ts_window, self.model, next_window)


                # Put the batch of the simulated EEG, E I M Ev Iv Mv in to placeholders for entire time-series.
                for name in self.model.state_names + [self.output_sim.output_name]:
                    name_next = name + '_window'
                    tmp_ls = getattr(self.output_sim, name + '_train')
                    tmp_ls.append(next_window[name_next].detach().numpy())

                    setattr(self.output_sim, name + '_train', tmp_ls)

                loss_his.append(loss.detach().numpy())

                # Calculate gradient using backward (backpropagation) method of the loss function.
                loss.backward(retain_graph=True)

                # Optimize the model based on the gradient method in updating the model parameters.
                optimizer.step()

                # Put the updated model parameters into the history placeholders.
                # sc_par.append(self.model.sc[mask].copy())
                for key, value in self.model.state_dict().items():
                    if key not in exclude_param:
                        fit_param[key].append(value.detach().numpy().ravel().copy())

                if self.model.use_fit_gains:
                    fit_sc.append(self.model.sc_fitted.detach().numpy()[mask].copy())
                if self.model.model_name == "JR" and self.model.use_fit_lfm:
                    fit_lm.append(self.model.lm.detach().numpy().ravel().copy())

                # last update current state using next state...
                # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
                X = torch.tensor(next_window['current_state'].detach().numpy(), dtype=torch.float32)
                hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)
                # print(hE_new.detach().numpy()[20:25,0:20])
                # print(hE.shape)
            ts_emp = np.concatenate(list(self.ts[i_epoch]),1)
            fc = np.corrcoef(ts_emp)

            tmp_ls = getattr(self.output_sim, self.output_sim.output_name + '_train')
            ts_sim = np.concatenate(tmp_ls, axis=1)
            fc_sim = np.corrcoef(ts_sim[:, 10:])

            print('epoch: ', i_epoch, loss.detach().numpy())

            print('epoch: ', i_epoch, np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], 'cos_sim: ',
                  np.diag(cosine_similarity(ts_sim, ts_emp)).mean())

            for name in self.model.state_names + [self.output_sim.output_name]:
                tmp_ls = getattr(self.output_sim, name + '_train')
                setattr(self.output_sim, name + '_train', np.concatenate(tmp_ls, axis=1))

            self.output_sim.loss = np.array(loss_his)

            if i_epoch > epoch_min and np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1] > r_lb:
                break

        if self.model.use_fit_gains:
            self.output_sim.weights = np.array(fit_sc)
        if self.model.model_name == 'JR' and self.model.use_fit_lfm:
            self.output_sim.leadfield = np.array(fit_lm)
        for key, value in fit_param.items():
            setattr(self.output_sim, key, np.array(value))

    def test(self, base_window_num, u=0):
        """
        Parameters
        ----------
        base_window_num: int
            length of num_windows for resting
        u : external or stimulus
        -----------
        """

        # define some constants
        state_lb = -0.01
        state_ub = 0.01
        delays_max = 500
        transient_num = 10

        self.u = u

        # initial state
        X = 0
        if self.model.model_name == 'RWW':
            # initial state
            X = torch.tensor(0.2 * np.random.uniform(0, 1, (self.model.node_size, self.model.state_size)) + np.array(
                [0, 0, 0, 1.0, 1.0, 1.0]), dtype=torch.float32)
        elif self.model.model_name == 'LIN':
            # initial state
            X = torch.tensor(0.2 * np.random.randn(self.model.node_size, self.model.state_size) + np.array(
                [0, 0.5, 1.0, 1.0, 1.0]), dtype=torch.float32)
        elif self.model.model_name == 'JR':
            X = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size, self.model.state_size)),
                             dtype=torch.float32)
        hE = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size, 500)), dtype=torch.float32)

        # placeholders for model parameters

        # define mask for getting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # define num_windows
        num_windows = self.ts.shape[1]
        # Create placeholders for the simulated BOLD E I x f and q of entire time series.
        for name in self.model.state_names + [self.output_sim.output_name]:
            setattr(self.output_sim, name + '_test', [])

        u_hat = np.zeros(
            (self.model.node_size,self.model.steps_per_TR,
             base_window_num *self.model.TRs_per_window + self.ts.shape[1]*self.ts.shape[3]))
        u_hat[:, :, base_window_num * self.model.TRs_per_window:] = self.u

        # Perform the training in batches.

        for TR_i in range(num_windows + base_window_num):

            # Get the input and output noises for the module.

            external = torch.tensor(
                (u_hat[:, :, TR_i * self.model.TRs_per_window:(TR_i + 1) * self.model.TRs_per_window]),
                dtype=torch.float32)

            # Use the model.forward() function to update next state and get simulated EEG in this batch.
            next_window, hE_new = self.model(external, X, hE)

            if TR_i > base_window_num - 1:
                for name in self.model.state_names + [self.output_sim.output_name]:
                    name_next = name + '_window'
                    tmp_ls = getattr(self.output_sim, name + '_test')
                    tmp_ls.append(next_window[name_next].detach().numpy())

                    setattr(self.output_sim, name + '_test', tmp_ls)

            # last update current state using next state...
            # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
            X = torch.tensor(next_window['current_state'].detach().numpy(), dtype=torch.float32)
            hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)
            # print(hE_new.detach().numpy()[20:25,0:20])
            # print(hE.shape)

        ts_emp = np.concatenate(list(self.ts[-1]),1)
        fc = np.corrcoef(ts_emp)
        tmp_ls = getattr(self.output_sim, self.output_sim.output_name + '_test')
        ts_sim = np.concatenate(tmp_ls, axis=1)

        fc_sim = np.corrcoef(ts_sim[:, transient_num:])
        print(np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], 'cos_sim: ',
                  np.diag(cosine_similarity(ts_sim, ts_emp)).mean())
        for name in self.model.state_names + [self.output_sim.output_name]:
            tmp_ls = getattr(self.output_sim, name + '_test')
            setattr(self.output_sim, name + '_test', np.concatenate(tmp_ls, axis=1))

    def test_realtime(self, tr_p, step_size_n, step_size, num_windows):
        if self.model.model_name == 'RWW':
            mask = np.tril_indices(self.model.node_size, -1)

            X_np = 0.2 * np.random.uniform(0, 1, (self.model.node_size, self.model.state_size)) + np.array(
                [0, 0, 0, 1.0, 1.0, 1.0])
            variables_p = [a for a in dir(self.model.param) if
                           not a.startswith('__') and not callable(getattr(self.model.param, a))]
            # get penalty on each model parameters due to prior distribution
            for var in variables_p:
                # print(var)
                if np.any(getattr(self.model.param, var)[1] > 0):
                    des = getattr(self.model.param, var)
                    value = getattr(self.model, var)
                    des[0] = value.detach().numpy().copy()
                    setattr(self.model.param, var, des)
            model_np = WWD_np(self.model.node_size, self.model.TRs_per_window, step_size_n, step_size, tr_p,
                              self.model.sc_fitted.detach().numpy().copy(),
                              self.model.use_dynamic_boundary, self.model.use_Laplacian, self.model.param)

            # Create placeholders for the simulated BOLD E I x f and q of entire time series.
            for name in self.model.state_names + [self.output_sim.output_name]:
                setattr(self.output_sim, name + '_test', [])

            # Perform the training in batches.

            for TR_i in range(num_windows + 10):

                noise_in_np = np.random.randn(self.model.node_size, self.model.TRs_per_window, int(tr_p / step_size_n),
                                              2)

                noise_BOLD_np = np.random.randn(self.model.node_size, self.model.TRs_per_window)

                next_window_np = model_np.forward(X_np, noise_in_np, noise_BOLD_np)
                if TR_i >= 10:
                    # Put the batch of the simulated BOLD, E I x f v q in to placeholders for entire time-series.
                    for name in self.model.state_names + [self.output_sim.output_name]:
                        name_next = name + '_window'
                        tmp_ls = getattr(self.output_sim, name + '_test')
                        tmp_ls.append(next_window_np[name_next])

                        setattr(self.output_sim, name + '_test', tmp_ls)

                # last update current state using next state...
                # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
                X_np = next_window_np['current_state']
            tmp_ls = getattr(self.output_sim, self.output_sim.output_name + '_test')

            for name in self.model.state_names + [self.output_sim.output_name]:
                tmp_ls = getattr(self.output_sim, name + '_test')
                setattr(self.output_sim, name + '_test', np.concatenate(tmp_ls, axis=1))
        else:
            print("only WWD model for the test_realtime function")


