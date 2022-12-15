"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for functions used in the model
"""

import numpy as np  # for numerical operations
import torch
from torch.nn.parameter import Parameter


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
            model.lm = Parameter(torch.tensor(model.lm, dtype=torch.float32))  # leadfield matrix from sourced data to eeg
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
