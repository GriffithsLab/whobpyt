"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for wong-wang model
"""

import numpy as np  # for numerical operations
from whobpyt.models.RWW import ParamsRWW


def h_tf_np(a, b, d, z):
    """
    Neuronal input-output functions of excitatory pools and inhibitory pools.

    Take the variables a, x, and b and convert them to a linear equation (a*x - b) while adding a small
    amount of noise 0.00001 while dividing that term to an exponential of the linear equation multiplied by the
    d constant for the appropriate dimensions.
    """
    num = 0.00001 + np.abs(a * z - b)
    den = 0.00001 * d + np.abs(1.0000 - np.exp(-d * (a * z - b)))
    return num / den


class RWW_np:
    """
    A module for forward model (WWD) to simulate a batch of BOLD signals

    Attibutes
    ---------
    state_size : int
        the number of states in the WWD model
    input_size : int
        the number of states with noise as input
    tr : float
        tr of fMRI image
    step_size: float
        Integration step for forward neural model in ms
    step_size_bold: float
        Integration step for forward balloon model in s
    steps_per_TR: int
        the number of step_size in a tr
    TRs_per_window: int
        the number of BOLD signals to simulate
    node_size: int
        the number of ROIs
    sc: float node_size x node_size array
        structural connectivity
    use_Laplacian: bool
            using Laplacian or not
        param: ParamsModel
            define model parameters(var:0 constant var:non-zero Parameter)
    Methods
    -------
    forward(input,  hx, u , u_out)
        forward model (WWD) for generating a number of BOLD signals with current model parameters
    """

    def __init__(self, node_size: int, TRs_per_window: int, step_size: float, step_size_bold: float, tr: float,
                 sc: float, use_dynamic_boundary: bool,
                 use_Laplacian: bool, param: ParamsRWW) -> None:
        """
        Parameters
        ----------

        tr : float in
            tr of fMRI image
        step_size: float
            Integration step for forward model

        TRs_per_window: int
            the number of BOLD signals to simulate
        node_size: int
            the number of ROIs
        sc: float node_size x node_size array
            structural connectivity

        """
        super(WWD_np, self).__init__()

        self.step_size = step_size  # integration step 0.05
        self.step_size_bold = step_size_bold  # integration step 0.05
        self.node_size = node_size  # num of ROI
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.use_Laplacian = use_Laplacian
        self.use_dynamic_boundary = use_dynamic_boundary
        vars_name = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars_name:
            setattr(self, var, getattr(param, var)[0])

    def forward(self, hx, u, u_out):
        """
        Forward step in simulating the BOLD signal.
        Parameters
        ----------
        u: tensor with node_size x steps_per_TR x TRs_per_window x input_size
            noise for states
        u_out: tensor with node_size x TRs_per_window
            noise for BOLD
        hx: tensor with node_size x state_size
            states of WWD model
        Outputs
        -------
        next_state: dictionary with keys:
        'current_state''bold_window''E_window''I_window''x_window''f_window''v_window''q_window'
            record new states and BOLD
        """
        next_state = {}
        dt = self.step_size

        if self.use_Laplacian:
            lap_adj = -np.diag(self.sc.sum(1)) + self.sc
        else:
            lap_adj = self.sc

        E = hx[:, 0:1]
        I = hx[:, 1:2]
        x = hx[:, 2:3]
        f = hx[:, 3:4]
        v = hx[:, 4:5]
        q = hx[:, 5:6]
        E_window = np.zeros((self.node_size, self.TRs_per_window))
        I_window = np.zeros((self.node_size, self.TRs_per_window))
        bold_window = np.zeros((self.node_size, self.TRs_per_window))
        x_window = np.zeros((self.node_size, self.TRs_per_window))
        v_window = np.zeros((self.node_size, self.TRs_per_window))
        f_window = np.zeros((self.node_size, self.TRs_per_window))
        q_window = np.zeros((self.node_size, self.TRs_per_window))

        E_hist = np.zeros((self.node_size, self.TRs_per_window, self.steps_per_TR))
        # Use the forward model to get neural activity at ith element in the batch.

        if self.use_dynamic_boundary:
            for TR_i in range(self.TRs_per_window):

                # print(E.shape)

                # Since tr is about second we need to use a small step size like 0.05 to integrate the model states.
                for step_i in range(self.steps_per_TR):
                    noise_E = u[:, TR_i, step_i, 0][:, np.newaxis]
                    noise_I = u[:, TR_i, step_i, 1][:, np.newaxis]

                    IE = self.W_E * self.I_0 + (0.001 + max([0, self.g_EE])) * E \
                         + self.g * lap_adj.dot(E) - (0.001 + max([0, self.g_IE])) * I  # input currents for E
                    II = self.W_I * self.I_0 + (0.001 + max([0, self.g_EI])) * E - I  # input currents for I
                    IE[IE < 0] = 0
                    II[II < 0] = 0
                    IE = np.tanh(IE)
                    II = np.tanh(II)
                    # Calculate the firing rates.
                    rE = h_tf_np(self.aE, self.bE, self.dE, IE)  # firing rate for E
                    rI = h_tf_np(self.aI, self.bI, self.dI, II)  # firing rate for I
                    # Update the states by step-size 0.05.

                    E_next = E + dt * (-E / self.tau_E + self.gamma_E * (1. - E) * rE) \
                             + np.sqrt(dt) * noise_E * (0.02 + max(
                        [0, self.std_in]))  ### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)
                    I_next = I + dt * (-I / self.tau_I + self.gamma_I * rI) \
                             + np.sqrt(dt) * noise_I * (0.02 + max([0, self.std_in]))
                    E_next[E_next < 0] = 0
                    I_next[I_next < 0] = 0

                    E = np.tanh(E_next)
                    I = np.tanh(I_next)
                    """E_plus = E.copy()  
                    E_plus[E_plus<0] = 0"""
                    E_hist[:, TR_i, step_i] = E[:, 0]  # np.tanh(E_plus[:,0])
                """E_plus = E.copy()  
                E_plus[E_plus<0] = 0 
                I_plus = I.copy()  
                I_plus[I_plus<0] = 0"""
                E_window[:, TR_i] = E[:, 0]  # np.tanh(E_plus[:,0])
                I_window[:, TR_i] = I[:, 0]  # np.tanh(I_plus[:,0])

            temp_avg_size = int(1000 * self.step_size_bold / dt)

            for TR_i in range(self.TRs_per_window):

                noise_BOLD = u_out[:, TR_i][:, np.newaxis]
                for step_i in range(int(self.steps_per_TR / temp_avg_size)):
                    x_next = x + self.step_size_bold * (
                            (E_hist[:, TR_i, step_i * temp_avg_size:(1 + step_i) * temp_avg_size]).mean(1)[:,
                            np.newaxis] - x / self.tau_s - (f - 1) / self.tau_f)
                    f_next = f + self.step_size_bold * x
                    v_next = v + self.step_size_bold * (f - np.power(v, 1 / self.alpha)) / self.tau_0
                    q_next = q + self.step_size_bold * (f * (1 - np.power(1 - self.rho, 1 / f)) / self.rho \
                                                        - q * np.power(v, 1 / self.alpha) / v) / self.tau_0

                    x = np.tanh(x_next)
                    f = (1 + np.tanh(f_next - 1))
                    v = (1 + np.tanh(v_next - 1))
                    q = (1 + np.tanh(q_next - 1))
                    # Put x f v q from each tr to the placeholders for checking them visually.
                x_window[:, TR_i] = x[:, 0]
                f_window[:, TR_i] = f[:, 0]
                v_window[:, TR_i] = v[:, 0]
                q_window[:, TR_i] = q[:, 0]

                bold_window[:, TR_i] = (0.00 + max([0, self.std_out]) * noise_BOLD +
                                        100.0 * self.V / self.E0 * (self.k1 * (1 - q)
                                                                    + self.k2 * (1 - q / v) + self.k3 * (1 - v)))[:, 0]
        else:

            for TR_i in range(self.TRs_per_window):

                # print(E.shape)

                # Since tr is about second we need to use a small step size like 0.05 to integrate the model states.
                for step_i in range(self.steps_per_TR):
                    noise_E = u[:, TR_i, step_i, 0][:, np.newaxis]
                    noise_I = u[:, TR_i, step_i, 1][:, np.newaxis]

                    IE = self.W_E * self.I_0 + (0.001 + max([0, self.g_EE])) * E \
                         + self.g * lap_adj.dot(E) - (0.001 + max([0, self.g_IE])) * I  # input currents for E
                    II = self.W_I * self.I_0 + (0.001 + max([0, self.g_EI])) * E - I  # input currents for I
                    IE[IE < 0] = 0
                    II[II < 0] = 0
                    IE = np.tanh(IE)
                    II = np.tanh(II)
                    # Calculate the firing rates.
                    rE = h_tf_np(self.aE, self.bE, self.dE, IE)  # firing rate for E
                    rI = h_tf_np(self.aI, self.bI, self.dI, II)  # firing rate for I
                    # Update the states by step-size 0.05.

                    E_next = E + dt * (-E / self.tau_E + self.gamma_E * (1. - E) * rE) \
                             + np.sqrt(dt) * noise_E * (0.02 / 10 + max(
                        [0, self.std_in]))  ### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)
                    I_next = I + dt * (-I / self.tau_I + self.gamma_I * rI) \
                             + np.sqrt(dt) * noise_I * (0.02 / 10 + max([0, self.std_in]))

                    E_next[E_next < 0] = 0
                    I_next[I_next < 0] = 0
                    E = E_next  # np.tanh(0.00001+E_next)
                    I = I_next  # np.tanh(0.00001+I_next)
                    """E_plus = E.copy()  
                    E_plus[E_plus<0] = 0"""
                    E_hist[:, TR_i, step_i] = E[:, 0]
                """E_plus = E.copy()  
                E_plus[E_plus<0] = 0 
                I_plus = I.copy()  
                I_plus[I_plus<0] = 0"""
                E_window[:, TR_i] = E[:, 0]  # np.tanh(E_plus[:,0])
                I_window[:, TR_i] = I[:, 0]  # np.tanh(I_plus[:,0])

            temp_avg_size = int(1000 * self.step_size_bold / dt)

            for TR_i in range(self.TRs_per_window):

                noise_BOLD = u_out[:, TR_i][:, np.newaxis]
                for step_i in range(int(self.steps_per_TR / temp_avg_size)):
                    x_next = x + self.step_size_bold * (
                            (np.tanh(E_hist)[:, TR_i, step_i * temp_avg_size:(1 + step_i) * temp_avg_size]).mean(1)[
                            :, np.newaxis] - x / self.tau_s - (f - 1) / self.tau_f)
                    f_next = f + self.step_size_bold * x
                    v_next = v + self.step_size_bold * (f - np.power(v, 1 / self.alpha)) / self.tau_0
                    q_next = q + self.step_size_bold * (f * (1 - np.power(1 - self.rho, 1 / f)) / self.rho \
                                                        - q * np.power(v, 1 / self.alpha) / v) / self.tau_0

                    f_next[f_next < 0.001] = 0.001
                    v_next[v_next < 0.001] = 0.001
                    q_next[q_next < 0.001] = 0.001
                    x = x_next  # np.tanh(x_next)
                    f = f_next  # (1 + np.tanh(f_next - 1))
                    v = v_next  # (1 + np.tanh(v_next - 1))
                    q = q_next  # (1 + np.tanh(q_next - 1))
                # Put x f v q from each tr to the placeholders for checking them visually.
                x_window[:, TR_i] = x[:, 0]
                f_window[:, TR_i] = f[:, 0]
                v_window[:, TR_i] = v[:, 0]
                q_window[:, TR_i] = q[:, 0]
                # Put the BOLD signal each tr to the placeholder being used in the cost calculation.

                bold_window[:, TR_i] = (0.00 + max([0, self.std_out]) * noise_BOLD +
                                        100.0 * self.V / self.E0 * (self.k1 * (1 - q) + self.k2 * (1 - q / v)
                                        + self.k3 * (1 - v)))[:, 0]

        # Update the current state.
        # print(E_m.shape)
        current_state = np.concatenate([E, I, x, f, v, q], axis=1)
        next_state['current_state'] = current_state
        next_state['bold_window'] = bold_window
        next_state['E_window'] = E_window
        next_state['I_window'] = I_window
        next_state['x_window'] = x_window
        next_state['f_window'] = f_window
        next_state['v_window'] = v_window
        next_state['q_window'] = q_window
        return next_state

    def update_param(self, param_new):
        vars_name = [a for a in dir(param_new) if not a.startswith('__') and not callable(getattr(param_new, a))]
        for var in vars_name:
            setattr(self, var, getattr(param_new, var)[0])
