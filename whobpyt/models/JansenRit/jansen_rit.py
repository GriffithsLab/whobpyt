"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak
Neural Mass Model fitting module for JR with forward, backward, and lateral connection for EEG
"""

# @title new function PyTepFit

# Pytorch stuff


"""
Importage
"""
import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes import AbstractNMM, par
from whobpyt.models.JansenRit.ParamsJR import ParamsJR
from whobpyt.functions.arg_type_check import method_arg_type_check
import numpy as np


class RNNJANSEN(AbstractNMM):
    """
    A module for forward model (JansenRit) to simulate EEG signals
    
    Attibutes
    ---------
    state_size : int
        Number of states in the JansenRit model

    output_size : int
        Number of EEG channels.

    node_size: int
        Number of ROIs

    hidden_size: int
        Number of step_size for each sampling step

    step_size: float
        Integration step for forward model

    tr : float # TODO: CHANGE THE NAME TO sampling_rate
        Sampling rate of the simulated EEG signals 

    TRs_per_window: int # TODO: CHANGE THE NAME
        Number of EEG signals to simulate

    sc: ndarray (node_size x node_size) of floats
        Structural connectivity

    lm: ndarray of floats
        Leadfield matrix from source space to EEG space

    dist: ndarray of floats
        Distance matrix

    use_fit_gains: bool
        Flag for fitting gains. 1: fit, 0: not fit

    use_fit_lfm: bool
        Flag for fitting the leadfield matrix. 1: fit, 0: not fit

    # FIGURE OUT: g, c1, c2, c3, c4: tensor with gradient on 
    #     model parameters to be fit

    std_in: tensor with gradient on
        Standard deviation for input noise

    params: ParamsJR
        Model parameters object.


    Methods
    -------
    createIC(self, ver):
        Creates the initial conditions for the model.

    createDelayIC(self, ver):
        Creates the initial conditions for the delays.

    setModelParameters(self):    
        Sets the parameters of the model.
    
    forward(input, noise_out, hx)
        Forward pass for generating a number of EEG signals with current model parameters
    
    """

    def __init__(self, node_size: int,
                 TRs_per_window: int, step_size: float, output_size: int, tr: float, sc: np.ndarray, lm: np.ndarray, dist: np.ndarray,
                 use_fit_gains: bool, use_fit_lfm: bool, params: ParamsJR) -> None:               
        """
        Parameters
        ----------
        node_size: int
            Number of ROIs
        TRs_per_window: int # TODO: CHANGE THE NAME
            Number of EEG signals to simulate
        step_size: float
            Integration step for forward model
        output_size : int
            Number of EEG channels.
        tr : float # TODO: CHANGE THE NAME TO sampling_rate
            Sampling rate of the simulated EEG signals 
        sc: ndarray node_size x node_size float array
            Structural connectivity
        lm: ndarray float array
            Leadfield matrix from source space to EEG space
        dist: ndarray float array
            Distance matrix
        use_fit_gains: bool
            Flag for fitting gains. 1: fit, 0: not fit
        use_fit_lfm: bool
            Flag for fitting the leadfield matrix. 1: fit, 0: not fit
        params: ParamsJR
            Model parameters object.
        """
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types
        
        super(RNNJANSEN, self).__init__()
        self.state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
        self.output_names = ["eeg"]
        self.track_params = [] #Is populated during setModelParameters()
        
        self.model_name = "JR"
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
        self.params = params
        self.output_size = lm.shape[0]  # number of EEG channels
        
        self.setModelParameters()

    def info(self):
        # TODO: Make sure this method is useful
        """
        Returns a dictionary with the names of the states and the output.

        Returns
        -------
        Dict[str, List[str]]
        """

        return {"state_names": ['E', 'Ev', 'I', 'Iv', 'P', 'Pv'], "output_names": ["eeg"]}
    
    def createIC(self, ver):
        """
        Creates the initial conditions for the model.

        Parameters
        ----------
        ver : int # TODO: ADD MORE DESCRIPTION
            Initial condition version. (in the JR model, the version is not used. It is just for consistency with other models)

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, state_size) with random values between `state_lb` and `state_ub`.
        """

        state_lb = -0.5
        state_ub = 0.5
        
        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size, self.state_size)),
                             dtype=torch.float32)
                             
    def createDelayIC(self, ver):
        """
        Creates the initial conditions for the delays.

        Parameters
        ----------
        ver : int
            Initial condition version. (in the JR model, the version is not used. It is just for consistency with other models)

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, delays_max) with random values between `state_lb` and `state_ub`.
        """

        delays_max = 500
        state_ub = 0.5
        state_lb = -0.5

        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size, delays_max)), dtype=torch.float32)
    
    def setModelParameters(self):
        """
        Sets the parameters of the model.
        """

        param_reg = []
        param_hyper = []

        # Set w_bb, w_ff, and w_ll as attributes as type Parameter if use_fit_gains is True
        if self.use_fit_gains:
            self.w_bb = Parameter(torch.tensor(np.zeros((self.node_size, self.node_size)) + 0.05, # the backwards gains
                                                dtype=torch.float32))
            self.w_ff = Parameter(torch.tensor(np.zeros((self.node_size, self.node_size)) + 0.05, # the forward gains
                                                dtype=torch.float32))
            self.w_ll = Parameter(torch.tensor(np.zeros((self.node_size, self.node_size)) + 0.05, # the lateral gains
                                                dtype=torch.float32))
            param_reg.append(self.w_ll)
            param_reg.append(self.w_ff)
            param_reg.append(self.w_bb)
        else:
            self.w_bb = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)
            self.w_ff = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)
            self.w_ll = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)

        # If use_fit_lfm is True, set lm as an attribute as type Parameter (containing variance information)
        if self.use_fit_lfm:
            self.lm = Parameter(torch.tensor(self.lm, dtype=torch.float32))  # leadfield matrix from sourced data to eeg
            param_reg.append(self.lm)
        else:
            self.lm = torch.tensor(self.lm, dtype=torch.float32)

        var_names = [a for a in dir(self.params) if (type(getattr(self.params, a)) == par)]
        for var_name in var_names:
            var = getattr(self.params, var_name)
            if (var.fit_hyper):
                if var_name == 'lm':
                    size = var.prior_var.shape
                    var.val = Parameter(var.val.detach() - 1 * torch.ones((size[0], size[1]))) # TODO: This is not consistent with what user would expect giving a variance 
                    param_hyper.append(var.prior_mean)
                    param_hyper.append(var.prior_var)
                elif (var != 'std_in'):
                    var.randSet() #TODO: This should be done before giving params to model class
                    param_hyper.append(var.prior_mean)
                    param_hyper.append(var.prior_var)

            if (var.fit_par):
                param_reg.append(var.val) #TODO: This should got before fit_hyper, but need to change where randomness gets added in the code first 
                
            if (var.fit_par | var.fit_hyper):
                self.track_params.append(var_name) #NMM Parameters
            
            if var_name == 'lm':
                setattr(self, var_name, var.val)
            
        self.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}


    def forward(self, external, hx, hE):
        """
        This function carries out the forward Euler integration method for the JR neural mass model,
        with time delays, connection gains, and external inputs considered. Each population (pyramidal,
        excitatory, inhibitory) in the network is modeled as a non-linear second order system. The function
        updates the state of each neural population and computes the EEG signals at each time step.

        Parameters
        ----------
        external : torch.Tensor
            Input tensor of shape (batch_size, num_ROIs) representing the input to the model.
        hx : Optional[torch.Tensor]
            Optional tensor of shape (batch_size, state_size, num_ROIs) representing the initial hidden state.
        hE : Optional[torch.Tensor]
            Optional tensor of shape (batch_size, num_ROIs, delays_max) representing the initial delays.

        Returns
        -------
        next_state : dict
            Dictionary containing the updated current state, EEG signals, and the history of
            each population's current and voltage at each time step.

        hE : torch.Tensor
            Tensor representing the updated history of the pyramidal population's current.
        """

        # Generate the ReLU module
        m = torch.nn.ReLU()
        
        # Define some constants        
        con_1 = torch.tensor(1.0, dtype=torch.float32) # Define constant 1 tensor
        conduct_lb = 1.5  # lower bound for conduct velocity
        u_2ndsys_ub = 500  # the bound of the input for second order system
        noise_std_lb = 20  # lower bound of std of noise
        lb = 0.01  # lower bound of local gains
        k_lb = 0.5  # lower bound of coefficient of external inputs


        # Defining NMM Parameters to simplify later equations
        #TODO: Change code so that params returns actual value used without extras below
        A = 0 * con_1 + m(self.params.A.value()) 
        a = 1 * con_1 + m(self.params.a.value())
        B = 0 * con_1 + m(self.params.B.value())
        b = 1 * con_1 + m(self.params.b.value())
        g = (lb * con_1 + m(self.params.g.value()))
        c1 = (lb * con_1 + m(self.params.c1.value()))
        c2 = (lb * con_1 + m(self.params.c2.value()))
        c3 = (lb * con_1 + m(self.params.c3.value()))
        c4 = (lb * con_1 + m(self.params.c4.value()))
        std_in = (noise_std_lb * con_1 + m(self.params.std_in.value()))
        vmax = self.params.vmax.value()
        v0 = self.params.v0.value()
        r = self.params.r.value()
        y0 = self.params.y0.value()
        mu = (conduct_lb * con_1 + m(self.params.mu.value()))
        k = (k_lb * con_1 + m(self.params.k.value()))
        cy0 = self.params.cy0.value()
        ki = self.params.ki.value()
        
        g_f = (lb * con_1 + m(self.params.g_f.value()))
        g_b = (lb * con_1 + m(self.params.g_b.value()))


        next_state = {}

        M = hx[:, 0:1]  # current of pyramidal population
        E = hx[:, 1:2]  # current of excitory population
        I = hx[:, 2:3]  # current of inhibitory population

        Mv = hx[:, 3:4]  # voltage of pyramidal population
        Ev = hx[:, 4:5]  # voltage of exictory population
        Iv = hx[:, 5:6]  # voltage of inhibitory population

        dt = self.step_size

        if self.sc.shape[0] > 1:

            # Update the Laplacian based on the updated connection gains w_bb.
            w_b = torch.exp(self.w_bb) * torch.tensor(self.sc, dtype=torch.float32)
            w_n_b = w_b / torch.linalg.norm(w_b)
            self.sc_m_b = w_n_b
            dg_b = -torch.diag(torch.sum(w_n_b, dim=1))

            # Update the Laplacian based on the updated connection gains w_ff.
            w_f = torch.exp(self.w_ff) * torch.tensor(self.sc, dtype=torch.float32)
            w_n_f = w_f / torch.linalg.norm(w_f)
            self.sc_m_f = w_n_f
            dg_f = -torch.diag(torch.sum(w_n_f, dim=1))

            # Update the Laplacian based on the updated connection gains w_ll.
            w_l = torch.exp(self.w_ll) * torch.tensor(self.sc, dtype=torch.float32)
            w_n_l = (0.5 * (w_l + torch.transpose(w_l, 0, 1))) / torch.linalg.norm(
                0.5 * (w_l + torch.transpose(w_l, 0, 1)))
            self.sc_fitted = w_n_l
            dg_l = -torch.diag(torch.sum(w_n_l, dim=1))
        else:
            l_s = torch.tensor(np.zeros((1, 1)), dtype=torch.float32) #TODO: This is not being called anywhere
            dg_l = 0
            dg_b = 0
            dg_f = 0
            w_n_l = 0
            w_n_b = 0
            w_n_f = 0

        self.delays = (self.dist / mu).type(torch.int64)

        # Placeholder for the updated current state
        current_state = torch.zeros_like(hx)

        # Initializing lists for the history of the EEG signals, as well as each population's current and voltage.
        eeg_window = []
        E_window = []
        I_window = []
        M_window = []
        Ev_window = []
        Iv_window = []
        Mv_window = []

        # Use the forward model to get EEG signal at the i-th element in the window.
        for i_window in range(self.TRs_per_window):
            for step_i in range(self.steps_per_TR):
                Ed = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)  # delayed E
                hE_new = hE.clone()
                Ed = hE_new.gather(1, self.delays)
                LEd_b = torch.reshape(torch.sum(w_n_b * torch.transpose(Ed, 0, 1), 1),
                                    (self.node_size, 1))
                LEd_f = torch.reshape(torch.sum(w_n_f * torch.transpose(Ed, 0, 1), 1),
                                    (self.node_size, 1))
                LEd_l = torch.reshape(torch.sum(w_n_l * torch.transpose(Ed, 0, 1), 1),
                                    (self.node_size, 1))
                
                # TMS (or external) input
                u_tms = external[:, step_i:step_i + 1, i_window]
                rM = k * ki * u_tms + std_in * torch.randn(self.node_size, 1) + \
                    1 * g * (LEd_l + 1 * torch.matmul(dg_l, M)) + \
                    sigmoid(E - I, vmax, v0, r)  # firing rate for pyramidal population
                rE = std_in * torch.randn(self.node_size, 1) + \
                    1 * g_f * (LEd_f + 1 * torch.matmul(dg_f, E - I)) + \
                    c2 * sigmoid(c1 * M, vmax, v0, r)  # firing rate for excitatory population
                rI = std_in * torch.randn(self.node_size, 1) + \
                    1 * g_b * (-LEd_b - 1 * torch.matmul(dg_b, E - I)) + \
                    c4 * sigmoid(c3 * M, vmax, v0, r)  # firing rate for inhibitory population

                # Update the states with every step size.
                ddM = M + dt * Mv
                ddE = E + dt * Ev
                ddI = I + dt * Iv
                ddMv = Mv + dt * sys2nd(A, a, u_2ndsys_ub * torch.tanh(rM / u_2ndsys_ub), M, Mv)
                ddEv = Ev + dt * sys2nd(A, a, u_2ndsys_ub * torch.tanh(rE / u_2ndsys_ub), E, Ev)
                ddIv = Iv + dt * sys2nd(B, b, u_2ndsys_ub * torch.tanh(rI / u_2ndsys_ub), I, Iv)

                # Calculate the saturation for model states (for stability and gradient calculation).
                E = 1000*torch.tanh(ddE/1000)
                I = 1000*torch.tanh(ddI/1000)
                M = 1000*torch.tanh(ddM/1000)
                Ev = 1000*torch.tanh(ddEv/1000)
                Iv = 1000*torch.tanh(ddIv/1000)
                Mv = 1000*torch.tanh(ddMv/1000)

                # Update placeholders for pyramidal buffer
                hE[:, 0] = M[:, 0]
                
            # Capture the states at every tr in the placeholders for checking them visually.
            M_window.append(M)
            I_window.append(I)
            E_window.append(E)
            Mv_window.append(Mv)
            Iv_window.append(Iv)
            Ev_window.append(Ev)
            hE = torch.cat([M, hE[:, :-1]], dim=1)  # update placeholders for pyramidal buffer

            # Capture the states at every tr in the placeholders which is then used in the cost calculation.
            lm_t = (self.lm.T / torch.sqrt(self.lm ** 2).sum(1)).T
            self.lm_t = (lm_t - 1 / self.output_size * torch.matmul(torch.ones((1, self.output_size)), lm_t))
            temp = cy0 * torch.matmul(self.lm_t, M[:200, :]) - 1 * y0
            eeg_window.append(temp)

        # Update the current state.
        current_state = torch.cat([M, E, I, Mv, Ev, Iv], dim=1)
        next_state['current_state'] = current_state
        next_state['eeg'] = torch.cat(eeg_window, dim=1)
        next_state['E'] = torch.cat(E_window, dim=1)
        next_state['I'] = torch.cat(I_window, dim=1)
        next_state['P'] = torch.cat(M_window, dim=1)
        next_state['Ev'] = torch.cat(Ev_window, dim=1)
        next_state['Iv'] = torch.cat(Iv_window, dim=1)
        next_state['Pv'] = torch.cat(Mv_window, dim=1)

        return next_state, hE


def sigmoid(x, vmax, v0, r):
    """
    Calculates the sigmoid function for a given input.

    Args:
        x (torch.Tensor): The input tensor.
        vmax (float): The maximum value of the sigmoid function.
        v0 (float): The midpoint of the sigmoid function.
        r (float): The slope of the sigmoid function.

    Returns:
        torch.Tensor: The output tensor.
    """
    return vmax / (1 + torch.exp(r * (v0 - x)))
    
def sys2nd(A, a, u, x, v):
    """
    Calculates the second-order system (for each population [represented by A, a]) for a given set of inputs.

    Args:
        A (float): The amplitude of the PSP.
        a (float): Metric of the rate constant of the PSP.
        u (torch.Tensor): The input tensor.
        x (torch.Tensor): The state tensor.
        v (torch.Tensor): The delay tensor.

    Returns:
        torch.Tensor: The output tensor.
    """
    return A * a * u - 2 * a * v - a ** 2 * x
