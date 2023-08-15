"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting module for Wong-Wang model
"""

import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes import AbstractNMM, AbstractParams, par
from whobpyt.models.RWW import ParamsRWW
from whobpyt.functions.arg_type_check import method_arg_type_check
import numpy as np # for numerical operations

class RNNRWW(AbstractNMM):
    """
    Reduced Wong Wang Excitatory Inhibitory (RWWExcInb) Model with integrated BOLD dynamics
    
    A module for forward model (WWD) to simulate a window of BOLD signals
    
    Note that the BOLD signal is not done in the standard way, 
    and there are other customizations to the neural mass model that may 
    deviate from standard differential equation simulation. Thus, the
    parameter should be tested on a validation model after fitting. 
 
 
    Attributes
    ---------
    
    state_names: list
        A list of model state variable names
    output_names: list
        A list of model output variable names
    model_name: string
        The name of the model itself
    state_size : int
        The number of states in the WWD model
    tr : float
        tr of fMRI image. That is, the spacing betweeen images in the time series. 
    step_size: float
        Integration step for forward model
    steps_per_TR: int
        The number of step_size in a tr. This is calculated automatically as int(tr / step_size).
    TRs_per_window: int
        The number of BOLD TRs to simulate in one forward call
    node_size: int
        The number of ROIs
    sampling_size: int
        This is related to an averaging of NMM values before inputing into hemodynamic equaitons. This is non-standard.        
    sc: float node_size x node_size array
        The structural connectivity matrix
    sc_fitted: bool
        The fitted structural connectivity
    use_fit_gains: tensor with node_size x node_size (grad on depends on fit_gains)
        Whether to fit the structural connectivity, will fit via connection gains: exp(gains_con)*sc
    use_Laplacian: bool
        Whether to use the negative laplacian of the (fitted) structural connectivity as the structural connectivity
    use_Bifurcation: bool
        Use a custom objective function component
    use_Gaussian_EI: bool
        Use a custom objective function component
    use_dynamic_boundary: bool
        Whether to have tanh function applied at each time step to constrain parameter values. Simulation results will become dependent on a certian step_size. 
    params: ParamsRWW
        A object that contains the parameters for the RWW nodes
    params_fitted: dictionary
        A dictionary containg fitted parameters and fitted hyper_parameters
    output_size: int
        Number of ROIs
  
    Methods
    -------
    
    forward(input, external, hx, hE)
        forward model (WWD) for generating a number of BOLD signals with current model parameters
    info(self)
        A function that returns a dictionary with model information.
    createIC(self, ver)
        A function to return an initial state tensor for the model.
    setModelParameters(self)
        A function that assigns model parameters as model attributes and also to assign parameters and hyperparameters for fitting, 
        so that the inherited Torch functionality can be used. 
        This practice may be replaced soon.  
   
    Other
    -------
        g_m g_v f_EE_m g_EE_v sup_ca sup_cb sup_cc: tensor with gradient on
        hyper parameters for prior distribution of g gEE gIE and gEI
        
        g, g_EE, gIE, gEI: tensor with gradient on
        model parameters to be fit
        
        std_in std_out: tensor with gradient on
        std for state noise and output noise

    """
    use_fit_lfm = False
    input_size = 2

    def __init__(self, node_size: int,
                 TRs_per_window: int, step_size: float, sampling_size: float, tr: float, sc: float, use_fit_gains: bool,
                 params: ParamsRWW, use_Bifurcation: bool = True, use_Gaussian_EI: bool = False, use_Laplacian: bool = True,
                 use_dynamic_boundary: bool = True) -> None:
        """
        Parameters
        ----------
            
        node_size: int
            The number of ROIs
        TRs_per_window: int
            The number of BOLD TRs to simulate in one forward call    
        step_size: float
            Integration step for forward model
        sampling_size:
            This is related to an averaging of NMM values before inputing into hemodynamic equaitons. This is non-standard. 
        tr : float
            tr of fMRI image. That is, the spacing betweeen images in the time series. 
        sc: float node_size x node_size array
            The structural connectivity matrix
        use_fit_gains: bool
            Whether to fit the structural connectivity, will fit via connection gains: exp(gains_con)*sc
        params: ParamsRWW
            A object that contains the parameters for the RWW nodes
        use_Bifurcation: bool
            Use a custom objective function component
        use_Gaussian_EI: bool
            Use a custom objective function component
        use_Laplacian: bool
            Whether to use the negative laplacian of the (fitted) structural connectivity as the structural connectivity
        use_dynamic_boundary: bool
            Whether to have tanh function applied at each time step to constrain parameter values. Simulation results will become dependent on a certian step_size.
        """        
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types
        
        super(RNNRWW, self).__init__()
        
        self.state_names = ['E', 'I', 'x', 'f', 'v', 'q']
        self.output_names = ["bold"]
        self.track_params = [] #Is populated during setModelParameters()
        
        self.model_name = "RWW"
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
        self.params = params
        self.params_fitted = {}

        self.output_size = node_size
        
        self.setModelParameters()
    
    def info(self):
        """
        
        A function that returns a dictionary with model information.
        
        Parameters
        ----------
        
        None
        
        
        Returns
        ----------
        
        Dictionary of Lists
            The List contain State Names and Output Names 
        
        
        """
    
        return {"state_names": ['E', 'I', 'x', 'f', 'v', 'q'], "output_names": ["bold"]}
    
    def createIC(self, ver):
        """
        
            A function to return an initial state tensor for the model.    
        
        Parameters
        ----------
        
        ver: int
            Ignored Parameter
        
        
        Returns
        ----------
        
        Tensor
            Random Initial Conditions for RWW & BOLD 
        
        
        """
        
        # initial state
        return torch.tensor(0.2 * np.random.uniform(0, 1, (self.node_size, self.state_size)) + np.array(
                [0, 0, 0, 1.0, 1.0, 1.0]), dtype=torch.float32)

    def setModelParameters(self):
        """
        
        A function that assigns model parameters as model attributes and also to assign parameters and hyperparameters for fitting, 
        so that the inherited Torch functionality can be used. 
        This practice may be replaced soon. 

        
        Parameters
        ----------
        
        None
        
        
        Returns
        ----------
        
        Dictionary of Lists
            Keys are State Names and Output Names (with _window appended to the name)
            Contents are the time series from model simulation
        
        """    
    
    
        # set states E I f v mean and 1/sqrt(variance)
        return setModelParameters(self)

    def forward(self, external, hx, hE):
        """
        
        Forward step in simulating the BOLD signal.
        
        Parameters
        ----------
        external: tensor with node_size x steps_per_TR x TRs_per_window x input_size
            noise for states
        
        hx: tensor with node_size x state_size
            states of WWD model
        
        Returns
        -------
        next_state: dictionary with Tensors
            Tensor dimension [Num_Time_Points, Num_Regions]
        
        with keys: 'current_state''bold_window''E_window''I_window''x_window''f_window''v_window''q_window'
            record new states and BOLD
            
        """
        
        return integration_forward(self, external, hx, hE)

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
    param_reg = [] #NMM Equation Parameters
    param_hyper = [] #Mean and Variance of NMM Equation Parameters, and others
    
    if model.use_Gaussian_EI:
        model.E_m = Parameter(torch.tensor(0.16, dtype=torch.float32))
        param_hyper.append(model.E_m)
        model.I_m = Parameter(torch.tensor(0.1, dtype=torch.float32))
        param_hyper.append(model.I_m)
        # model.f_m = Parameter(torch.tensor(1.0, dtype=torch.float32))
        model.v_m = Parameter(torch.tensor(1.0, dtype=torch.float32))
        param_hyper.append(model.v_m)
        # model.x_m = Parameter(torch.tensor(0.16, dtype=torch.float32))
        model.q_m = Parameter(torch.tensor(1.0, dtype=torch.float32))
        param_hyper.append(model.q_m)

        model.E_v_inv = Parameter(torch.tensor(2500, dtype=torch.float32))
        param_hyper.append(model.E_v_inv)
        model.I_v_inv = Parameter(torch.tensor(2500, dtype=torch.float32))
        param_hyper.append(model.I_v_inv)
        # model.f_v = Parameter(torch.tensor(100, dtype=torch.float32))
        model.v_v_inv = Parameter(torch.tensor(100, dtype=torch.float32))
        param_hyper.append(model.v_v_inv)
        # model.x_v = Parameter(torch.tensor(100, dtype=torch.float32))
        model.q_v_inv = Parameter(torch.tensor(100, dtype=torch.float32))
        param_hyper.append(model.v_v_inv)

    # hyper parameters (variables: need to calculate gradient) to fit density
    # of gEI and gIE (the shape from the bifurcation analysis on an isolated node)
    if model.use_Bifurcation:
        model.sup_ca = Parameter(torch.tensor(0.5, dtype=torch.float32))
        param_hyper.append(model.sup_ca)
        model.sup_cb = Parameter(torch.tensor(20, dtype=torch.float32))
        param_hyper.append(model.sup_cb)
        model.sup_cc = Parameter(torch.tensor(10, dtype=torch.float32))
        param_hyper.append(model.sup_cc)

    # set gains_con as Parameter if fit_gain is True
    if model.use_fit_gains:
        model.gains_con = Parameter(torch.tensor(np.zeros((model.node_size, model.node_size)) + 0.05,
                                                 dtype=torch.float32))  # connenction gain to modify empirical sc
        param_reg.append(model.gains_con)
    else:
        model.gains_con = torch.tensor(np.zeros((model.node_size, model.node_size)), dtype=torch.float32)

    var_names = [a for a in dir(model.params) if not a.startswith('__')]
    for var_name in var_names:
        var = getattr(model.params, var_name)
        if (type(var) == par): 
            if (var.fit_hyper == True):
                var.randSet() #TODO: This should be done before giving params to model class
                param_hyper.append(var.prior_mean)
                param_hyper.append(var.prior_var) #TODO: Currently this is _v_inv but should set everything to just variance unless there is a reason to keep the inverse?
                
            if (var.fit_par == True):
                param_reg.append(var.val) #TODO: This should got before fit_hyper, but need to change where randomness gets added in the code first                
                model.track_params.append(var_name)
            
            if (var.fit_par | var.fit_hyper):
                model.track_params.append(var_name) #NMM Parameters

    model.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}

def integration_forward(model, external, hx, hE):

    # Generate the ReLU module for model parameters gEE gEI and gIE
    m = torch.nn.ReLU()


    # Defining NMM Parameters to simplify later equations
    std_in = (0.02 + m(model.params.std_in.value()))  # standard deviation of the Gaussian noise
    std_out = (0.00 + m(model.params.std_out.value()))  # standard deviation of the Gaussian noise
    
    # Parameters for the ODEs
    # Excitatory population
    W_E = model.params.W_E.value()  # scale of the external input
    tau_E = model.params.tau_E.value()  # decay time
    gamma_E = model.params.gamma_E.value()  # other dynamic parameter (?)

    # Inhibitory population
    W_I = model.params.W_I.value()  # scale of the external input
    tau_I = model.params.tau_I.value()  # decay time
    gamma_I = model.params.gamma_I.value()  # other dynamic parameter (?)

    # External input
    I_0 = model.params.I_0.value()  # external input
    I_external = model.params.I_external.value()  # external stimulation

    # Coupling parameters
    g = model.params.g.value()  # global coupling (from all nodes E_j to single node E_i)
    g_EE = (0.001 + m(model.params.g_EE.value()))  # local self excitatory feedback (from E_i to E_i)
    g_IE = (0.001 + m(model.params.g_IE.value()))  # local inhibitory coupling (from I_i to E_i)
    g_EI = (0.001 + m(model.params.g_EI.value()))  # local excitatory coupling (from E_i to I_i)

    aE = model.params.aE.value()
    bE = model.params.bE.value()
    dE = model.params.dE.value()
    aI = model.params.aI.value()
    bI = model.params.bI.value()
    dI = model.params.dI.value()

    # Output (BOLD signal)
    alpha = model.params.alpha.value()
    rho = model.params.rho.value()
    k1 = model.params.k1.value()
    k2 = model.params.k2.value()
    k3 = model.params.k3.value()  # adjust this number from 0.48 for BOLD fluctruate around zero
    V = model.params.V.value()
    E0 = model.params.E0.value()
    tau_s = model.params.tau_s.value()
    tau_f = model.params.tau_f.value()
    tau_0 = model.params.tau_0.value()
    mu = model.params.mu.value()


    next_state = {}

    # hx is current state (6) 0: E 1:I (neural activities) 2:x 3:f 4:v 5:f (BOLD)

    x = hx[:, 2:3]
    f = hx[:, 3:4]
    v = hx[:, 4:5]
    q = hx[:, 5:6]

    dt = torch.tensor(model.step_size, dtype=torch.float32)

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

    # Use the forward model to get neural activity at ith element in the window.
    if model.use_dynamic_boundary:
        for TR_i in range(model.TRs_per_window):

            # Since tr is about second we need to use a small step size like 0.05 to integrate the model states.
            for step_i in range(model.steps_per_TR):
                E = torch.zeros((model.node_size, model.sampling_size))
                I = torch.zeros((model.node_size, model.sampling_size))
                for sample_i in range(model.sampling_size):
                    E[:, sample_i] = E_mean[:, 0] + 0.02 * torch.randn(model.node_size)
                    I[:, sample_i] = I_mean[:, 0] + 0.001 * torch.randn(model.node_size)

                # Calculate the input recurrent.
                IE = torch.tanh(m(W_E * I_0 + g_EE * E + g * torch.matmul(lap_adj, E) - g_IE * I))  # input currents for E
                II = torch.tanh(m(W_I * I_0 + g_EI * E - I))  # input currents for I

                # Calculate the firing rates.
                rE = h_tf(aE, bE, dE, IE)  # firing rate for E
                rI = h_tf(aI, bI, dI, II)  # firing rate for I
                
                # Update the states by step-size 0.05.
                E_next = E + dt * (-E * torch.reciprocal(tau_E) + gamma_E * (1. - E) * rE) \
                         + torch.sqrt(dt) * torch.randn(model.node_size, model.sampling_size) * std_in  ### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)
                I_next = I + dt * (-I * torch.reciprocal(tau_I) + gamma_I * rI) \
                         + torch.sqrt(dt) * torch.randn(model.node_size, model.sampling_size) * std_in

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
                x_next = x + 1 * dt * (1 * E_hist[:, TR_i, step_i][:, np.newaxis] - torch.reciprocal(tau_s) * x \
                         - torch.reciprocal(tau_f) * (f - 1))
                f_next = f + 1 * dt * x
                v_next = v + 1 * dt * (f - torch.pow(v, torch.reciprocal(alpha))) * torch.reciprocal(tau_0)
                q_next = q + 1 * dt * (f * (1 - torch.pow(1 - rho, torch.reciprocal(f))) * torch.reciprocal(rho) \
                         - q * torch.pow(v, torch.reciprocal(alpha)) * torch.reciprocal(v)) * torch.reciprocal(tau_0)

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
            bold_window[:, TR_i] = (std_out * torch.randn(model.node_size, 1) +
                                    100.0 * V * torch.reciprocal(E0) *
                                    (k1 * (1 - q) + k2 * (1 - q * torch.reciprocal(v)) + k3 * (1 - v)))[:, 0]
    else:

        for TR_i in range(model.TRs_per_window):

            # Since tr is about second we need to use a small step size like 0.05 to integrate the model states.
            for step_i in range(model.steps_per_TR):
                E = torch.zeros((model.node_size, model.sampling_size))
                I = torch.zeros((model.node_size, model.sampling_size))
                for sample_i in range(model.sampling_size):
                    E[:, sample_i] = E_mean[:, 0] + 0.001 * torch.randn(model.node_size)
                    I[:, sample_i] = I_mean[:, 0] + 0.001 * torch.randn(model.node_size)

                # Calculate the input recurrent.
                IE = 1 * torch.tanh(m(W_E * I_0 + g_EE * E + g * torch.matmul(lap_adj, E) - g_IE * I))  # input currents for E
                II = 1 * torch.tanh(m(W_I * I_0 + g_EI * E - I))  # input currents for I

                # Calculate the firing rates.
                rE = h_tf(aE, bE, dE, IE)  # firing rate for E
                rI = h_tf(aI, bI, dI, II)  # firing rate for I
                
                # Update the states by step-size dt.
                E_next = E + dt * (-E * torch.reciprocal(tau_E) + gamma_E * (1. - E) * rE) \
                         + torch.sqrt(dt) * torch.randn(model.node_size, model.sampling_size) * std_in  ### equlibrim point at E=(tau_E*gamma_E*rE)/(1+tau_E*gamma_E*rE)
                I_next = I + dt * (-I * torch.reciprocal(tau_I) + gamma_I * rI) \
                         + torch.sqrt(dt) * torch.randn(model.node_size, model.sampling_size) * std_in

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
                x_next = x + 1 * dt * (1 * E_hist[:, TR_i, step_i][:, np.newaxis] - torch.reciprocal(tau_s) * x \
                         - torch.reciprocal(tau_f) * (f - 1))
                f_next = f + 1 * dt * x
                v_next = v + 1 * dt * (f - torch.pow(v, torch.reciprocal(alpha))) * torch.reciprocal(tau_0)
                q_next = q + 1 * dt * (f * (1 - torch.pow(1 - rho, torch.reciprocal(f))) * torch.reciprocal(rho) \
                         - q * torch.pow(v, torch.reciprocal(alpha)) * torch.reciprocal(v)) * torch.reciprocal(tau_0)

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
            bold_window[:, TR_i] = (std_out * torch.randn(model.node_size, 1) +
                                    100.0 * V * torch.reciprocal(E0) * 
                                    (k1 * (1 - q) + k2 * (1 - q * torch.reciprocal(v)) + k3 * (1 - v)))[:, 0]

    # Update the current state.
    current_state = torch.cat([E_mean, I_mean, x, f, v, q], dim=1)
    next_state['current_state'] = current_state
    next_state['bold'] = bold_window
    next_state['E'] = E_hist.reshape((model.node_size, -1))
    next_state['I'] = I_hist.reshape((model.node_size, -1))
    next_state['x'] = x_window
    next_state['f'] = f_window
    next_state['v'] = v_window
    next_state['q'] = q_window

    return next_state, hE