"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting module for Wong-Wang model
"""

import torch
from torch.nn.parameter import Parameter as ptParameter
from whobpyt.datatypes import AbstractNeuralModel, AbstractParams, Parameter as par
from whobpyt.functions.arg_type_check import method_arg_type_check
import numpy as np # for numerical operations

class ReducedWongWangModel(AbstractNeuralModel):
    """
    Reduced Wong Wang Excitatory Inhibitory (RWWExcInb) Model with integrated BOLD dynamics
    
    A module for forward model (WWD) to simulate a window of BOLD signals
    
    Note that the BOLD signal is not done in the standard way, 
    and there are other customizations to the neural mass model that may 
    deviate from standard differential equation simulation. Thus, the
    parameter should be tested on a validation model after fitting. 
 
 
    Attributes
    ---------
    
    state_names: an array of the list
        An array of list of model state variable names
    pop_names: an array of list
        An array of list of population names
    output_names: list
        A list of model output variable names
    model_name: string
        The name of the model itself
    pop_size : int in this model just one
        The number of population in the WWD model
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
         
    sc: float node_size x node_size array
        The structural connectivity matrix
    sc_fitted: bool
        The fitted structural connectivity
    use_fit_gains: tensor with node_size x node_size (grad on depends on fit_gains)
        Whether to fit the structural connectivity, will fit via connection gains: exp(gains_con)*sc
    
    params: ReducedWongWangParams
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
    

    def __init__(self, params: 'ReducedWongWangParams', node_size = 68, TRs_per_window = 20, step_size = 0.05,  \
                   tr=1.0, sc=np.ones((68,68)), use_fit_gains= True):
        """
        Parameters
        ----------
            
        node_size: int
            The number of ROIs
        TRs_per_window: int
            The number of BOLD TRs to simulate in one forward call    
        step_size: float
            Integration step for forward model
        tr : float
            tr of fMRI image. That is, the spacing betweeen images in the time series. 
        sc: float node_size x node_size array
            The structural connectivity matrix
        use_fit_gains: bool
            Whether to fit the structural connectivity, will fit via connection gains: exp(gains_con)*sc
        params: ReducedWongWangParams
            A object that contains the parameters for the RWW nodes.
        """        
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types
        
        super(ReducedWongWangModel, self).__init__(params)
        
        self.state_names = ['E', 'I', 'x', 'f', 'v', 'q']
        self.output_names = ["bold"]
        
        self.model_name = "RWW"
        self.state_size = 6  # 6 states WWD model
        # self.input_size = input_size  # 1 or 2
        self.tr = tr  # tr fMRI image
        self.step_size = step_size  # integration step 0.05
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.sc_fitted = torch.tensor(sc, dtype=torch.float32)  # placeholder
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        
        self.output_size = node_size
        
        self.setModelParameters()
        self.setModelSCParameters()
    
    
    
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
        state_lb = 0.1

        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size,  delays_max)), dtype=torch.float32)
    
    def setModelSCParameters(self):
        
        """
        Sets the parameters of the model.
        """

        

        # Set w_bb, w_ff, and w_ll as attributes as type Parameter if use_fit_gains is True
        if self.use_fit_gains:
            
            self.w_ll = ptParameter(torch.tensor(np.zeros((self.node_size, self.node_size)) + 0.05, # the lateral gains
                                                dtype=torch.float32))
            self.params_fitted['modelparameter'].append(self.w_ll)
        else:
            self.w_ll = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)
        
    def forward(self, external, hx, hE):
        """
        
        Forward step in simulating the BOLD signal.
        
        Parameters
        ----------
        external: tensor with node_size x pop_size x steps_per_TR x TRs_per_window x input_size
            noise for states
        
        hx: tensor with node_size x state_size
            states of WWD model
        
        Returns
        -------
        next_state: dictionary with Tensors
            Tensor dimension [Num_Time_Points, Num_Regions]
        
        with keys: 'current_state''states_window''bold_window'
            record new states and BOLD
            
        """
        # Generate the ReLU module for model parameters gEE gEI and gIE
        m = torch.nn.ReLU()
    
    
        # Defining NMM Parameters to simplify later equations
        std_in =  self.params.std_in.value()  # 0.02 the lower bound (standard deviation of the Gaussian noise)
        
        # Parameters for the ODEs
        # Excitatory population
        W_E = self.params.W_E.value()  # scale of the external input
        tau_E = self.params.tau_E.value()  # decay time
        gamma_E = self.params.gamma_E.value()  # other dynamic parameter (?)
    
        # Inhibitory population
        W_I = self.params.W_I.value()  # scale of the external input
        tau_I = self.params.tau_I.value()  # decay time
        gamma_I = self.params.gamma_I.value()  # other dynamic parameter (?)
    
        # External input
        I_0 = self.params.I_0.value()  # external input
        I_external = self.params.I_external.value()  # external stimulation
    
        # Coupling parameters
        g = self.params.g.value()  # global coupling (from all nodes E_j to single node E_i)
        g_EE =  self.params.g_EE.value()  # local self excitatory feedback (from E_i to E_i)
        g_IE = self.params.g_IE.value()  # local inhibitory coupling (from I_i to E_i)
        g_EI = self.params.g_EI.value()  # local excitatory coupling (from E_i to I_i)
    
        aE = self.params.aE.value()
        bE = self.params.bE.value()
        dE = self.params.dE.value()
        aI = self.params.aI.value()
        bI = self.params.bI.value()
        dI = self.params.dI.value()
    
        # Output (BOLD signal)
        alpha = self.params.alpha.value()
        rho = self.params.rho.value()
        k1 = self.params.k1.value()
        k2 = self.params.k2.value()
        k3 = self.params.k3.value()  # adjust this number from 0.48 for BOLD fluctruate around zero
        V = self.params.V.value()
        E0 = self.params.E0.value()
        tau_s = self.params.tau_s.value()
        tau_f = self.params.tau_f.value()
        tau_0 = self.params.tau_0.value()
        mu = self.params.mu.value()
    
    
        next_state = {}
    
        # hx is current state (6) 0: E 1:I (neural activities) 2:x 3:f 4:v 5:f (BOLD)
    
        x = hx[:,2:3]
        f = hx[:,3:4]
        v = hx[:,4:5]
        q = hx[:,5:6]
    
        dt = torch.tensor(self.step_size, dtype=torch.float32)
    
        # Update the Laplacian based on the updated connection gains gains_con.
        if self.sc.shape[0] > 1:
    
            # Update the Laplacian based on the updated connection gains gains_con.
            sc_mod = torch.exp(self.w_ll) * torch.tensor(self.sc, dtype=torch.float32)
            sc_mod_normalized = (0.5 * (sc_mod + torch.transpose(sc_mod, 0, 1))) / torch.linalg.norm(
                0.5 * (sc_mod + torch.transpose(sc_mod, 0, 1)))
            self.sc_fitted = sc_mod_normalized
    
            lap_adj = -torch.diag(sc_mod_normalized.sum(1)) + sc_mod_normalized
            
        else:
            lap_adj = torch.tensor(np.zeros((1, 1)), dtype=torch.float32)
    
        
    
        # placeholders for output BOLD, history of E I x f v and q
        # placeholders for output BOLD, history of E I x f v and q
        bold_window = []
        E_window = []
        I_window = []
        x_window = []
        f_window = []
        q_window = []
        v_window = []
    
        
        
        E = hx[:,0:1]
        I = hx[:,1:2]
        #print(E.shape)
        # Use the forward model to get neural activity at ith element in the window.
        
        for TR_i in range(self.TRs_per_window):

            # Since tr is about second we need to use a small step size like 0.05 to integrate the model states.
            for step_i in range(self.steps_per_TR):
                
                # Calculate the input recurrent.
                IE = torch.tanh(m(W_E * I_0 + g_EE * E + g * torch.matmul(lap_adj, E) - g_IE * I))  # input currents for E
                II = torch.tanh(m(W_I * I_0 + g_EI * E - I))  # input currents for I

                # Calculate the firing rates.
                rE = h_tf(aE, bE, dE, IE)  # firing rate for E
                rI = h_tf(aI, bI, dI, II)  # firing rate for I
                
                # Update the states by step-size 0.05.
                E_next = E + dt * (-E * torch.reciprocal(tau_E) + gamma_E * (1. - E) * rE) \
                         + torch.sqrt(dt) * torch.randn(self.node_size, 1) * std_in  
                I_next = I + dt * (-I * torch.reciprocal(tau_I) + gamma_I * rI) \
                         + torch.sqrt(dt) * torch.randn(self.node_size, 1) * std_in

                # Calculate the saturation for model states (for stability and gradient calculation).

                # E_next[E_next>=0.9] = torch.tanh(1.6358*E_next[E_next>=0.9])
                E = torch.tanh(0.0000 + m(1.0 * E_next))
                I = torch.tanh(0.0000 + m(1.0 * I_next))

                

                
                x_next = x + 1 * dt * (1 * E - torch.reciprocal(tau_s) * x \
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
            E_window.append(E)
            I_window.append(I)
            x_window.append(x)
            f_window.append(f)
            v_window.append(v)
            q_window.append(q)

            # Put the BOLD signal each tr to the placeholder being used in the cost calculation.
            bold_window.append((0.01 * torch.randn(self.node_size, 1) +
                                    100.0 * V * torch.reciprocal(E0) *
                                    (k1 * (1 - q) + k2 * (1 - q * torch.reciprocal(v)) + k3 * (1 - v))))
        
    
        # Update the current state.
        current_state = torch.cat([E, I, x,\
                  f, v, q], dim=1)
        next_state['current_state'] = current_state
        next_state['bold'] = torch.cat(bold_window, dim =1)
        next_state['E'] = torch.cat(E_window, dim =1)
        next_state['I'] = torch.cat(I_window, dim =1)
        next_state['x'] = torch.cat(x_window, dim =1)
        next_state['f'] = torch.cat(f_window, dim =1)
        next_state['v'] = torch.cat(v_window, dim =1)
        next_state['q'] = torch.cat(q_window, dim =1)
        
        return next_state, hE
        
        

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


"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for wong-wang model
"""

class ReducedWongWangParams(AbstractParams):
    
    def __init__(self, **kwargs):
        
        param = {

            "std_in": par(0.02),  # standard deviation of the Gaussian noise
            "std_out": par(0.02),  # standard deviation of the Gaussian noise
            # Parameters for the ODEs
            # Excitatory population
            "W_E": par(1.),  # scale of the external input
            "tau_E": par(100.),  # decay time
            "gamma_E": par(0.641 / 1000.),  # other dynamic parameter (?)

            # Inhibitory population
            "W_I": par(0.7),  # scale of the external input
            "tau_I": par(10.),  # decay time
            "gamma_I": par(1. / 1000.),  # other dynamic parameter (?)

            # External input
            "I_0": par(0.32),  # external input
            "I_external": par(0.),  # external stimulation

            # Coupling parameters
            "g": par(20.),  # global coupling (from all nodes E_j to single node E_i)
            "g_EE": par(.1),  # local self excitatory feedback (from E_i to E_i)
            "g_IE": par(.1),  # local inhibitory coupling (from I_i to E_i)
            "g_EI": par(0.1),  # local excitatory coupling (from E_i to I_i)

            "aE": par(310),
            "bE": par(125),
            "dE": par(0.16),
            "aI": par(615),
            "bI": par(177),
            "dI": par(0.087),

            # Output (BOLD signal)
            "alpha": par(0.32),
            "rho": par(0.34),
            "k1": par(2.38),
            "k2": par(2.0),
            "k3": par(0.48),  # adjust this number from 0.48 for BOLD fluctruate around zero
            "V": par(.02),
            "E0": par(0.34),
            "tau_s": par(1 / 0.65),
            "tau_f": par(1 / 0.41),
            "tau_0": par(0.98),
            "mu": par(0.5)

        }

        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])


"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting module for Wong-Wang model
"""

import numpy as np  # for numerical operations
from whobpyt.functions.arg_type_check import method_arg_type_check

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


class ReducedWongWang_np:
    """
    A module for forward model (WWD) to simulate a batch of BOLD signals

    Attibutes
    ---------
    state_size: int
        the number of states in the WWD model
    input_size: int
        the number of states with noise as input
    tr: float
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
                 sc: float, use_dynamic_boundary: bool, use_Laplacian: bool, param: ReducedWongWangParams) -> None:
        """
        Parameters
        ----------

        tr: float
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
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types
        
        super(ReducedWongWang_np, self).__init__()

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
