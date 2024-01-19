"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Clemens Pellengahr, Hussain Ather, Davide Momi, Sorenza Bastiaens, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita
Neural Mass Model fitting module for Wong-Wang model
"""

import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes import AbstractNMM, AbstractParams, par
from whobpyt.models.RWW import ParamsRWW
from whobpyt.functions.arg_type_check import method_arg_type_check
import numpy as np # for numerical operations

class RNNRWWMM(AbstractNMM):
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
    

    def __init__(self, params: ParamsRWW, node_size = 68, output_size = 64, TRs_per_window = 20, step_size = 0.1,  \
                   tr=1.0, tr_eeg= 0.001, sc=np.ones((68,68)), use_fit_gains= True):
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
        
        super(RNNRWWMM, self).__init__(params)
        
        self.state_names = ['E', 'I', 'x', 'f', 'v', 'q']
        self.output_names = ["bold"]
        self.track_params = [] #Is populated during setModelParameters()
        self.pop_names =['E']
        self.pop_size = 1
        self.model_name = "RWWMM"
        self.state_size = 6  # 6 states WWD model
        # self.input_size = input_size  # 1 or 2
        self.tr = tr  # tr fMRI image
        self.tr_eeg = tr_eeg  # tr fMRI image
        self.step_size = step_size  # integration step 0.05
        
        self.steps_per_TR = int(tr/ step_size)
        
        self.steps_per_TR_eeg = int(tr_eeg/ step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.sc_fitted = torch.tensor(sc, dtype=torch.float32)  # placeholder
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        
        self.params = params
        
        self.params_fitted = {}

        self.output_size = output_size
        
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
        return torch.tensor(0.2 * np.random.uniform(0, 1, (self.node_size, self.pop_size, self.state_size)) + np.array(
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
            
            self.w_ll = Parameter(torch.tensor(np.zeros((self.node_size, self.node_size)) + 0.05, # the lateral gains
                                                dtype=torch.float32))
            self.params_fitted['modelparameter'].append(self.w_ll)
        else:
            self.w_ll = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)
        
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
        # Generate the ReLU module for model parameters gEE gEI and gIE
        m = torch.nn.ReLU()
    
    
        # Defining NMM Parameters to simplify later equations
        std_in =  0.02+m(self.params.std_in.value())  # standard deviation of the Gaussian noise
        
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
        g_EE =  m(self.params.g_EE.value())  # local self excitatory feedback (from E_i to E_i)
        g_IE = m(self.params.g_IE.value())  # local inhibitory coupling (from I_i to E_i)
        g_EI = m(self.params.g_EI.value())  # local excitatory coupling (from E_i to I_i)
    
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
        lm = self.params.lm.value()
    
    
        next_state = {}
    
        # hx is current state (6) 0: E 1:I (neural activities) 2:x 3:f 4:v 5:f (BOLD)
    
        x = hx[:,:,2]
        f = hx[:,:,3]
        v = hx[:,:,4]
        q = hx[:,:,5]
    
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
    
        
        E = hx[:,:,0]
        I = hx[:,:,1]
        #print(E.shape)
        # Use the forward model to get neural activity at ith element in the window.
        
        for TR_i in range(self.TRs_per_window):
            E_holder = []
            for step_i in range(self.steps_per_TR):
                
            
                
                # Calculate the input recurrent.
                IE = torch.tanh(m(W_E * I_0 + g_EE * E + g * torch.matmul(lap_adj, E) - g_IE * I))  # input currents for E
                II = torch.tanh(m(W_I * I_0 + g_EI * E - I))  # input currents for I
    
                # Calculate the firing rates.
                rE = h_tf(aE, bE, dE, IE)  # firing rate for E
                rI = h_tf(aI, bI, dI, II)  # firing rate for I
                
                # Update the states by step-size 0.05.
                E_next = E + dt * (-E * torch.reciprocal(tau_E) + gamma_E * (1. - E) * rE) \
                         + torch.sqrt(dt) * torch.randn(self.node_size, self.pop_size) * std_in  
                I_next = I + dt * (-I * torch.reciprocal(tau_I) + gamma_I * rI) \
                         + torch.sqrt(dt) * torch.randn(self.node_size, self.pop_size) * std_in
    
                # Calculate the saturation for model states (for stability and gradient calculation).
    
                # E_next[E_next>=0.9] = torch.tanh(1.6358*E_next[E_next>=0.9])
                E = torch.tanh(0.0000 + m(1.0 * E_next))
                I = torch.tanh(0.0000 + m(1.0 * I_next))
                
                if (step_i+1) % self.steps_per_TR_eeg == 0:
                    E_window.append(torch.matmul(lm, E))
                
                    
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
            
            
            # Put the BOLD signal each tr to the placeholder being used in the cost calculation.
            bold_window.append((0.01 * torch.randn(self.node_size, 1) +
                                    100.0 * V * torch.reciprocal(E0) *
                                    (k1 * (1 - q) + k2 * (1 - q * torch.reciprocal(v)) + k3 * (1 - v))))
        
    
        # Update the current state.
        current_state = torch.cat([E[:,:, np.newaxis], I[:,:, np.newaxis], x[:,:, np.newaxis],\
                  f[:,:, np.newaxis], v[:,:, np.newaxis], q[:,:, np.newaxis]], dim=2)
        next_state['current_state'] = current_state
        next_state['bold'] = torch.cat(bold_window, dim =1)
        next_state['states'] = torch.cat(E_window, dim =1)
        
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





    