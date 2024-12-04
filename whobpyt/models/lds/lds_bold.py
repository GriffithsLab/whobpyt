"""

WhoBPyT Jansen-Rit model classes
---------------------------------

Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak

Neural Mass Model fitting module for Linear with connections E population for BOLD signals

"""


"""
Importage
---------
"""

# PyTorch stuff
from torch.nn.parameter import Parameter as ptParameter
from torch.nn import ReLU as ptReLU
from torch.linalg import norm as ptnorm
from torch import (tensor as pttensor, float32 as ptfloat32, sum as ptsum, exp as ptexp, diag as ptdiag, 
                   transpose as pttranspose, zeros_like as ptzeros_like, int64 as ptint64, randn as ptrandn, 
                   matmul as ptmatmul, tanh as pttanh, matmul as ptmatmul, reshape as ptreshape, sqrt as ptsqrt,
                   ones as ptones, cat as ptcat, pow as ptpow)

# Numpy stuff
from numpy.random import uniform 
from numpy import ones,zeros

# WhoBPyT stuff
from ...datatypes import AbstractNeuralModel, AbstractParams, Parameter as par
from ...functions.arg_type_check import method_arg_type_check



"""
JR params class
---------------
"""

class LDSBOLDParams(AbstractParams):
    """
    A class for setting the parameters of a neural mass model for M/EEG data fitting.

    Attributes:
        tau (par): time constant
        std_in (par): The standard deviation of the input noise.

        
    """
    def __init__(self, **kwargs):
        """
        Initializes the ParamsJR object.

        Args:
            **kwargs: Keyword arguments for the model parameters.

        Returns:
            None
        """
        param = {

            "std_in": par(0.02),  # standard deviation of the Gaussian noise
            "std_out": par(0.02),  # standard deviation of the Gaussian noise
            # Parameters for the ODEs
            
            "tau": par(1.),  # decay time
            
            

            # Coupling parameters
            "g": par(20.),  # global coupling (from all nodes E_j to single node E_i)
            

            
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
            "tau_0": par(0.98)

        }
        
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])



"""
LDS BOLD model class
--------------
"""

class LDSBOLDModel(AbstractNeuralModel):
    """
    A module for forward model (lds + Balloon) to simulate BOLD signals
    
    Attibutes
    ---------
    state_size : int
        Number of states in the JansenRit model


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

    

    use_fit_gains: bool
        Flag for fitting gains. 1: fit, 0: not fit

    

    
    std_in: tensor with gradient on
        Standard deviation for input noise

    params: LDSBOLDParams
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

    def __init__(self, 
                 params: LDSBOLDParams, 
                 node_size=200,
                 TRs_per_window= 20, 
                 step_size=0.05,
                 tr=2, 
                 sc=ones((200,200)), 
                 
                 use_fit_gains=True,
                 use_laplacian=True,
                 use_fit_lfm=False
                 ):               
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
        use_laplacian: bool
            Flat for using laplacian. 1: yes, 0: no. 
        
        params: LDSBOLD
            Model parameters object.
        """
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types
        
        super(LDSBOLDModel, self).__init__(params)
        self.state_names = ['I', 'x', 'f', 'v', 'q']
        self.output_names = ["bold"]
        self.track_params = [] #Is populated during setModelParameters()
        
        self.model_name = "lds_bold"
        self.state_size = 6  # 6 states JR model
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = pttensor(step_size, dtype=ptfloat32)  # integration step 0.1 ms
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.sc = sc  # matrix node_size x node_size structure connectivity
        
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        self.use_laplacian = use_laplacian
        self.params = params
        
        self.setModelParameters()
        self.setModelSCParameters()

    
    
    def createIC(self, ver, state_lb = -0.5, state_ub = 0.5):
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

        n_nodes = self.node_size
        n_states = self.state_size
        init_conds = uniform(state_lb, state_ub, (n_nodes, n_states))
        ptinit_conds = pttensor(init_conds, dtype=ptfloat32)
                             
        return ptinit_conds
                            

    def createDelayIC(self, ver, delays_max=500, state_lb=-0.5, state_ub=0.5):
        """
        Creates the initial conditions for the delays.

        Parameters
        ----------
        ver : int
            Initial condition version. 
            (in the JR model, the version is not used. It is just for consistency with other models)

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, delays_max) with random values between `state_lb` and `state_ub`.
        """

        n_nodes = self.node_size
        init_delays = uniform(state_lb, state_ub, (n_nodes, delays_max))
        ptinit_delays = pttensor(init_delays, dtype=ptfloat32)
  
        return ptinit_delays


    def setModelSCParameters(self, small_constant=0.05):
        """
        Sets the parameters of the model.
        """
        
        # Create the arrays in numpy
        n_nodes = self.node_size
        zsmat = zeros((self.node_size, self.node_size)) + small_constant 
        
        w_p2p = zsmat.copy() # the pyramidal to pyramidal cells same-layer gains

        # Set w_p2i, w_p2e, and w_p2p as attributes as type Parameter if use_fit_gains is True
        if self.use_fit_gains:
            
            
            w_p2p = ptParameter(pttensor(w_p2p, dtype=ptfloat32))
            mps = self.params_fitted['modelparameter']
            mps.append(w_p2p)

        # Add to the current object
        
        self.w_p2p = w_p2p
        


    def forward(self, external, hx, hE):
        """
        This function carries out the forward Euler integration method for the JR neural mass model,
        with time delays, connection gains, and external inputs considered. Each population (pyramidal,
        excitatory, inhibitory) in the network is modeled as a nonlinear second order system. The function
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
        m = ptReLU()
        
        # Define some constants
        con_1 = pttensor(1.0, dtype=ptfloat32) # Define constant 1 tensor
       
        u_2ndsys_ub = 500  # the bound of the input for second order system

        # Defining NMM Parameters to simplify later equations
        #TODO: Change code so that params returns actual value used without extras below
        tau = self.params.tau.value()
        
        g = self.params.g.value()
        
        std_in = self.params.std_in.value() #around 20
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

        next_state = {}

        E = hx[:, 0:1]  # current of exictory population
        

        x = hx[:, 1:2]  
        f = hx[:, 2:3]  
        v = hx[:, 3:4] 
        q = hx[:, 4:5]  
        
        dt = self.step_size

        n_nodes = self.node_size
        n_chans = self.output_size

        sc = self.sc
        ptsc = pttensor(sc, dtype=ptfloat32)

        if self.use_fit_gains:

            # Update the pyramidal to excitatory, pyramidal to inhibitory, and pyramidal to pyramidal connectivity matrices based on the gains w_xx
            
            

            w_l = ptexp(self.w_p2p) * ptsc         
            w_n_l = (0.5 * (w_l + pttranspose(w_l, 0, 1))) / ptnorm(   0.5 * (w_l + pttranspose(w_l, 0, 1)))
            self.sc_p2p = w_n_l


        if self.use_laplacian:
            dg_l = -ptdiag(ptsum(w_n_l, dim=1))


        
        # Placeholder for the updated current state
        current_state = ptzeros_like(hx)

        # Initializing lists for the history of the M/EEG signals, as well as each population's current and voltage.
        E_window   = [];     x_window  = [];  f_window = []; r_window  = [];   q_window = []; 
        bold_window = []; states_window = []

        # Use the model to get M/EEG signal at the i-th element in the window.

        # Run through the number of specified sample points for this window 
        for i_window in range(self.TRs_per_window):
            

            # For each sample point, run the model by solving the differential 
            # equations for a defined number of integration steps, 
            # and keep only the final activity state within this set of steps 
            for step_i in range(self.steps_per_TR):
                
                # Collect the delayed inputs:

                # i) index the history of E
                
                if self.use_laplacian:
                    
                    LEd_p2p =   ptmatmul(w_n_l + dg_l, E)

                # External input (e.g. TMS, sensory)
                u = external[:, step_i:step_i + 1, i_window]
               
                # Stochastic / noise term
               
                E_noise = std_in * ptrandn(n_nodes, 1)
                

                # Compute the firing rate for each neural populatin 
                # at every node using the wave-to-pulse (sigmoid) functino
                

                # Sum the four different input types into a single input value for each neural 
                # populatin state variable
                # The four input types are:
                # - Local      (L)      - from other neural populations within a node (E->P,P->I, etc.)
                # - Long-range (L-R)    - from other nodes in the network, weighted by the long-range 
                #                         connectivity matrices, and time-delayed
                # - Noise      (N)      - stochastic noise input
                # - External   (E)      - external stimulation, eg from TMS or sensory stimulus
                #
                #        L        
                
                rE = g * LEd_p2p  # input currents for E

                
                
                # Compute d/dt   ('_tp1' = state variable at time t+1) 
                
                E_tp1 =  E + dt * (-tau* E + rE)
                x_tp1 = x + 1 * dt * ( E - 1/tau_s * x  - 1/tau_f* (f - 1))
                f_tp1 = f + 1 * dt * x
                v_tp1 = v + 1 * dt * (f - ptpow(v, 1/alpha)) * 1/tau_0
                q_tp1 = q + 1 * dt * (f * (1 - ptpow(1 - rho, 1/f)) * 1/rho \
                         - q * ptpow(v, 1/alpha) * 1/v) * 1/tau_0

                # Add some additional saturation on the model states
                # (for stability and gradient calculation).
                
                E_tp1 = pttanh(E_tp1)
                x_tp1 = 1+ pttanh(x_tp1 -1)
                f_tp1 = 1+ pttanh(f_tp1 -1)
                r_tp1 = 1+ pttanh(r_tp1 -1)
                q_tp1 = 1+ pttanh(q_tp1 -1)
                
                
            
                # Set state variables to currrent values for next round of the loop
                
                E = E_tp1
                x = x_tp1
                f = f_tp1
                r = r_tp1
                q = q_tp1
                # (note - we do this because we aren't (explicitly) keeping the history 
                # by doing something like P[t+1] = P + dt*Pv 
                # because (for the purpose of the paramer estimation) we don't want to 
                # keep the entire integration loop history of P
                #

                # *end 'step_i' loop*

            # Capture the states at the end of every window in the placeholders for checking them visually
            E_window.append(E);    x_window.append(x) ;  f_window.append(f)
            r_window.append(r);  q_window.append(q)
            
            

            

            # Compute BOLD window
            bold_window.append((0.01 * ptrandn(self.node_size, 1) +
                                    100.0 * V * 1/E0*
                                    (k1 * (1 - q) + k2 * (1 - q * 1/v) + k3 * (1 - v))))

            # *end 'i_window' loop

        # Update the current state.
        current_state = ptcat([E, x, f, r, q], dim=1)
        next_state['current_state'] = current_state
        next_state['bold'] = ptcat(bold_window, dim=1)
        next_state['E'] = ptcat(E_window, dim=1)
        next_state['x'] = ptcat(x_window, dim=1)
        next_state['f'] = ptcat(f_window, dim=1)
        next_state['r'] = ptcat(r_window, dim=1)
        next_state['q'] = ptcat(q_window, dim=1)
        


        return next_state, hE


