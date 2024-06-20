"""

WhoBPyT Jansen-Rit model classes
---------------------------------

Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak

Neural Mass Model fitting module for JR with connections from pyramidal to pyramidal, excitatory, and inhibitory populations for M/EEG

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
                   ones as ptones, cat as ptcat)

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

class JansenRitParams(AbstractParams):
    """
    A class for setting the parameters of a neural mass model for M/EEG data fitting.

    Attributes:
        A (par): The amplitude of the EPSP (excitatory post synaptic potential).
        a (par): A metric of the rate constant for the EPSP.
        B (par): The amplitude of the IPSP (inhibitory post synaptic potential).
        b (par): A metric of the rate constant for the IPSP.
        g (par): The gain of ???.
        c1 (par): The connectivity parameter from the pyramidal to excitatory interneurons.
        c2 (par): The connectivity parameter from the excitatory interneurons to the pyramidal cells.
        c3 (par): The connectivity parameter from the pyramidal to inhibitory interneurons.
        c4 (par): The connectivity parameter from the inhibitory interneurons to the pyramidal cells.
        std_in (par): The standard deviation of the input noise.
        vmax (par): The maximum value of the sigmoid function.
        v0 (par): The midpoint of the sigmoid function.
        r (par): The slope of the sigmoid function.
        y0 (par): ???.
        mu (par): The mean of the input.
        k (par): ???.
        cy0 (par): ???.
        ki (par): ???.
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
            "A": par(3.25), 
            "a": par(100), 
            "B": par(22), 
            "b": par(50), 
            "g": par(1000),
            
            "c1": par(135), 
            "c2": par(135 * 0.8), 
            "c3 ": par(135 * 0.25), 
            "c4": par(135 * 0.25),
            
            "std_in": par(100), 
            "vmax": par(5), 
            "v0": par(6), 
            "r": par(0.56), 
            "y0": par(2),
            
            "mu": par(.5), 
            "k": par(5), 
            "cy0": par(5), 
            "ki": par(1)
        }
        
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])



"""
JR model class
--------------
"""

class JansenRitModel(AbstractNeuralModel):
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

    def __init__(self, 
                 params: JansenRitParams, 
                 node_size=200,
                 TRs_per_window= 20, 
                 step_size=0.0001, 
                 output_size=64, 
                 tr=0.001, 
                 sc=ones((200,200)), 
                 lm=ones((64,200)), 
                 dist=ones((200,200)),
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
        use_fit_lfm: bool
            Flag for fitting the leadfield matrix. 1: fit, 0: not fit
        params: ParamsJR
            Model parameters object.
        """
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types
        
        super(JansenRitModel, self).__init__(params)
        self.state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
        self.output_names = ["eeg"]
        self.track_params = [] #Is populated during setModelParameters()
        
        self.model_name = "JR"
        self.state_size = 6  # 6 states JR model
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = pttensor(step_size, dtype=ptfloat32)  # integration step 0.1 ms
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.output_size = output_size  # num of EEG channels
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.dist = pttensor(dist, dtype=ptfloat32)
        self.lm = lm
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        self.use_laplacian = use_laplacian
        self.use_fit_lfm = use_fit_lfm
        self.params = params
        self.output_size = lm.shape[0]  # number of EEG channels
        
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
        w_p2e = zsmat.copy() # the pyramidal to excitatory interneuron cross-layer gains
        w_p2i = zsmat.copy() # the pyramidal to inhibitory interneuron cross-layer gains
        w_p2p = zsmat.copy() # the pyramidal to pyramidal cells same-layer gains

        # Set w_p2i, w_p2e, and w_p2p as attributes as type Parameter if use_fit_gains is True
        if self.use_fit_gains:
            
            w_p2e = ptParameter(pttensor(w_p2e, dtype=ptfloat32))
            w_p2i = ptParameter(pttensor(w_p2i, dtype=ptfloat32))
            w_p2p = ptParameter(pttensor(w_p2p, dtype=ptfloat32))
            mps = self.params_fitted['modelparameter']
            mps.append(w_p2e); mps.append(w_p2i); mps.append(w_p2p)

        # Add to the current object
        self.w_p2e = w_p2e
        self.w_p2i = w_p2i
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
        A = self.params.A.value()
        a = self.params.a.value()
        B = self.params.B.value()
        b = self.params.b.value()
        g = self.params.g.value()
        c1 = self.params.c1.value()
        c2 = self.params.c2.value()
        c3 = self.params.c3.value()
        c4 = self.params.c4.value()
        std_in = self.params.std_in.value() #around 20
        vmax = self.params.vmax.value()
        v0 = self.params.v0.value()
        r = self.params.r.value()
        y0 = self.params.y0.value()
        mu = self.params.mu.value()
        k =  self.params.k.value()
        cy0 = self.params.cy0.value()
        ki = self.params.ki.value()

        g_f = self.params.g_f.value()
        g_b = self.params.g_b.value()
        lm = self.params.lm.value()

        next_state = {}

        P = hx[:, 0:1]  # current of pyramidal population
        E = hx[:, 1:2]  # current of excitory population
        I = hx[:, 2:3]  # current of inhibitory population

        Pv = hx[:, 3:4]  # voltage of pyramidal population
        Ev = hx[:, 4:5]  # voltage of exictory population
        Iv = hx[:, 5:6]  # voltage of inhibitory population
        
        dt = self.step_size

        n_nodes = self.node_size
        n_chans = self.output_size

        sc = self.sc
        ptsc = pttensor(sc, dtype=ptfloat32)

        if self.use_fit_gains:

            # Update the pyramidal to excitatory, pyramidal to inhibitory, and pyramidal to pyramidal connectivity matrices based on the gains w_xx
            
            w_b = ptexp(self.w_p2i) * ptsc
            w_n_b = w_b / ptnorm(w_b)
            self.sc_p2i = w_n_b

            w_f = ptexp(self.w_p2e) * ptsc     
            w_n_f = w_f / ptnorm(w_f)
            self.sc_p2e = w_n_f

            w_l = ptexp(self.w_p2p) * ptsc         
            w_n_l = (0.5 * (w_l + pttranspose(w_l, 0, 1))) / ptnorm(   0.5 * (w_l + pttranspose(w_l, 0, 1)))
            self.sc_p2p = w_n_l


        if self.use_laplacian:
            dg_b = -ptdiag(ptsum(w_n_b, dim=1))
            dg_l = -ptdiag(ptsum(w_n_l, dim=1))
            dg_f = -ptdiag(ptsum(w_n_f, dim=1))


        self.delays = (self.dist / mu).type(ptint64)

        # Placeholder for the updated current state
        current_state = ptzeros_like(hx)

        # Initializing lists for the history of the M/EEG signals, as well as each population's current and voltage.
        E_window   = [];     I_window  = [];  P_window = [];
        Ev_window  = [];     Iv_window = []; Pv_window = [];
        eeg_window = []; states_window = [];

        # Use the model to get M/EEG signal at the i-th element in the window.

        # Run through the number of specified sample points for this window 
        for i_window in range(self.TRs_per_window):
            

            # For each sample point, run the model by solving the differential 
            # equations for a defined number of integration steps, 
            # and keep only the final activity state within this set of steps 
            for step_i in range(self.steps_per_TR):
                
                # Collect the delayed inputs:

                # i) index the history of E
                Ed = pttranspose(hE.clone().gather(1,self.delays), 0, 1)

                # ii) multiply the past states by the connectivity weights matrix, and sum over rows
                LEd_p2e =  ptsum(w_n_f * Ed, 1)
                LEd_p2i = -ptsum(w_n_b * Ed, 1)
                LEd_p2p =  ptsum(w_n_l * Ed, 1)
                
                # iii) reshape for next step
                LEd_p2e = ptreshape(LEd_p2e, (n_nodes, 1))
                LEd_p2i = ptreshape(LEd_p2i, (n_nodes, 1))
                LEd_p2p = ptreshape(LEd_p2p, (n_nodes, 1))
                
                # iv) if specified, add the laplacian component (self-connections from diagonals)
                if self.use_laplacian:
                    LEd_p2e =  LEd_p2e + ptmatmul(dg_f, E - I)
                    LEd_p2i =  LEd_p2i - ptmatmul(dg_b, E - I)
                    LEd_p2p =  LEd_p2p + ptmatmul(dg_l, P)

                # External input (e.g. TMS, sensory)
                u = external[:, step_i:step_i + 1, i_window]
               
                # Stochastic / noise term
                P_noise = std_in * ptrandn(n_nodes, 1) 
                E_noise = std_in * ptrandn(n_nodes, 1)
                I_noise = std_in * ptrandn(n_nodes, 1)

                # Compute the firing rate for each neural populatin 
                # at every node using the wave-to-pulse (sigmoid) functino
                # (vmax = max value of sigmoid, v0 = midpoint of sigmoid)
                P_sigm = vmax / ( 1 + ptexp ( r*(v0 -  (E-I) ) ) )
                E_sigm = vmax / ( 1 + ptexp ( r*(v0 - (c1*P) ) ) )
                I_sigm = vmax / ( 1 + ptexp ( r*(v0 - (c3*P) ) ) )

                # Sum the four different input types into a single input value for each neural 
                # populatin state variable
                # The four input types are:
                # - Local      (L)      - from other neural populations within a node (E->P,P->I, etc.)
                # - Long-range (L-R)    - from other nodes in the network, weighted by the long-range 
                #                         connectivity matrices, and time-delayed
                # - Noise      (N)      - stochastic noise input
                # - External   (E)      - external stimulation, eg from TMS or sensory stimulus
                #
                #        Local    Long-range   Noise   External
                rP =     P_sigm  + g*LEd_p2p   + P_noise + k*ki*u 
                rE =  c2*E_sigm  + g_f*LEd_p2e + E_noise          
                rI =  c4*I_sigm  + g_b*LEd_p2i + I_noise          

                # Apply some additional scaling
                rP = u_2ndsys_ub * pttanh(rP / u_2ndsys_ub)
                rE = u_2ndsys_ub * pttanh(rE / u_2ndsys_ub)
                rI = u_2ndsys_ub * pttanh(rI / u_2ndsys_ub)
                
                # Compute d/dt   ('_tp1' = state variable at time t+1) 
                P_tp1 =  P + dt * Pv
                E_tp1 =  E + dt * Ev
                I_tp1 =  I + dt * Iv
                Pv_tp1 = Pv + dt * ( A*a*rP  -  2*a*Pv  -  a**2 * P )
                Ev_tp1 = Ev + dt * ( A*a*rE  -  2*a*Ev  -  a**2 * E )
                Iv_tp1 = Iv + dt * ( B*b*rI  -  2*b*Iv  -  b**2 * I )

                # Add some additional saturation on the model states
                # (for stability and gradient calculation).
                P_tp1 = 1000*pttanh(P_tp1/1000)
                E_tp1 = 1000*pttanh(E_tp1/1000)
                I_tp1 = 1000*pttanh(I_tp1/1000)
                Pv_tp1 = 1000*pttanh(Pv_tp1/1000)
                Ev_tp1 = 1000*pttanh(Ev_tp1/1000)
                Iv_tp1 = 1000*pttanh(Iv_tp1/1000)
                
                # Update placeholders for pyramidal buffer
                hE[:, 0] = P_tp1[:, 0]
            
                # Set state variables to currrent values for next round of the loop
                P = P_tp1
                E = E_tp1
                I = I_tp1
                Pv = Pv_tp1
                Ev = Ev_tp1
                Iv = Iv_tp1
                # (note - we do this because we aren't (explicitly) keeping the history 
                # by doing something like P[t+1] = P + dt*Pv 
                # because (for the purpose of the paramer estimation) we don't want to 
                # keep the entire integration loop history of P
                #

                # *end 'step_i' loop*

            # Capture the states at the end of every window in the placeholders for checking them visually
            P_window.append(P);    I_window.append(I) ;  E_window.append(E)
            Pv_window.append(Pv);  Iv_window.append(Iv); Ev_window.append(Ev)
            
            # Capture the states at every tr in the placeholders for checking them visually.
            hE = ptcat([P, hE[:, :-1]], dim=1)  # update placeholders for pyramidal buffer

            # Lead field matrix
            onesmat = ptones(1,n_chans)
            lm_t = (lm.T / ptsqrt((lm ** 2).sum(1))).T
            self.lm_t = (lm_t - 1 / n_chans * ptmatmul(onesmat, lm_t))

            # Compute M/EEG window
            temp = cy0 * ptmatmul(self.lm_t, E-I) - 1 * y0
            eeg_window.append(temp)

            # *end 'i_window' loop

        # Update the current state.
        current_state = ptcat([P, E, I, Pv, Ev, Iv], dim=1)
        next_state['current_state'] = current_state
        next_state['eeg'] = ptcat(eeg_window, dim=1)
        next_state['E'] = ptcat(E_window, dim=1)
        next_state['I'] = ptcat(I_window, dim=1)
        next_state['P'] = ptcat(P_window, dim=1)
        next_state['Ev'] = ptcat(Ev_window, dim=1)
        next_state['Iv'] = ptcat(Iv_window, dim=1)
        next_state['Pv'] = ptcat(Pv_window, dim=1)


        return next_state, hE


