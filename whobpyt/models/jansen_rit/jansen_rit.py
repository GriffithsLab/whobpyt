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
    A class for setting the parameters of the Jansen-Rit neural mass model for M/EEG data fitting.

    Default values are provided for all standard parameters. Additional parameters
    (such as ``g_f``, ``g_b``, and ``lm``) must be supplied via ``**kwargs`` as needed.

    Attributes
    ----------
    A : par
        Amplitude of the excitatory post-synaptic potential (EPSP), in mV.
    a : par
        Rate constant for the EPSP, controlling the rise and decay of excitatory
        post-synaptic responses, in s^-1.
    B : par
        Amplitude of the inhibitory post-synaptic potential (IPSP), in mV.
    b : par
        Rate constant for the IPSP, controlling the rise and decay of inhibitory
        post-synaptic responses, in s^-1.
    g : par
        Gain scaling factor for the long-range pyramidal-to-pyramidal (P-to-P) connectivity.
    c1 : par
        Connectivity constant from pyramidal cells to excitatory interneurons.
    c2 : par
        Connectivity constant from excitatory interneurons back to pyramidal cells.
    c3 : par
        Connectivity constant from pyramidal cells to inhibitory interneurons.
    c4 : par
        Connectivity constant from inhibitory interneurons back to pyramidal cells.
    std_in : par
        Standard deviation of the stochastic (noise) input to each neural population.
    vmax : par
        Maximum firing rate of the sigmoid (wave-to-pulse) transfer function, in s^-1.
    v0 : par
        Midpoint membrane potential of the sigmoid transfer function, in mV.
    r : par
        Steepness (slope) of the sigmoid transfer function, in mV^-1.
    y0 : par
        Baseline offset subtracted from the EEG output signal.
    mu : par
        Scaling factor used together with the inter-node distance matrix to compute
        propagation time delays between nodes.
    k : par
        Scaling factor for the external input to the pyramidal population.
    cy0 : par
        Scaling factor applied to the leadfield-projected source signal for EEG output.
    ki : par
        Secondary scaling factor for the external input, used together with ``k``
        (effective external input = ``k * ki * u``).
    g_f : par
        Gain scaling factor for the long-range pyramidal-to-excitatory interneuron
        (P-to-E) forward connectivity. Must be provided via ``**kwargs``.
    g_b : par
        Gain scaling factor for the long-range pyramidal-to-inhibitory interneuron
        (P-to-I) backward connectivity. Must be provided via ``**kwargs``.
    lm : par
        Leadfield matrix parameter mapping source space to EEG channel space.
        Must be provided via ``**kwargs`` when ``use_fit_lfm=True``.
    """
    def __init__(self, **kwargs):
        """
        Initialize a JansenRitParams object with default parameter values.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments to override default parameter values or supply additional
            parameters such as ``g_f``, ``g_b``, and ``lm``. Each value should be an
            instance of :class:`~whobpyt.datatypes.Parameter`.

        Returns
        -------
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
    A PyTorch module implementing the Jansen-Rit neural mass model for simulating M/EEG signals.

    The model simulates interactions between three neural populations per region of interest
    (ROI): pyramidal cells (P), excitatory interneurons (E), and inhibitory interneurons (I).
    Long-range connectivity with time delays, stochastic input, Laplacian regularization,
    and optional leadfield-based EEG projection are supported.

    Attributes
    ----------
    state_size : int
        Number of state variables in the model (6: P, E, I and their velocities Pv, Ev, Iv).
    output_size : int
        Number of EEG output channels, determined by ``lm.shape[0]``.
    node_size : int
        Number of regions of interest (ROIs) in the network.
    step_size : torch.Tensor
        Integration step size (in seconds) for the forward Euler ODE solver.
    steps_per_TR : int
        Number of ODE integration steps per sampling interval (``tr / step_size``).
    tr : float
        Sampling interval (in seconds) of the simulated EEG signals.
    TRs_per_window : int
        Number of EEG time points to simulate per forward pass window.
    sc : numpy.ndarray
        Structural connectivity matrix of shape (node_size, node_size).
    lm : numpy.ndarray
        Leadfield matrix mapping source space to EEG channel space,
        shape (output_size, node_size).
    dist : torch.Tensor
        Inter-node distance matrix of shape (node_size, node_size),
        used for computing propagation time delays.
    use_fit_gains : bool
        If ``True``, the long-range connectivity gain matrices (w_p2e, w_p2i, w_p2p)
        are treated as trainable parameters.
    use_laplacian : bool
        If ``True``, a Laplacian regularization term is applied to the connectivity matrices.
    use_fit_lfm : bool
        If ``True``, the leadfield matrix is treated as a trainable parameter.
    w_p2e : torch.nn.Parameter or numpy.ndarray
        Gain matrix for pyramidal-to-excitatory interneuron long-range connections.
    w_p2i : torch.nn.Parameter or numpy.ndarray
        Gain matrix for pyramidal-to-inhibitory interneuron long-range connections.
    w_p2p : torch.nn.Parameter or numpy.ndarray
        Gain matrix for pyramidal-to-pyramidal long-range connections.
    params : JansenRitParams
        Object containing all model parameters.

    Methods
    -------
    createIC(ver, state_lb=-0.5, state_ub=0.5)
        Create random initial conditions for each node and state variable.
    createDelayIC(ver, delays_max=500, state_lb=-0.5, state_ub=0.5)
        Create a random initial delay history buffer for all nodes.
    setModelSCParameters(small_constant=0.05)
        Initialize the structural connectivity gain matrices.
    forward(external, hx, hE)
        Run the model simulation for one window and return updated states and EEG output.
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
        Construct a JansenRitModel instance.

        Parameters
        ----------
        params : JansenRitParams
            Object containing the model parameters (see :class:`JansenRitParams`).
        node_size : int, optional
            Number of ROIs (regions of interest) in the network. Default is 200.
        TRs_per_window : int, optional
            Number of EEG time points to simulate per forward pass window. Default is 20.
        step_size : float, optional
            ODE integration step size in seconds. Default is 0.0001.
        output_size : int, optional
            Number of EEG output channels. This value is overridden by ``lm.shape[0]``.
            Default is 64.
        tr : float, optional
            Sampling interval in seconds for the simulated EEG signals. Default is 0.001.
        sc : numpy.ndarray, optional
            Structural connectivity matrix of shape (node_size, node_size).
            Default is a (200, 200) matrix of ones.
        lm : numpy.ndarray, optional
            Leadfield matrix of shape (output_size, node_size) mapping source space
            to EEG channel space. Default is a (64, 200) matrix of ones.
        dist : numpy.ndarray, optional
            Inter-node distance matrix of shape (node_size, node_size) used for
            computing propagation time delays. Default is a (200, 200) matrix of ones.
        use_fit_gains : bool, optional
            If ``True``, the long-range connectivity gain matrices are set as trainable
            parameters. Default is True.
        use_laplacian : bool, optional
            If ``True``, a Laplacian regularization term is applied to the connectivity
            matrices during the forward pass. Default is True.
        use_fit_lfm : bool, optional
            If ``True``, the leadfield matrix is treated as a trainable parameter.
            Default is False.
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
        Create random initial conditions for the model state variables.

        Generates a tensor of uniformly distributed random values within the
        specified bounds to use as the initial neural state for each node.

        Parameters
        ----------
        ver : int
            Version indicator. Not used in the JR model; included for consistency
            with the :class:`~whobpyt.datatypes.AbstractNeuralModel` interface.
        state_lb : float, optional
            Lower bound for the uniform distribution used to initialize state values.
            Default is -0.5.
        state_ub : float, optional
            Upper bound for the uniform distribution used to initialize state values.
            Default is 0.5.

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, state_size) with random initial values
            drawn from a uniform distribution on [``state_lb``, ``state_ub``].
        """

        n_nodes = self.node_size
        n_states = self.state_size
        init_conds = uniform(state_lb, state_ub, (n_nodes, n_states))
        ptinit_conds = pttensor(init_conds, dtype=ptfloat32)
                             
        return ptinit_conds
                            

    def createDelayIC(self, ver, delays_max=500, state_lb=-0.5, state_ub=0.5):
        """
        Create a random initial delay history buffer for all nodes.

        Generates a tensor of uniformly distributed random values to initialize
        the time-delay history of the pyramidal population state across all nodes.

        Parameters
        ----------
        ver : int
            Version indicator. Not used in the JR model; included for consistency
            with the :class:`~whobpyt.datatypes.AbstractNeuralModel` interface.
        delays_max : int, optional
            Maximum number of past time steps to store in the delay buffer.
            Default is 500.
        state_lb : float, optional
            Lower bound for the uniform distribution used to initialize delay history values.
            Default is -0.5.
        state_ub : float, optional
            Upper bound for the uniform distribution used to initialize delay history values.
            Default is 0.5.

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, delays_max) with random initial values
            drawn from a uniform distribution on [``state_lb``, ``state_ub``].
        """

        n_nodes = self.node_size
        init_delays = uniform(state_lb, state_ub, (n_nodes, delays_max))
        ptinit_delays = pttensor(init_delays, dtype=ptfloat32)
  
        return ptinit_delays


    def setModelSCParameters(self, small_constant=0.05):
        """
        Initialize the structural connectivity gain matrices for long-range connections.

        Creates three gain matrices representing the three types of long-range connections:
        pyramidal-to-excitatory (w_p2e), pyramidal-to-inhibitory (w_p2i), and
        pyramidal-to-pyramidal (w_p2p). Each matrix is initialized to ``small_constant``
        for all entries. If ``use_fit_gains`` is ``True``, the matrices are registered
        as trainable PyTorch parameters and appended to the model parameter list.

        Parameters
        ----------
        small_constant : float, optional
            Initial value for all entries in the gain matrices. Default is 0.05.

        Notes
        -----
        This method is called automatically during :meth:`__init__` and does not
        normally need to be called again.
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
        Run the Jansen-Rit model simulation for one window of time steps.

        Performs forward Euler integration of the JR neural mass model ODEs across
        ``TRs_per_window`` sampling intervals, each subdivided into ``steps_per_TR``
        integration steps. At each step, the firing rates of pyramidal (P), excitatory (E),
        and inhibitory (I) populations are computed via a sigmoid transfer function.
        Long-range delayed inputs from other nodes, stochastic noise, and optional external
        stimulation (e.g., TMS or sensory input) are incorporated. EEG signals are computed
        at each sampling interval by projecting the source activity through the leadfield matrix.

        Parameters
        ----------
        external : torch.Tensor
            External stimulation input of shape (node_size, steps_per_TR, TRs_per_window),
            representing time-varying inputs (e.g., TMS pulses or sensory stimuli) at each node.
        hx : torch.Tensor
            Current neural state tensor of shape (node_size, state_size), where the state
            variables are ordered as [P, E, I, Pv, Ev, Iv] (currents and their velocities
            for the pyramidal, excitatory, and inhibitory populations respectively).
        hE : torch.Tensor
            Delay history buffer of shape (node_size, delays_max), containing past values
            of the pyramidal population current used for computing delayed long-range inputs.

        Returns
        -------
        next_state : dict
            Dictionary with the following keys:

            - ``'current_state'`` (torch.Tensor): Updated state tensor of shape
              (node_size, state_size) at the end of the window.
            - ``'eeg'`` (torch.Tensor): Simulated EEG signals of shape
              (output_size, TRs_per_window).
            - ``'E'``, ``'I'``, ``'P'`` (torch.Tensor): Excitatory, inhibitory, and
              pyramidal population currents across the window, each of shape
              (node_size, TRs_per_window).
            - ``'Ev'``, ``'Iv'``, ``'Pv'`` (torch.Tensor): Velocity state variables
              for each population across the window, each of shape
              (node_size, TRs_per_window).

        hE : torch.Tensor
            Updated delay history buffer of shape (node_size, delays_max), shifted to
            include the most recent pyramidal population current values.
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


