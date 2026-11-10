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
    """Parameters of the :class:`JansenRitModel`.

    Every parameter is a :class:`~whobpyt.datatypes.Parameter`,
    so it can be fixed or learnable and can carry a prior. Pass any
    parameter as a keyword argument to override its default, for example
    ``JansenRitParams(a=par(100, 100, 2, True))``.

    The symbols follow Jansen & Rit (1995). In the equations below, ``P``,
    ``E`` and ``I`` are the pyramidal, excitatory interneuron and inhibitory
    interneuron populations of each brain region.

    Parameters
    ----------
    A : Parameter, default 3.25
        Maximum amplitude of the excitatory postsynaptic potential (mV).
    a : Parameter, default 100
        Excitatory synaptic rate constant (1/s), the inverse of the
        excitatory time constant.
    B : Parameter, default 22
        Maximum amplitude of the inhibitory postsynaptic potential (mV).
    b : Parameter, default 50
        Inhibitory synaptic rate constant (1/s), the inverse of the
        inhibitory time constant.
    c1, c2 : Parameter, default 135 and 108
        Average number of synaptic contacts between the pyramidal cells and
        the excitatory interneurons. ``c1`` scales the pyramidal input to the
        excitatory sigmoid and ``c2`` scales the excitatory interneurons'
        output.
    c3, c4 : Parameter, default 33.75 and 33.75
        Average number of synaptic contacts between the pyramidal cells and
        the inhibitory interneurons, used in the same way as ``c1`` and
        ``c2``.
    vmax : Parameter, default 5
        Maximum firing rate of the sigmoid (1/s).
    v0 : Parameter, default 6
        Membrane potential at which the sigmoid reaches half its maximum
        firing rate (mV).
    r : Parameter, default 0.56
        Steepness of the sigmoid (1/mV).
    g : Parameter, default 1000
        Global coupling gain of the long-range, delayed input from other
        regions to the pyramidal population.
    g_f : Parameter, default 10
        Gain of the long-range input to the excitatory interneurons
        ("feedforward").
    g_b : Parameter, default 10
        Gain of the long-range input to the inhibitory interneurons
        ("feedback").
    std_in : Parameter, default 100
        Standard deviation of the Gaussian noise added to the input of each
        population at every integration step.
    mu : Parameter, default 0.5
        Conduction velocity used to compute the long-range delays,
        ``delay = distance / mu``, in output samples.
    k : Parameter, default 5
        Gain of the external input (stimulus) to the pyramidal population.
    ki : Parameter, default 1
        Weight of the external input for each region, shape
        ``(node_size, 1)``. Set it to zero for regions that should not be
        stimulated.
    cy0 : Parameter, default 5
        Scaling of the simulated M/EEG signal.
    y0 : Parameter, default 2
        Constant offset subtracted from the simulated M/EEG signal.
    lm : Parameter
        Lead field matrix, shape ``(output_size, node_size)``, mapping region
        activity to sensors. It has no default and must be given. It can be
        made learnable.
    k0 : Parameter, optional
        Accepted for compatibility with older code but not used by the
        model.

    Notes
    -----
    The simulated M/EEG signal is ``cy0 * L @ (E - I) - y0``, where ``L`` is
    the lead field ``lm`` normalised per channel and average-referenced.

    References
    ----------
    Jansen BH, Rit VG. Electroencephalogram and visual evoked potential
    generation in a mathematical model of coupled cortical columns.
    Biological Cybernetics 73, 357-366 (1995).
    """
    def __init__(self, **kwargs):
        param = {
            "A": par(3.25), 
            "a": par(100), 
            "B": par(22), 
            "b": par(50), 
            "g": par(1000),
            "g_f": par(10),
            "g_b": par(10),
            
            "c1": par(135), 
            "c2": par(135 * 0.8), 
            "c3": par(135 * 0.25), 
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
    """Connectome-based Jansen-Rit neural mass model of M/EEG.

    Each brain region is a Jansen-Rit neural mass with pyramidal,
    excitatory and inhibitory populations. Regions are coupled through the
    structural connectivity, with conduction delays computed from the
    distance matrix. A lead field maps the activity of every region onto the
    M/EEG sensors. The equations are integrated with the forward Euler
    method in PyTorch, so the simulation is differentiable and its
    parameters can be fitted by gradient descent.

    Parameters
    ----------
    params : JansenRitParams
        The model parameters.
    node_size : int
        Number of brain regions.
    TRs_per_window : int
        Number of output samples simulated per call to :meth:`forward`.
        Training back-propagates through one window at a time.
    step_size : float
        Integration step, in seconds.
    output_size : int
        Number of M/EEG channels. Overridden by ``lm.shape[0]``.
    tr : float
        Interval between output samples, in seconds (for example 0.001 for
        1000 Hz). Must be a multiple of ``step_size``.
    sc : ndarray, shape (node_size, node_size)
        Structural connectivity.
    lm : ndarray, shape (output_size, node_size)
        Lead field matrix.
    dist : ndarray, shape (node_size, node_size)
        Distances between regions, used for the conduction delays.
    use_fit_gains : bool
        If True, learn a separate gain for every connection of the
        pyramidal-to-pyramidal, -excitatory and -inhibitory connectivity
        matrices. If False, the gains are fixed.
    use_laplacian : bool
        If True, add a Laplacian (diagonal) term to the long-range coupling.
    use_fit_lfm : bool
        Whether to fit the lead field matrix. To make the lead field
        learnable, pass ``lm`` in ``params`` as a learnable Parameter.

    Attributes
    ----------
    state_names : list of str
        The model's state variables: ``'E', 'Ev', 'I', 'Iv', 'P', 'Pv'``
        (each population's current and its derivative).
    output_names : list of str
        ``['eeg']``.
    steps_per_TR : int
        Integration steps per output sample, ``int(tr / step_size)``.
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
        Create random initial conditions for the model's states.

        Parameters
        ----------
        ver : int
            Initial condition version. Not used by this model; kept for
            consistency with other models.
        state_lb, state_ub : float
            Lower and upper bound of the uniform random initial values.

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, state_size) with random values between ``state_lb`` and ``state_ub``.
        """

        n_nodes = self.node_size
        n_states = self.state_size
        init_conds = uniform(state_lb, state_ub, (n_nodes, n_states))
        ptinit_conds = pttensor(init_conds, dtype=ptfloat32)
                             
        return ptinit_conds
                            

    def createDelayIC(self, ver, delays_max=500, state_lb=-0.5, state_ub=0.5):
        """
        Create random initial history for the delayed long-range input.

        Parameters
        ----------
        ver : int
            Initial condition version. Not used by this model; kept for
            consistency with other models.
        delays_max : int
            Length of the history buffer, in output samples. Must be longer
            than the longest conduction delay.
        state_lb, state_ub : float
            Lower and upper bound of the uniform random initial values.

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, delays_max) with random values between ``state_lb`` and ``state_ub``.
        """

        n_nodes = self.node_size
        init_delays = uniform(state_lb, state_ub, (n_nodes, delays_max))
        ptinit_delays = pttensor(init_delays, dtype=ptfloat32)
  
        return ptinit_delays


    def setModelSCParameters(self, small_constant=0.05):
        """
        Create the connection gains for the long-range coupling.

        Parameters
        ----------
        small_constant : float
            Initial value of every gain (they enter the model as
            ``exp(gain)``).
        """
        
        # Create the arrays in numpy
        n_nodes = self.node_size
        zsmat = zeros((self.node_size, self.node_size)) + small_constant 
        w_p2e = zsmat.copy() # the pyramidal to excitatory interneuron cross-layer gains
        w_p2i = zsmat.copy() # the pyramidal to inhibitory interneuron cross-layer gains
        w_p2p = zsmat.copy() # the pyramidal to pyramidal cells same-layer gains

        # Set w_p2i, w_p2e, and w_p2p as attributes as type Parameter if use_fit_gains is True;
        # otherwise keep them as fixed (non-learnable) tensors.
        if self.use_fit_gains:
            
            w_p2e = ptParameter(pttensor(w_p2e, dtype=ptfloat32))
            w_p2i = ptParameter(pttensor(w_p2i, dtype=ptfloat32))
            w_p2p = ptParameter(pttensor(w_p2p, dtype=ptfloat32))
            mps = self.params_fitted['modelparameter']
            mps.append(w_p2e); mps.append(w_p2i); mps.append(w_p2p)
        else:
            w_p2e = pttensor(w_p2e, dtype=ptfloat32)
            w_p2i = pttensor(w_p2i, dtype=ptfloat32)
            w_p2p = pttensor(w_p2p, dtype=ptfloat32)

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
        external : torch.Tensor, shape (node_size, steps_per_TR, TRs_per_window)
            External input (stimulus) for this window.
        hx : torch.Tensor, shape (node_size, 6)
            Current state of every region.
        hE : torch.Tensor, shape (node_size, delays_max)
            History of the pyramidal population, used for the delayed
            long-range input.

        Returns
        -------
        next_state : dict
            ``'current_state'`` (the state at the end of the window),
            ``'eeg'`` (shape ``(output_size, TRs_per_window)``) and each
            state variable (shape ``(node_size, TRs_per_window)``).
        hE : torch.Tensor
            The updated history of the pyramidal population.
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

        # Pyramidal to excitatory, pyramidal to inhibitory, and pyramidal to pyramidal connectivity
        # matrices, scaled by the gains w_xx (learnable if use_fit_gains, otherwise fixed).
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

            # For each sample point, run the model by solving the differential 
            # equations for a defined number of integration steps, 
            # and keep only the final activity state within this set of steps 
            for step_i in range(self.steps_per_TR):
                
                
                
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


