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
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
from whobpyt.datatypes import AbstractNMM, par
from whobpyt.models.CBNet.ParamsCBNet import ParamsCBnet
from whobpyt.functions.arg_type_check import method_arg_type_check
import numpy as np


class RNNCBNET(torch.nn.Module):
    """
    A module for forward model (Conductance based model) to simulate EEG signals

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
    state_names = ['states']
    model_name = "cb_net"
    output_names = ['eeg']

    def __init__(self, node_size: int, pop_size: int,
                 TRs_per_window: int, step_size: float, output_size: int, tr: float, sc: float,  lm: float, dist: float,
                 use_fit_gains: bool,  param: ParamsCBnet) -> None:
        """
        Parameters
        ----------

        tr : float
            tr of image
        step_size: float
            Integration step for forward model

        TRs_per_window: int
            the number of EEG signals to simulate
        node_size: int
            the number of ROIs
        output_size: int
            the number of channels EEG
        sc: float node_size x node_size array
            structural connectivity
        use_fit_gains: bool
            flag for fitting gains 1: fit 0: not fit
        use_fit_lfm: bool
            flag for fitting gains 1: fit 0: not fit
        param from ParamJR
        """

        super(RNNCBNET, self).__init__()
        self.state_size = 4  # 4 states CB_net model
        self.pop_size =pop_size
        if pop_size == 3:
            self.pop_names = np.array(['E', 'I', 'P'])
        elif pop_size == 4:
            self.pop_names = np.array(['P', 'I', 'E', 'DP'])
        self.track_params = [] #Is populated during setModelParameters()
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = torch.tensor(step_size, dtype=torch.float32)  # integration step 0.1 ms
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.output_size = output_size  # num of EEG channels
        self.sc = sc  # matrix node_size x node_size structure connectivity

        self.dist = torch.tensor(dist, dtype=torch.float32)
        #self.lm = lm
        self.use_fit_gains = use_fit_gains  # flag for fitting gains
        #self.use_fit_lfm = use_fit_lfm
        self.params = param

        self.output_size = lm.shape[0]  # number of EEG channels
        #set variables
        self.setModelParameters()

    def m_nmda(self,alpha, v):

        m = 1.0/(1 + 0.2*torch.exp(-alpha*v))
        return m

    def info(self):
        # TODO: Make sure this method is useful
        """
        Returns a dictionary with the names of the states and the output.

        Returns
        -------
        Dict[str, List[str]]
        """

        return {"pop_names": self.pop_names, "state_names": self.state_names, "output_names": self.output_names}

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

        return torch.tensor(np.random.uniform(-1, 1, (self.node_size, self.pop_size, self.state_size))\
                             + np.array([-60, 64, 64, 64]), dtype=torch.float32)

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
        state_ub = 0.2
        state_lb = -0.2

        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size,  delays_max)), dtype=torch.float32)

    def setModelParameters(self):
        """
        Sets the parameters of the model.
        """

        param_reg = []
        param_hyper = []

        # set w_bb as Parameter if fit_gain is True
        if self.use_fit_gains:
            self.w_bb = Parameter(torch.tensor(np.zeros((self.node_size, self.node_size)) + 0.05,
                                                dtype=torch.float32))  # connenction gain to modify empirical sc
            self.w_ff = Parameter(torch.tensor(np.zeros((self.node_size, self.node_size)) + 0.05,
                                                dtype=torch.float32))
            self.w_ll = Parameter(torch.tensor(np.zeros((self.node_size, self.node_size)) + 0.05,
                                                dtype=torch.float32))
            param_reg.append(self.w_ll)
            param_reg.append(self.w_ff)
            param_reg.append(self.w_bb)
        else:
            self.w_bb = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)
            self.w_ff = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)
            self.w_ll = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)


        var_names = [a for a in dir(self.params) if (type(getattr(self.params, a)) == par)]
        for var_name in var_names:
            var = getattr(self.params, var_name)
            if (var.fit_par):
                if var_name == 'lm':
                    size = var.val.shape
                    var.val = Parameter(- 1 * torch.ones((size[0], size[1]))) # TODO: This is not consistent with what user would expect giving a variance
                    var.prior_mean = Parameter(var.prior_mean)
                    var.prior_var = Parameter(var.prior_var)
                    param_reg.append(var.val)
                    param_hyper.append(var.prior_mean)
                    param_hyper.append(var.prior_var)
                    self.track_params.append(var_name)
                else:
                    var.val = Parameter(var.val) # TODO: This is not consistent with what user would expect giving a variance
                    var.prior_mean = Parameter(var.prior_mean)
                    var.prior_var = Parameter(var.prior_var)
                    param_reg.append(var.val)
                    param_hyper.append(var.prior_mean)
                    param_hyper.append(var.prior_var)
                    self.track_params.append(var_name)



        self.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}

    def forward(self, external, hx, hE):
        # define some constants
        conduct_lb = 0  # lower bound for conduct velocity

        noise_std_lb = 0  # lower bound of std of noise
        lb = 0.0  # lower bound of local gains

        k_lb = 0.0  # lower bound of coefficient of external inputs

        # Generate the ReLU module for model parameters gEE gEI and gIE

        m = torch.nn.ReLU()
        # define constant 1 tensor
        con_1 = torch.tensor(1.0, dtype=torch.float32)
        # Defining NMM Parameters to simplify later equations
        #TODO: Change code so that params returns actual value used without extras below
        VL = m(self.params.VL.value())
        VI = m(self.params.VI.value())
        VE = m(self.params.VE.value())
        VNMDA = m(self.params.VNMDA.value())

        alpha_mg = m(self.params.alpha_mg.value())

        VR = m(self.params.VR.value())
        pi_sigma = m(self.params.pi_sigma.value())

        gL = m(self.params.gL.value())

        C = m(self.params.C.value())
        kappa = m(self.params.kappa.value())

        gamma_gE = m(self.params.gamma_gE.value())
        gamma_gE_sc = m(self.params.gamma_gE_sc.value())
        gamma_gI = m(self.params.gamma_gI.value())
        gamma_gI_sc = m(self.params.gamma_gI_sc.value())
        gamma_gNMDA = m(self.params.gamma_gNMDA.value())
        gamma_gNMDA_sc = m(self.params.gamma_gNMDA_sc.value())
        gamma_k = m(self.params.gamma_k.value())

        sigma_V = (noise_std_lb * con_1 + m(self.params.sigma_V.value())) #around 2
        sigma_g = (noise_std_lb * con_1 + m(self.params.sigma_g.value()))

        y0 = self.params.y0.value()
        mu = (0.1 * con_1 + m(self.params.mu.value()))
        k = (0.0 * con_1 + m(self.params.k.value()))
        cy0 = self.params.cy0.value()
        ki = self.params.ki.value()

        g = m(self.params.g.value())
        g_f = (lb * con_1 + m(self.params.g_f.value()))
        g_b = (lb * con_1 + m(self.params.g_b.value()))
        lm = self.params.lm.value()

        next_state = {}

        V = hx[:, :, 0]  # current of main population
        gE = hx[:,:, 1]  # current of excitory population
        gI = hx[:,:, 2]  # current of inhibitory population
        gNMDA = hx[:,:, 3]  # voltage of main population

        P_ind = np.arange(self.pop_size)[self.pop_names == 'P']
        E_ind = np.arange(self.pop_size)[self.pop_names == 'E']
        I_ind = np.arange(self.pop_size)[self.pop_names == 'I']
        dt = self.step_size



        if self.sc.shape[0] > 1:

            # Update the Laplacian based on the updated connection gains w_bb.
            w_b = torch.exp(self.w_bb) * torch.tensor(self.sc, dtype=torch.float32)
            w_n_b = w_b / torch.linalg.norm(w_b)

            self.sc_m_b = w_n_b
            dg_b = -torch.diag(torch.sum(w_n_b, dim=1))
            # Update the Laplacian based on the updated connection gains w_bb.
            w_f = torch.exp(self.w_ff) * torch.tensor(self.sc, dtype=torch.float32)
            w_n_f = w_f / torch.linalg.norm(w_f)

            self.sc_m_f = w_n_f
            dg_f = -torch.diag(torch.sum(w_n_f, dim=1))
            # Update the Laplacian based on the updated connection gains w_bb.
            w = torch.exp(self.w_ll) * torch.tensor(self.sc, dtype=torch.float32)
            w_n_l = (0.5 * (w + torch.transpose(w, 0, 1))) / torch.linalg.norm(
                0.5 * (w + torch.transpose(w, 0, 1)))



            self.sc_fitted = w_n_l
            dg_l = -torch.diag(torch.sum(w_n_l, dim=1))


        else:
            l_s = torch.tensor(np.zeros((1, 1)), dtype=torch.float32)
            dg_l = 0
            dg_b = 0
            dg_f = 0
            w_n_l = 0
            w_n_b = 0
            w_n_f = 0

        self.delays = (self.dist / (conduct_lb * con_1 + m(mu))).type(torch.int64)
        # print(torch.max(model.delays), model.delays.shape)

        # placeholder for the updated current state
        current_state = torch.zeros_like(hx)

        # placeholders for output BOLD, history of E I x f v and q
        eeg_window = []
        states_window = []

        max_xi = max([torch.norm(gamma_gE*gamma_gE_sc), \
                      torch.norm(gamma_gI*gamma_gI_sc), \
                      torch.norm(gamma_gNMDA*gamma_gE_sc)])
        #print('V', V.shape)
        # Use the forward model to get EEG signal at ith element in the window.
        for i_window in range(self.TRs_per_window):

            for step_i in range(self.steps_per_TR):
                Ed = torch.tensor(np.zeros((self.node_size, self.node_size)), dtype=torch.float32)  # delayed E

                """for ind in range(model.node_size):
                    #print(ind, hE[ind,:].shape, model.delays[ind,:].shape)
                    Ed[ind] = torch.index_select(hE[ind,:], 0, model.delays[ind,:])"""
                hE_new = hE.clone()
                Ed = hE_new.gather(1, self.delays)  # delayed E

                LEd_b = torch.reshape(torch.sum(w_n_b * torch.transpose(Ed, 0, 1), 1),
                                      (self.node_size, 1))  # weights on delayed E
                LEd_f = torch.reshape(torch.sum(w_n_f * torch.transpose(Ed, 0, 1), 1),
                                      (self.node_size, 1))  # weights on delayed E
                LEd_l = torch.reshape(torch.sum(w_n_l * torch.transpose(Ed, 0, 1), 1),
                                      (self.node_size, 1))  # weights on delayed E
                # Input noise for M.

                u_tms = external[:, step_i, i_window]
                #print("u_tms", u_tms.shape)
                #u_aud = external[:, i_hidden:i_hidden + 1, i_window, 1]
                #u_0 = external[:, i_hidden:i_hidden + 1, i_window, 2]
                norm =[]

                for i in range(self.pop_size):
                    norm.append(Normal(VR,pi_sigma[i]))
                # LEd+torch.matmul(dg,E): Laplacian on delayed E
                rV = gL*(VL) + gE*VE +gI*VI \
                          + gNMDA*self.m_nmda(alpha_mg,V)*(VNMDA)-(0.1+m(gL + gE +gI + gNMDA*self.m_nmda(alpha_mg,V)))*V\
                          + u_tms + sigma_V*torch.randn(self.node_size, self.pop_size)
                #print((LEd_f + 1 * torch.matmul(dg_f, (V[:,self.pop_names == 'E']- V[:,self.pop_names == 'I']))).shape)
                rV[:,P_ind] += g * (
                          LEd_l + 1 * torch.matmul(dg_l, (V[:,P_ind])))
                rV[:,E_ind] += g_f * \
                      (LEd_f + 1 * torch.matmul(dg_f, (V[:,E_ind]- V[:,I_ind])))
                rV[:,I_ind] += g_b * \
                      (-LEd_b - 1 * torch.matmul(dg_b, (V[:,E_ind]- V[:,I_ind])))
                xi = torch.cat([norm[j].cdf(V[:,:,np.newaxis].mean(0)[j]).float() for j in range(self.pop_size)])
                #print(xi.shape)
                rgE = gamma_k *torch.matmul(1*(gamma_gE*gamma_gE_sc)/max_xi, xi) - gE\
                        +1*sigma_g*torch.randn(self.node_size, self.pop_size)

                #xi = torch.concatenate([norm[j].cdf(V[:,j]).float()[:,np.newaxis] for j in range(self.pop_size)], dim=1)
                rgI = gamma_k *torch.matmul(1*(gamma_gI*gamma_gI_sc)/max_xi, xi) -gI\
                        +sigma_g*torch.randn(self.node_size, self.pop_size)


                #xi_gNMDA = torch.concatenate([norm[j].cdf(gNMDA[:,j]).float()[:,np.newaxis] for j in range(self.pop_size)], dim=1)
                rgNMDA = gamma_k *torch.matmul(1*(gamma_gNMDA*gamma_gE_sc)/max_xi, xi) - gNMDA\
                        +sigma_g*torch.randn(self.node_size, self.pop_size)
                # Update the states by step-size.

                ddV = V + dt * (rV)/C

                ddgE = gE + dt * (rgE) * kappa[0]
                ddgI = gI + dt * (rgI) * kappa[1]
                ddgNMDA =gNMDA + dt* (rgNMDA) *kappa[2]
                # Calculate the saturation for model states (for stability and gradient calculation).
                V = 1000*torch.tanh(ddV/1000)
                gE = 1000*torch.tanh(ddgE/1000)
                gI = 1000*torch.tanh(ddgI/1000)
                gNMDA = 1000*torch.tanh(ddgNMDA/1000)

                # update placeholders for E buffer
                hE[:, 0] = V[:,P_ind][:,0]
                # update the states
            

            # Put M E I Mv Ev and Iv at every tr to the placeholders for checking them visually.
            states_window.append(torch.cat([V[:,:,np.newaxis] , gE[:,:,np.newaxis], gI[:,:,np.newaxis],\
                                            gNMDA[:,:,np.newaxis]], dim =2)[:,:,:,np.newaxis])

            hE = torch.cat([V[:,P_ind], hE[:, :-1]], dim=1)  # update placeholders for E buffer

            # Put the EEG signal each tr to the placeholder being used in the cost calculation.
            lm_t = (lm.T / torch.sqrt(lm ** 2).sum(1)).T

            self.lm_t = (lm_t - 1 / self.output_size * torch.matmul(torch.ones((1, self.output_size)),
                                                                      lm_t))  # s2o_coef *
            #print(V[:,E_ind].shape)
            temp = cy0 * torch.matmul(self.lm_t, (V[:,E_ind]-1*V[:,I_ind].mean())) - 1 * y0
            eeg_window.append(temp)  # torch.abs(E) - torch.abs(I) + 0.0*noiseEEG)

        # Update the current state.


        next_state['eeg'] = torch.cat(eeg_window, dim=1)
        next_state['states'] = torch.cat(states_window, dim=3)
        next_state['current_state'] = torch.cat([V[:,:,np.newaxis] , gE[:,:,np.newaxis], gI[:,:,np.newaxis],\
                                            gNMDA[:,:,np.newaxis]], dim =2)


        return next_state, hE