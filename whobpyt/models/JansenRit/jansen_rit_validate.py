# Simulate JR with numpy code for validation
# Sorenza Bastiaens
import numpy as np

class JansenRit_np():

    def __init__(self, node_size, step_size, output_size, sc, lm, dist, params):      

        
        # Initialize the JR Model 
        #
        # INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: Params_JR - The parameters that all nodes in the network will share
        #  Con_Mtx: Tensor [num_regions, num_regions] - With connectivity (eg. structural connectivity)
        #step_size=0.1
        self.step_size = step_size
        self.sc = sc  # structural connectivity factor
        self.node_size = node_size  # num of ROI
        self.output_size = output_size  # num of EEG channels
        self.params = params
        self.lm = lm  # leadfield matrix
        self.dist = dist  # distance between nodes

    def forward(self, external, hx, hE):

        # Runs the JR model

        # Defining JR parameters as numpy
        A = self.params.A.npValue()
        a = self.params.a.npValue()
        B = self.params.B.npValue()
        b = self.params.b.npValue()
        g = self.params.g.npValue()
        c1 = self.params.c1.npValue()
        c2 = self.params.c2.npValue()
        c3  = self.params.c3.npValue()
        c4  = self.params.c4.npValue()
        std_in  = self.params.std_in.npValue()
        vmax = self.params.vmax.npValue()
        v0 = self.params.v0.npValue()
        r = self.params.r.npValue()
        y0  = self.params.y0.npValue()
        mu =  self.params.mu.npValue()
        k = self.params.k.npValue()
        cy0 = self.params.cy0.npValue()
        ki = self.params.ki.npValue()
                
        g_f = self.params.g_f.npValue()
        g_b = self.params.g_b.npValue()

        next_state = {}
        
        # Sigmoid function
        def sigmoid(x, vmax, v0, r):
            return vmax / (1 + np.exp(r * (v0 - x)))
        
        init_state = hx
        sim_len = self.sim_len
        step_size = self.step_size

        state_hist = np.zeros((int(sim_len/step_size), self.node_size, 7))
        M = init_state[:, 0:1]
        E = init_state[:, 1:2]
        I = init_state[:, 2:3]
        Mv = init_state[:, 3:4]
        Ev = init_state[:, 4:5]
        Iv = init_state[:, 5:6]

        num_steps = int(sim_len/step_size)
        dt = step_size

        # Update the Laplacian based on the updated connection gains w_bb.
        w_b = np.exp(self.w_bb) * np.tensor(self.sc, dtype=np.float32)
        w_n_b = w_b / np.linalg.norm(w_b)
        self.sc_m_b = w_n_b
        dg_b = -np.diag(np.sum(w_n_b, dim=1))

        # Update the Laplacian based on the updated connection gains w_ff.
        w_f = np.exp(self.w_ff) * np.tensor(self.sc, dtype=np.float32)
        w_n_f = w_f / np.linalg.norm(w_f)
        self.sc_m_f = w_n_f
        dg_f = -np.diag(np.sum(w_n_f, dim=1))

        # Update the Laplacian based on the updated connection gains w_ll.
        w_l = np.exp(self.w_ll) * np.tensor(self.sc, dtype=np.float32)
        w_n_l = (0.5 * (w_l + np.transpose(w_l, 0, 1))) / np.linalg.norm(
            0.5 * (w_l + np.transpose(w_l, 0, 1)))
        self.sc_fitted = w_n_l
        dg_l = -np.diag(np.sum(w_n_l, dim=1))



        self.delays = (self.dist / self.mu).astype(int)

        # TODO currently single node, need to add all the connections and make it multiple nodes
        for i in range(num_steps):
            
            # LEd is to include the delays from other nodes
            # con_1 = 1
            # Don't include boundaries so no k_lb for example and no m(x) stuff

            # Basically rM inludes  (LEd_l + 1 * torch.matmul(dg_l, M)) 
            # Calculate the derivatives
            # Lateral is P-P
            # Forward is P-E
            # Backward is P-I
            Ed = np.zeros((self.node_size, self.node_size))
            hE_new = hE.copy()
            Ed = hE_new.gather(1,self.delays)
            LEd_b = np.reshape(np.sum(self.w_n_b * np.transpose(Ed, 0, 1), 1), (self.node_size, 1)) # Not sure if this needs to be included in validation
            LEd_f = np.reshape(np.sum(self.w_n_f * np.transpose(Ed, 0, 1), 1), (self.node_size, 1))
            LEd_l = np.reshape(np.sum(self.w_n_l * np.transpose(Ed, 0, 1), 1), (self.node_size, 1))

            rM = k * ki * u_tms + std_in*np.random.randn(self.node_size, 1) + g * (LEd_l + 1 * np.matmul(dg_l, M)) 
            rE = std_in*np.random.randn(self.node_size, 1) + g_f * (LEd_f + 1 * np.matmul(dg_f, E - I))
            rI = std_in*np.random.randn(self.node_size, 1) + g_b * (-LEd_b - 1 * np.matmul(dg_b, E - I))

            dM = dt * Mv
            dE = dt * Ev
            dI = dt * Iv
            dMv = dt * (A*a*( rM + sigmoid(vmax,v0,r, E - I))- (2*a*Mv) - (a**(2)*M)) # BE CAREGUL rM in code has the sigmoid so only take everything else from original code
            dEv = dt * (A*a*(mu + rE + (c2*sigmoid(vmax,v0,r,(c1*M)))) - (2*a*Ev) - (a**(2)*E))
            dIv = dt * (B*b*(rI + c4*sigmoid(vmax,v0,r,(c3*M))) - (2*b*Iv) - (b**(2)*I))

            # Update the state
            M = M + dM
            E = E + dE
            I = I + dI
            Mv = Mv + dMv
            Ev = Ev + dEv
            Iv = Iv + dIv
            hE = np.cat([M, hE[:, :-1]], dim=1)  # update placeholders for pyramidal buffer


            state_hist[i, :, 0:1] = M
            state_hist[i, :, 1:2] = E
            state_hist[i, :, 2:3] = I
            state_hist[i, :, 3:4] = Mv
            state_hist[i, :, 4:5] = Ev
            state_hist[i, :, 5:6] = Iv

            # Capture the states at every step .
            lm_t = (self.lm.T / np.sqrt(self.lm ** 2).sum(1)).T
            self.lm_t = (lm_t - 1 / self.output_size * np.matmul(np.ones((1, self.output_size)), lm_t))
            temp = cy0 * np.matmul(self.lm_t, M[:200, :]) - 1 * y0
            state_hist[i, :, 6:7] = temp # eeg_window

        next_state['M'] = state_hist[:,:,0]
        next_state['E'] = state_hist[:,:,1]
        next_state['I'] = state_hist[:,:,2]
        next_state['Mv'] = state_hist[:,:,3]
        next_state['Ev'] = state_hist[:,:,4]
        next_state['Iv'] = state_hist[:,:,5]
        next_state['eeg'] = state_hist[:,:,6]
            # Should then downsample the state_hist to the sampling rate of the EEG
        return next_state, hE

    
