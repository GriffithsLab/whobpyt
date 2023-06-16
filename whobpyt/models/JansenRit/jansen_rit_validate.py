# Simulate JR with numpy code for validation

import numpy as np

class JansenRit_np():

    def __init__(self, num_regions, params, Con_Mtx, Dist_Mtx, step_size = 0.1):        
        
        # Initialize the RWW Model 
        #
        # INPUT
        #  num_regions: Int - Number of nodes in network to model
        #  params: RWW_Params - The parameters that all nodes in the network will share
        #  Con_Mtx: Tensor [num_regions, num_regions] - With connectivity (eg. structural connectivity)
            
        self.num_regions = num_regions
        self.params = params
        self.Con_Mtx = Con_Mtx
        self.Dist_Mtx = Dist_Mtx
        self.step_size = step_size

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


        # Simoid function

        def sigmoid(x, vmax, v0, r):
            return vmax / (1 + np.exp(r * (v0 - x)))
        
        init_state = hx
        sim_len = self.sim_len
        step_size = self.step_size

        state_hist = np.zeros((int(sim_len/step_size), self.num_regions, 6))
        M = init_state[:, 0:1]
        E = init_state[:, 1:2]
        I = init_state[:, 2:3]
        Mv = init_state[:, 3:4]
        Ev = init_state[:, 4:5]
        Iv = init_state[:, 5:6]

        num_steps = int(sim_len/step_size)
        dt = step_size

        for i in range(num_steps):

            # Calculate the derivatives
            dM = dt * Mv
            dE = dt * Ev
            dI = dt * Iv
            dMv = dt * (A*a*(sigmoid(vmax,v0,r, E - I))- (2*a*Mv) - (a**(2)*M))
            dEv = dt * (A*a*(mu + (c2*sigmoid(vmax,v0,r,(c1*M)))) - (2*a*Ev) - (a**(2)*E))
            dIv = dt * (B*b*(c4*sigmoid(vmax,v0,r,(c3*M))) - (2*b*Iv) - (b**(2)*I))

            # Update the state
            M = M + dM
            E = E + dE
            I = I + dI
            Mv = Mv + dMv
            Ev = Ev + dEv
            Iv = Iv + dIv

            state_hist[i, :, 0:1] = M
            state_hist[i, :, 1:2] = E
            state_hist[i, :, 2:3] = I
            state_hist[i, :, 3:4] = Mv
            state_hist[i, :, 4:5] = Ev
            state_hist[i, :, 5:6] = Iv


