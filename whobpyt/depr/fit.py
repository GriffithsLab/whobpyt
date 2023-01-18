"""
WhoBPyt Model Fitting Classes
"""

class Model_fitting:
    """
    Using ADAM and AutoGrad to fit JansenRit to empirical EEG
    Attributes
    ----------
    model: instance of class RNNJANSEN
        forward model JansenRit
    ts: array with num_tr x node_size
        empirical EEG time-series
    num_epoches: int
        the times for repeating training
    Methods:
    train()
        train model
    test()
        using the optimal model parater to simulate the BOLD
    """

    # from sklearn.metrics.pairwise import cosine_similarity
    def __init__(self, model, ts, num_epoches, cost):
        """
        Parameters
        ----------
        model: instance of class RNNJANSEN
            forward model JansenRit
        ts: array with num_tr x node_size
            empirical EEG time-series
        num_epoches: int
            the times for repeating trainning
        """
        self.model = model
        self.num_epoches = num_epoches
        # self.u = u
        """if ts.shape[1] != model.node_size:
            print('ts is a matrix with the number of datapoint X the number of node')
        else:
            self.ts = ts"""
        self.ts = ts

        self.cost = Costs(cost)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train(self, u=0):
        """
        Parameters
        ----------
        None
        Outputs: OutputRJ
        """

        # define some constants
        lb = 0.001
        delays_max = 500
        state_ub = 2
        state_lb = 0.5
        w_cost = 10

        epoch_min = 150  # run minimum epoch # part of stop criteria
        r_lb = 0.85  # lowest pearson correlation # part of stop criteria

        self.u = u

        # placeholder for output(EEG and histoty of model parameters and loss)
        self.output_sim = OutputNM(self.model.model_name, self.model.node_size, self.model.param,
                                   self.model.fit_gains_flat, self.model.fit_lfm_flat)
        # define an optimizor(ADAM)
        optimizer = optim.Adam(self.model.parameters(), lr=0.05, eps=1e-7)

        # initial state
        if self.model.model_name == 'WWD':
            # initial state
            X = torch.tensor(0.45 * np.random.uniform(0, 1, (self.model.node_size, self.model.state_size)) + np.array(
                [0, 0, 0, 1.0, 1.0, 1.0]), dtype=torch.float32)
        elif self.model.model_name == 'JR':
            X = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size, self.model.state_size)),
                             dtype=torch.float32)
        hE = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size, delays_max)),
                          dtype=torch.float32)

        # define masks for geting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # placeholders for the history of model parameters
        fit_param = {}
        exclude_param = []
        if self.model.fit_gains_flat == True:
            exclude_param.append('w_bb')
            fit_sc = [self.model.sc[mask].copy()]  # sc weights history
        if self.model.model_name == "JR" and self.model.fit_lfm_flat == True:
            exclude_param.append('lm')
            fit_lm = [self.model.lm.detach().numpy().ravel().copy()]  # leadfield matrix history

        for key, value in self.model.state_dict().items():
            if key not in exclude_param:
                fit_param[key] = [value.detach().numpy().ravel().copy()]

        loss_his = []

        # define constant 1 tensor

        con_1 = torch.tensor(1.0, dtype=torch.float32)

        # define num_batches
        num_batches = np.int(self.ts.shape[2] / self.model.batch_size)
        for i_epoch in range(self.num_epoches):

            # X = torch.tensor(np.random.uniform(0, 5, (self.model.node_size, self.model.state_size)) , dtype=torch.float32)
            # hE = torch.tensor(np.random.uniform(0, 5, (self.model.node_size,83)), dtype=torch.float32)
            eeg = self.ts[i_epoch % self.ts.shape[0]]
            # Create placeholders for the simulated EEG E I M Ev Iv and Mv of entire time series.
            for name in self.model.state_names + [self.output_sim.output_name]:
                setattr(self.output_sim, name + '_train', [])

            external = torch.tensor(np.zeros([self.model.node_size, self.model.hidden_size, self.model.batch_size]),
                                    dtype=torch.float32)

            # Perform the training in batches.

            for i_batch in range(num_batches):

                # Reset the gradient to zeros after update model parameters.
                optimizer.zero_grad()

                # Initialize the placeholder for the next state.
                X_next = torch.zeros_like(X)

                
                if not isinstance(self.u, int):
                    external = torch.tensor(
                        (self.u[:, :, i_batch * self.model.batch_size:(i_batch + 1) * self.model.batch_size]),
                        dtype=torch.float32)

                # Use the model.forward() function to update next state and get simulated EEG in this batch.
                
                next_batch, hE_new = self.model(external, X, hE)

                # Get the batch of emprical EEG signal.
                ts_batch = torch.tensor(
                    (eeg.T[i_batch * self.model.batch_size:(i_batch + 1) * self.model.batch_size, :]).T,
                    dtype=torch.float32)

                if self.model.model_name == 'WWD':
                    E_batch = next_batch['E_batch']
                    I_batch = next_batch['I_batch']
                    f_batch = next_batch['f_batch']
                    v_batch = next_batch['v_batch']
                    """loss_EI = 0.1 * torch.mean(
                        torch.mean(E_batch * torch.log(E_batch) + (con_1 - E_batch) * torch.log(con_1 - E_batch) \
                                   + 0.5 * I_batch * torch.log(I_batch) + 0.5 * (con_1 - I_batch) * torch.log(
                            con_1 - I_batch), axis=1))"""
                    loss_EI = torch.mean(self.model.E_v * (E_batch - self.model.E_m) ** 2) \
                                          + torch.mean(-torch.log(self.model.E_v)) +\
                              torch.mean(self.model.I_v * (I_batch - self.model.I_m) ** 2) \
                                          + torch.mean(-torch.log(self.model.I_v)) +\
                              torch.mean(self.model.f_v * (f_batch - self.model.f_m) ** 2) \
                                          + torch.mean(-torch.log(self.model.f_v)) +\
                              torch.mean(self.model.v_v * (v_batch - self.model.v_m) ** 2) \
                                          + torch.mean(-torch.log(self.model.v_v)) 
                else:
                    lose_EI = 0
                loss_prior = []
                # define the relu function
                m = torch.nn.ReLU()
                variables_p = [a for a in dir(self.model.param) if
                               not a.startswith('__') and not callable(getattr(self.model.param, a))]
                # get penalty on each model parameters due to prior distribution
                for var in variables_p:
                    # print(var)
                    if np.any(getattr(self.model.param, var)[1] > 0) and var != 'std_in' and var not in exclude_param:
                        # print(var)
                        dict_np = {}
                        dict_np['m'] = var + '_m'
                        dict_np['v'] = var + '_v'
                        loss_prior.append(torch.sum((lb + m(self.model.get_parameter(dict_np['v']))) * \
                                                    (m(self.model.get_parameter(var)) - m(
                                                        self.model.get_parameter(dict_np['m']))) ** 2) \
                                          + torch.sum(-torch.log(lb + m(self.model.get_parameter(dict_np['v'])))))
                # total loss
                if self.model.model_name == 'WWD':
                    loss = 0.1 * w_cost * self.cost.cost_eff(next_batch['bold_batch'], ts_batch) + 1*sum(
                        loss_prior) + 1 * loss_EI
                elif self.model.model_name == 'JR':
                    loss = w_cost * self.cost.cost_eff(next_batch['eeg_batch'], ts_batch) + sum(loss_prior)

                # Put the batch of the simulated EEG, E I M Ev Iv Mv in to placeholders for entire time-series.
                for name in self.model.state_names + [self.output_sim.output_name]:
                    name_next = name + '_batch'
                    tmp_ls = getattr(self.output_sim, name + '_train')
                    tmp_ls.append(next_batch[name_next].detach().numpy())
                    # print(name+'_train', name+'_batch', tmp_ls)
                    setattr(self.output_sim, name + '_train', tmp_ls)
                """eeg_sim_train.append(next_batch['eeg_batch'].detach().numpy())
                E_sim_train.append(next_batch['E_batch'].detach().numpy())
                I_sim_train.append(next_batch['I_batch'].detach().numpy())
                M_sim_train.append(next_batch['M_batch'].detach().numpy())
                Ev_sim_train.append(next_batch['Ev_batch'].detach().numpy())
                Iv_sim_train.append(next_batch['Iv_batch'].detach().numpy())
                Mv_sim_train.append(next_batch['Mv_batch'].detach().numpy())"""

                loss_his.append(loss.detach().numpy())
                # print('epoch: ', i_epoch, 'batch: ', i_batch, loss.detach().numpy())

                # Calculate gradient using backward (backpropagation) method of the loss function.
                loss.backward(retain_graph=True)

                # Optimize the model based on the gradient method in updating the model parameters.
                optimizer.step()

                # Put the updated model parameters into the history placeholders.
                # sc_par.append(self.model.sc[mask].copy())
                for key, value in self.model.state_dict().items():
                    if key not in exclude_param:
                        fit_param[key].append(value.detach().numpy().ravel().copy())

                if self.model.fit_gains_flat == True:
                    fit_sc.append(self.model.sc_m.detach().numpy()[mask].copy())
                if self.model.model_name == "JR" and self.model.fit_lfm_flat == True:
                    fit_lm.append(self.model.lm.detach().numpy().ravel().copy())

                # last update current state using next state... (no direct use X = X_next, since gradient calculation only depends on one batch no history)
                X = torch.tensor(next_batch['current_state'].detach().numpy(), dtype=torch.float32)
                hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)
                # print(hE_new.detach().numpy()[20:25,0:20])
                # print(hE.shape)
            fc = np.corrcoef(self.ts.mean(0))
            """ts_sim = np.concatenate(eeg_sim_train, axis=1)
            E_sim = np.concatenate(E_sim_train, axis=1)
            I_sim = np.concatenate(I_sim_train, axis=1)
            M_sim = np.concatenate(M_sim_train, axis=1)
            Ev_sim = np.concatenate(Ev_sim_train, axis=1)
            Iv_sim = np.concatenate(Iv_sim_train, axis=1)
            Mv_sim = np.concatenate(Mv_sim_train, axis=1)"""
            tmp_ls = getattr(self.output_sim, self.output_sim.output_name + '_train')
            ts_sim = np.concatenate(tmp_ls, axis=1)
            fc_sim = np.corrcoef(ts_sim[:, 10:])

            print('epoch: ', i_epoch, loss.detach().numpy())

            print('epoch: ', i_epoch, np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], 'cos_sim: ',
                  np.diag(cosine_similarity(ts_sim, self.ts.mean(0))).mean())

            for name in self.model.state_names + [self.output_sim.output_name]:
                tmp_ls = getattr(self.output_sim, name + '_train')
                setattr(self.output_sim, name + '_train', np.concatenate(tmp_ls, axis=1))
            """self.output_sim.EEG_train = ts_sim
            self.output_sim.E_train = E_sim
            self.output_sim.I_train= I_sim
            self.output_sim.P_train = M_sim
            self.output_sim.Ev_train = Ev_sim
            self.output_sim.Iv_train= Iv_sim
            self.output_sim.Pv_train = Mv_sim"""
            self.output_sim.loss = np.array(loss_his)

            if i_epoch > epoch_min and np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1] > r_lb:
                break
        # print('epoch: ', i_epoch, np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1])
        if self.model.fit_gains_flat == True:
            self.output_sim.weights = np.array(fit_sc)
        if self.model.model_name == 'JR' and self.model.fit_lfm_flat == True:
            self.output_sim.leadfield = np.array(fit_lm)
        for key, value in fit_param.items():
            setattr(self.output_sim, key, np.array(value))

    def test(self, x0, he0, base_batch_num, u=0):
        """
        Parameters
        ----------
        num_batches: int
            length of simEEG = batch_size x num_batches
        values of model parameters from model.state_dict
        Outputs:
        output_test: OutputJR
        """

        # define some constants
        state_lb = 0
        state_ub = 5
        delays_max = 500
        # base_batch_num = 20
        transient_num = 10

        self.u = u

        # initial state
        X = torch.tensor(x0, dtype=torch.float32)
        hE = torch.tensor(he0, dtype=torch.float32)

        # X = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size, self.model.state_size)) , dtype=torch.float32)
        # hE = torch.tensor(np.random.uniform(state_lb, state_ub, (self.model.node_size,500)), dtype=torch.float32)

        # placeholders for model parameters

        # define mask for geting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # define num_batches
        num_batches = np.int(self.ts.shape[2] / self.model.batch_size) + base_batch_num
        # Create placeholders for the simulated BOLD E I x f and q of entire time series.
        for name in self.model.state_names + [self.output_sim.output_name]:
            setattr(self.output_sim, name + '_test', [])

        u_hat = np.zeros(
            (self.model.node_size, self.model.hidden_size, base_batch_num * self.model.batch_size + self.ts.shape[2]))
        u_hat[:, :, base_batch_num * self.model.batch_size:] = self.u

        # Perform the training in batches.

        for i_batch in range(num_batches):

            # Initialize the placeholder for the next state.
            X_next = torch.zeros_like(X)

            # Get the input and output noises for the module.
            noise_in = torch.tensor(np.random.randn(self.model.node_size, self.model.hidden_size, \
                                                    self.model.batch_size, self.model.input_size), dtype=torch.float32)
            noise_out = torch.tensor(np.random.randn(self.model.node_size, self.model.batch_size), dtype=torch.float32)
            external = torch.tensor(
                (u_hat[:, :, i_batch * self.model.batch_size:(i_batch + 1) * self.model.batch_size]),
                dtype=torch.float32)

            # Use the model.forward() function to update next state and get simulated EEG in this batch.
            next_batch, hE_new = self.model(external, noise_in, noise_out, X, hE)

            if i_batch > base_batch_num - 1:
                for name in self.model.state_names + [self.output_sim.output_name]:
                    name_next = name + '_batch'
                    tmp_ls = getattr(self.output_sim, name + '_test')
                    tmp_ls.append(next_batch[name_next].detach().numpy())
                    # print(name+'_train', name+'_batch', tmp_ls)
                    setattr(self.output_sim, name + '_test', tmp_ls)

            # last update current state using next state... (no direct use X = X_next, since gradient calculation only depends on one batch no history)
            X = torch.tensor(next_batch['current_state'].detach().numpy(), dtype=torch.float32)
            hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)
            # print(hE_new.detach().numpy()[20:25,0:20])
            # print(hE.shape)
        fc = np.corrcoef(self.ts.mean(0))
        """ts_sim = np.concatenate(eeg_sim_test, axis=1)
        E_sim = np.concatenate(E_sim_test, axis=1)
        I_sim = np.concatenate(I_sim_test, axis=1)
        M_sim = np.concatenate(M_sim_test, axis=1)
        Ev_sim = np.concatenate(Ev_sim_test, axis=1)
        Iv_sim = np.concatenate(Iv_sim_test, axis=1)
        Mv_sim = np.concatenate(Mv_sim_test, axis=1)"""
        tmp_ls = getattr(self.output_sim, self.output_sim.output_name + '_test')
        ts_sim = np.concatenate(tmp_ls, axis=1)

        fc_sim = np.corrcoef(ts_sim[:, transient_num:])
        # print('r: ', np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], 'cos_sim: ', np.diag(cosine_similarity(ts_sim, self.ts.mean(0))).mean())

        for name in self.model.state_names + [self.output_sim.output_name]:
            tmp_ls = getattr(self.output_sim, name + '_test')
            setattr(self.output_sim, name + '_test', np.concatenate(tmp_ls, axis=1))
        
    def test_realtime(self, num_batches):
        if self.model.model_name == 'WWD':
            mask = np.tril_indices(self.model.node_size, -1)
            par_WWD = ParamsJR('WWD', g=[100,0])
            
            tr_p = 750
            X_np = 0.2 * np.random.uniform(0, 1, (self.model.node_size, self.model.state_size)) + np.array(
                        [0, 0, 0, 1.0, 1.0, 1.0])
                        
            model_np = WWD_np(self.model.node_size, self.model.batch_size, self.model.step_size, tr_p, self.model.sc_m.detach().numpy().copy(), par_WWD)
            par_WWD.g[0]= self.model.g.detach().numpy().copy()
            par_WWD.std_in[0]=self.model.std_in.detach().numpy().copy()
            par_WWD.std_out[0]=self.model.std_out.detach().numpy().copy()
            par_WWD.g_EE[0]= self.model.g_EE.detach().numpy().copy()
            par_WWD.g_IE[0]= self.model.g_IE.detach().numpy().copy()
            par_WWD.g_EI[0]= self.model.g_EI.detach().numpy().copy()
            model_np.update_param(par_WWD)
            #model_np.sc =  F.model.sc_m.detach().numpy().copy()
            # Create placeholders for the simulated BOLD E I x f and q of entire time series. 
            for name in self.model.state_names + [self.output_sim.output_name]:
                setattr(self.output_sim, name + '_test', [])
            
            # Perform the training in batches.
            
            for i_batch in range(num_batches+20):
                
                
                
                
                noise_in_np = np.random.randn(self.model.node_size,  self.model.batch_size, int(tr_p/self.model.step_size), \
                  2)
        

                noise_out_np = np.random.randn(self.model.node_size, self.model.batch_size)
        


        
                next_batch_np = model_np.forward(X_np,noise_in_np, noise_out_np)
                if i_batch >= 20:
                    # Put the batch of the simulated BOLD, E I x f v q in to placeholders for entire time-series. 
                    for name in self.model.state_names + [self.output_sim.output_name]:
                        name_next = name + '_batch'
                        tmp_ls = getattr(self.output_sim, name + '_test')
                        tmp_ls.append(next_batch_np[name_next].detach().numpy())
                        # print(name+'_train', name+'_batch', tmp_ls)
                        setattr(self.output_sim, name + '_test', tmp_ls)
                
                # last update current state using next state... (no direct use X = X_next, since gradient calculation only depends on one batch no history)
                X_np = next_batch_np['current_state']
            tmp_ls = getattr(self.output_sim, self.output_sim.output_name + '_test')
            ts_sim = np.concatenate(tmp_ls, axis=1)

            fc_sim = np.corrcoef(ts_sim[:, 10:])
            # print('r: ', np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], 'cos_sim: ', np.diag(cosine_similarity(ts_sim, self.ts.mean(0))).mean())
            print(np.corrcoef(fc_sim[mask], fc[mask])[0, 1])
            for name in self.model.state_names + [self.output_sim.output_name]:
                tmp_ls = getattr(self.output_sim, name + '_test')
                setattr(self.output_sim, name + '_test', np.concatenate(tmp_ls, axis=1))   
        else:
            print("only WWD model for the test_realtime funcion")   
