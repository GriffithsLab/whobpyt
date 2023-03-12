"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for model fitting using pytorch
"""

import numpy as np  # for numerical operations
import torch
import torch.optim as optim
from whobpyt.datatypes.outputs import OutputNM
from whobpyt.models.RWW.RWW_np import RWW_np #This should be removed and made general
import pickle
from sklearn.metrics.pairwise import cosine_similarity


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
        the times for repeating trainning
    cost: choice of the cost function
    """
    u = 0  # external input

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
        # placeholder for output(EEG and histoty of model parameters and loss)
        self.output_sim = OutputNM(self.model)
 
        self.ts = ts

        self.cost = cost

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train(self, learningrate=0.05, u=0, epoch_min = 10, r_lb = 0.85, lr_hyper = 0.05/40, lr_scheduler = True):
        """
        Parameters
        ----------
        learningrate : for machine learing speed
        u: stimulus

        """

        self.u = u

        #Define two different optimizers for each group
        modelparameter_optimizer = optim.Adam(self.model.params_fitted['modelparameter'], lr=learningrate, eps=1e-7)
        hyperparameter_optimizer = optim.Adam(self.model.params_fitted['hyperparameter'], lr=lr_hyper, eps=1e-7)

        # Define the learning rate schedulers for each group of parameters
        if lr_scheduler:
            total_steps = self.ts.shape[1]*self.num_epoches
            hyperparameter_scheduler = optim.lr_scheduler.OneCycleLR(hyperparameter_optimizer, lr_hyper, total_steps, anneal_strategy = "cos")
            hlrs = []
            modelparameter_scheduler = optim.lr_scheduler.OneCycleLR(modelparameter_optimizer, learningrate, total_steps, anneal_strategy = "cos")
            mlrs = []
        
        # initial state
        X = self.model.createIC(ver = 0)
        # initials of history of E
        hE = self.model.createDelayIC(ver = 0)

        # define masks for getting lower triangle matrix indices
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # placeholders for the history of model parameters
        fit_param = {}
        exclude_param = []
        fit_sc = 0
        fit_lm = 0
        loss = 0
        if self.model.use_fit_gains:
            exclude_param.append('gains_con')
            fit_sc = [self.model.sc[mask].copy()]  # sc weights history
        if self.model.use_fit_lfm:
            exclude_param.append('lm')
            fit_lm = [self.model.lm.detach().numpy().ravel().copy()]  # leadfield matrix history

        if(self.model.track_params):
            for par_name in self.model.track_params:
                var = getattr(self.model.param, par_name)
                fit_param[par_name] = [var.value().detach().numpy()]
        else:
            for key, value in self.model.state_dict().items():
                if key not in exclude_param:
                    fit_param[key] = [value.detach().numpy().ravel().copy()]

        loss_his = []  # loss placeholder

        # define num_windows
        num_windows = self.ts.shape[1]
        for i_epoch in range(self.num_epoches):

            # Create placeholders for the simulated states and outputs of entire time series.
            for name in self.model.state_names + self.output_sim.output_names:
                setattr(self.output_sim, name + '_train', [])

            # initial the external inputs
            external = torch.tensor(
                np.zeros([self.model.node_size, self.model.steps_per_TR, self.model.TRs_per_window]),
                dtype=torch.float32)

            # Perform the training in windows.

            for TR_i in range(num_windows):

                # Reset the gradient to zeros after update model parameters.
                hyperparameter_optimizer.zero_grad()
                modelparameter_optimizer.zero_grad()

                # if the external not empty
                if not isinstance(self.u, int):
                    external = torch.tensor(
                        (self.u[:, :, TR_i * self.model.TRs_per_window:(TR_i + 1) * self.model.TRs_per_window]),
                        dtype=torch.float32)

                # Use the model.forward() function to update next state and get simulated EEG in this batch.
                next_window, hE_new = self.model(external, X, hE)

                # Get the batch of empirical signal.
                ts_window = torch.tensor(self.ts[i_epoch, TR_i, :, :], dtype=torch.float32)

                # calculating loss
                
                sim = next_window[self.model.output_names[0] + "_window"]
                loss = self.cost.loss(sim, ts_window, self.model, next_window)
                
                # Put the batch of the simulated EEG, E I M Ev Iv Mv in to placeholders for entire time-series.
                for name in self.model.state_names + self.output_sim.output_names:
                    name_next = name + '_window'
                    tmp_ls = getattr(self.output_sim, name + '_train')
                    tmp_ls.append(next_window[name_next].detach().numpy())

                    setattr(self.output_sim, name + '_train', tmp_ls)

                loss_his.append(loss.detach().numpy())

                # Calculate gradient using backward (backpropagation) method of the loss function.
                loss.backward(retain_graph=True)

                # Optimize the model based on the gradient method in updating the model parameters.
                hyperparameter_optimizer.step()
                modelparameter_optimizer.step()
                
                if lr_scheduler:
                    #appending (needed to plot learning rate)
                    hlrs.append(hyperparameter_optimizer.param_groups[0]["lr"])
                    mlrs.append(modelparameter_optimizer.param_groups[0]["lr"])
                    
                    # schedular step 
                    hyperparameter_scheduler.step()
                    modelparameter_scheduler.step()

                # Put the updated model parameters into the history placeholders.
                # sc_par.append(self.model.sc[mask].copy())
                if(self.model.track_params):
                    for par_name in self.model.track_params:
                        var = getattr(self.model.param, par_name)
                        fit_param[par_name].append(var.value().detach().numpy())
                else:
                    for key, value in self.model.state_dict().items():
                        if key not in exclude_param:
                            fit_param[key].append(value.detach().numpy().ravel().copy())

                if self.model.use_fit_gains:
                    fit_sc.append(self.model.sc_fitted.detach().numpy()[mask].copy())
                if self.model.use_fit_lfm:
                    fit_lm.append(self.model.lm.detach().numpy().ravel().copy())

                # last update current state using next state...
                # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
                X = torch.tensor(next_window['current_state'].detach().numpy(), dtype=torch.float32)
                hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)

            ts_emp = np.concatenate(list(self.ts[i_epoch]),1)
            fc = np.corrcoef(ts_emp)

            tmp_ls = getattr(self.output_sim, self.output_sim.output_names[0] + '_train')
            ts_sim = np.concatenate(tmp_ls, axis=1)
            fc_sim = np.corrcoef(ts_sim[:, 10:])

            print('epoch: ', i_epoch, 
                  'loss:', loss.detach().numpy(),
                  'FC_cor: ', np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], 
                  'cos_sim: ', np.diag(cosine_similarity(ts_sim, ts_emp)).mean())
                  
            if lr_scheduler:
                print('Modelparam_lr: ', modelparameter_scheduler.get_last_lr()[0])
                print('Hyperparam_lr: ', hyperparameter_scheduler.get_last_lr()[0])

            for name in self.model.state_names + self.output_sim.output_names:
                tmp_ls = getattr(self.output_sim, name + '_train')
                setattr(self.output_sim, name + '_train', np.concatenate(tmp_ls, axis=1))

            self.output_sim.loss = np.array(loss_his)

            if i_epoch > epoch_min and np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1] > r_lb:
                break
        
        # Writing the training statistics to the output class
        if self.model.use_fit_gains:
            self.output_sim.weights = np.array(fit_sc)
        if self.model.use_fit_lfm:
            self.output_sim.leadfield = np.array(fit_lm)
        for key, value in fit_param.items():
            setattr(self.output_sim, key, np.array(value))

    def test(self, base_window_num, u=0):
        """
        Parameters
        ----------
        base_window_num: int
            length of num_windows for resting
        u : external or stimulus
        -----------
        """

        # define some constants
        transient_num = 10

        self.u = u

        # initial state
        X = self.model.createIC(ver = 1)
        # initials of history of E
        hE = self.model.createDelayIC(ver = 1)

        # placeholders for model parameters

        # define mask for getting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # define num_windows
        num_windows = self.ts.shape[1]
        # Create placeholders for the simulated BOLD E I x f and q of entire time series.
        for name in self.model.state_names + self.output_sim.output_names:
            setattr(self.output_sim, name + '_test', [])

        u_hat = np.zeros(
            (self.model.node_size,self.model.steps_per_TR,
             base_window_num *self.model.TRs_per_window + self.ts.shape[1]*self.ts.shape[3]))
        u_hat[:, :, base_window_num * self.model.TRs_per_window:] = self.u

        # Perform the training in batches.

        for TR_i in range(num_windows + base_window_num):

            # Get the input and output noises for the module.

            external = torch.tensor(
                (u_hat[:, :, TR_i * self.model.TRs_per_window:(TR_i + 1) * self.model.TRs_per_window]),
                dtype=torch.float32)

            # Use the model.forward() function to update next state and get simulated EEG in this batch.
            next_window, hE_new = self.model(external, X, hE)

            if TR_i > base_window_num - 1:
                for name in self.model.state_names + self.output_sim.output_names:
                    name_next = name + '_window'
                    tmp_ls = getattr(self.output_sim, name + '_test')
                    tmp_ls.append(next_window[name_next].detach().numpy())

                    setattr(self.output_sim, name + '_test', tmp_ls)

            # last update current state using next state...
            # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
            X = torch.tensor(next_window['current_state'].detach().numpy(), dtype=torch.float32)
            hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)
        
        ts_emp = np.concatenate(list(self.ts[-1]),1)
        fc = np.corrcoef(ts_emp)
        tmp_ls = getattr(self.output_sim, self.output_sim.output_names[0] + '_test')
        ts_sim = np.concatenate(tmp_ls, axis=1)

        fc_sim = np.corrcoef(ts_sim[:, transient_num:])
        print('FC_cor: ', np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], 
              'cos_sim: ', np.diag(cosine_similarity(ts_sim, ts_emp)).mean())
              
        for name in self.model.state_names + self.output_sim.output_names:
            tmp_ls = getattr(self.output_sim, name + '_test')
            setattr(self.output_sim, name + '_test', np.concatenate(tmp_ls, axis=1))

    def test_realtime(self, tr_p, step_size_n, step_size, num_windows):
        if self.model.model_name == 'RWW':
            mask = np.tril_indices(self.model.node_size, -1)

            X_np = 0.2 * np.random.uniform(0, 1, (self.model.node_size, self.model.state_size)) + np.array(
                [0, 0, 0, 1.0, 1.0, 1.0])
            variables_p = [a for a in dir(self.model.param) if
                           not a.startswith('__') and not callable(getattr(self.model.param, a))]
            # get penalty on each model parameters due to prior distribution
            for var in variables_p:
                # print(var)
                if np.any(getattr(self.model.param, var)[1] > 0):
                    des = getattr(self.model.param, var)
                    value = getattr(self.model, var)
                    des[0] = value.detach().numpy().copy()
                    setattr(self.model.param, var, des)
            model_np = RWW_np(self.model.node_size, self.model.TRs_per_window, step_size_n, step_size, tr_p,
                              self.model.sc_fitted.detach().numpy().copy(),
                              self.model.use_dynamic_boundary, self.model.use_Laplacian, self.model.param)

            # Create placeholders for the simulated BOLD E I x f and q of entire time series.
            for name in self.model.state_names + self.output_sim.output_names:
                setattr(self.output_sim, name + '_test', [])

            # Perform the training in batches.

            for TR_i in range(num_windows + 10):

                noise_in_np = np.random.randn(self.model.node_size, self.model.TRs_per_window, int(tr_p / step_size_n), 2)

                noise_BOLD_np = np.random.randn(self.model.node_size, self.model.TRs_per_window)

                next_window_np = model_np.forward(X_np, noise_in_np, noise_BOLD_np)
                if TR_i >= 10:
                    # Put the batch of the simulated BOLD, E I x f v q in to placeholders for entire time-series.
                    for name in self.model.state_names + self.output_sim.output_names:
                        name_next = name + '_window'
                        tmp_ls = getattr(self.output_sim, name + '_test')
                        tmp_ls.append(next_window_np[name_next])

                        setattr(self.output_sim, name + '_test', tmp_ls)

                # last update current state using next state...
                # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
                X_np = next_window_np['current_state']
            tmp_ls = getattr(self.output_sim, self.output_sim.output_names[0] + '_test')

            for name in self.model.state_names + self.output_sim.output_names:
                tmp_ls = getattr(self.output_sim, name + '_test')
                setattr(self.output_sim, name + '_test', np.concatenate(tmp_ls, axis=1))
        else:
            print("only RWW model for the test_realtime function")
