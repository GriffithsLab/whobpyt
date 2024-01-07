"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting module for model fitting using pytorch
"""

import numpy as np  # for numerical operations
import torch
import torch.optim as optim
from whobpyt.datatypes import Recording
from whobpyt.datatypes.AbstractFitting import AbstractFitting
from whobpyt.datatypes.AbstractNMM import AbstractNMM
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.datatypes.outputs import TrainingStats
from whobpyt.models.RWW.RWW_np import RWW_np #This should be removed and made general
from whobpyt.functions.arg_type_check import method_arg_type_check
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class Model_fitting(AbstractFitting):
    """
    This Model_fitting class is able to fit resting state data or evoked potential data
    for which the input training data is empty or some stimulus to one or more NMM nodes,
    and the label is an associated empirical neuroimaging recording.

    Studies which consider different kinds of input, such as if SC or some other variable
    is associated with an empirical recording, must use a different fitting class.

    Attributes
    ----------
    model: AbstractNMM
        Whole Brain Model to Simulate
    cost: AbstractLoss
        A particular objective function which the model will be optimized for.
    trainingStats: TrainingStats
        Information about objective function loss and parameter values over training windows/epochs
    lastRec: Recording
        The last simulation of fitting(), evaluation(), or simulation()
    device : torch.device
        Whether the fitting is to run on CPU or GPU
    """

    def __init__(self, model: AbstractNMM, cost: AbstractLoss, device = torch.device('cpu')):
        """
        Parameters
        ----------
        model: AbstractNMM
            Whole Brain Model to Simulate
        cost: AbstractLoss
            A particular objective function which the model will be optimized for.
        device : torch.device
            Whether the fitting is to run on CPU or GPU
        """
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types

        self.model = model
        self.cost = cost

        self.device = device

        self.trainingStats = TrainingStats(self.model)
        #self.lastRec = None #A dictionary or Recordings of the last simulation preformed (either training or evaluation)

        #self.u = None #This is the ML "Training Input"
        #self.empTS = ts #This is the ML "Training Labels" - A list

    def save(self, filename):
        """
        Parameters
        ----------
        filename: String
            filename to use when saving object to file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train(self, u, empRec,
              num_epochs: int, TPperWindow: int, warmupWindow: int = 0, learningrate: float = 0.05, lr_2ndLevel: float = 0.05, lr_scheduler: bool = False):
        """
        Parameters
        ----------
        u: type
           This stimulus is the ML "Training Input"
        empRec: list of Recording
            This is the ML "Training Labels"
        num_epochs: int
            the number of times to go through the entire training data set
        TPperWindow: int
            Number of Empirical Time Points per window. model.forward does one window at a time.
        learningrate: float
            rate of gradient descent
        lr_2ndLevel: float
            learning rate for priors of model parameters, and possibly others
        lr_scheduler: bool
            Whether to use the learning rate scheduler
        """
        method_arg_type_check(self.train, exclude = ['u', 'empRec']) # Check that the passed arguments (excluding self) abide by their expected data types

        # Define two different optimizers for each group
        modelparameter_optimizer = optim.Adam(self.model.params_fitted['modelparameter'], lr=learningrate, eps=1e-7)
        hyperparameter_optimizer = optim.Adam(self.model.params_fitted['hyperparameter'], lr=lr_2ndLevel, eps=1e-7)



        # initial state
        X = self.model.createIC(ver = 0)
        #print(X.shape)
        # initials of history of E
        hE = self.model.createDelayIC(ver = 0)
        #print(hE.shape)

        # define masks for getting lower triangle matrix indices
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)



        # LOOP 1/4: Number of Training Epochs
        for i_epoch in range(num_epochs):

            # Perform the training in windows.
            if i_epoch == 0:
                warmup_windows = 100

            else:
                warmup_windows = warmupWindow
            # TRAINING_STATS: placeholders for the history of trainingStats
            loss_his = []  # loss placeholder to take the average for the epoch at the end of the epoch

            print("Epoch: ", i_epoch)

            # LOOP 2/4: Number of Recordings in the Training Dataset

            # TIME SERIES: Create placeholders for the simulated states and outputs of entire time series corresponding to one recording
            windListDict = {} # A Dictionary with a List of windowed time series
            for name in ['states', self.model.output_names[0]]:
                windListDict[name] = []





            # initial the external inputs
            external = torch.tensor(
                np.zeros([self.model.node_size, self.model.steps_per_TR, self.model.TRs_per_window, self.model.pop_size]),
                dtype=torch.float32)


            for TR_i in range(warmup_windows):
                windowedTS = empRec[i_epoch]



                # Use the model.forward() function to update next state and get simulated EEG in this batch.
                next_window, hE_new = self.model(external, X, hE)
                X = torch.tensor(next_window['current_state'].detach().numpy(), dtype=torch.float32)
                hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)

            # LOOP 3/4: Number of windowed segments for the recording
            for win_idx in range(windowedTS.shape[0]):

                # Reset the gradient to zeros after update model parameters.
                hyperparameter_optimizer.zero_grad()
                modelparameter_optimizer.zero_grad()

                # if the external not empty
                if not isinstance(u, int):
                    external = torch.tensor(
                        (u[:, :, win_idx * self.model.TRs_per_window:(win_idx + 1) * self.model.TRs_per_window]),
                        dtype=torch.float32)

                # LOOP 4/4: The loop within the forward model (numerical solver), which is number of time points per windowed segment
                next_window, hE_new = self.model(external, X, hE)

                # Get the batch of empirical signal.
                ts_window = torch.tensor(windowedTS[win_idx, :, :], dtype=torch.float32)

                # calculating loss
                loss, loss_main = self.cost.loss(next_window, ts_window)

                # TIME SERIES: Put the window of simulated forward model.
                for name in ['states', self.model.output_names[0]]:
                    windListDict[name].append(next_window[name].detach().cpu().numpy())

                # TRAINING_STATS: Adding Loss for every training window (corresponding to one backpropagation)
                loss_his.append(loss_main.detach().cpu().numpy())

                # Calculate gradient using backward (backpropagation) method of the loss function.
                loss.backward(retain_graph=True)

                # Optimize the model based on the gradient method in updating the model parameters.
                hyperparameter_optimizer.step()
                modelparameter_optimizer.step()



                # last update current state using next state...
                # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
                X = next_window['current_state'].detach().clone() # dtype=torch.float32
                hE = hE_new.detach().clone() #dtype=torch.float32

            ts_emp = np.concatenate(list(windowedTS),1) #TODO: Check this code
            fc = np.corrcoef(ts_emp)

            # TIME SERIES: Concatenate all windows together to get one recording
            for name in ['states', self.model.output_names[0]]:
                    windListDict[name] = np.concatenate(windListDict[name], axis=len(windListDict[name][0].shape)-1)

            ts_sim = windListDict[self.model.output_names[0]]
            fc_sim = np.corrcoef(ts_sim[:, 10:])

            print('epoch: ', i_epoch,
                  'loss:', loss_main.detach().cpu().numpy(),
                  'Pseudo FC_cor: ', np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], #Calling this Pseudo as different windows of the time series have slighly different parameter values
                  'cos_sim: ', np.diag(cosine_similarity(ts_sim, ts_emp)).mean())



            # TRAINING_STATS: Put the updated model parameters into the history placeholders at the end of every epoch.
            # Additing Mean Loss for the Epoch
            self.trainingStats.appendLoss(np.mean(loss_his))
            # NMM/Other Parameter info for the Epoch (a list where a number is recorded every window of every record)
            trackedParam = {}
            exclude_param = ['gains_con'] #This stores SC and LF which are saved seperately
            if(self.model.track_params):
                for par_name in self.model.track_params:
                    var = getattr(self.model.params, par_name)
                    if (var.fit_par):
                        trackedParam[par_name] = var.value().detach().cpu().numpy().copy()

                        trackedParam[par_name + "_prior_mean"] = var.prior_mean.detach().cpu().numpy().copy()
                        trackedParam[par_name + "_prior_var"] = var.prior_var.detach().cpu().numpy().copy()
            for key, value in self.model.state_dict().items():
                if key not in exclude_param:
                    trackedParam[key] = value.detach().cpu().numpy().ravel().copy()
            self.trainingStats.appendParam(trackedParam)
            # Saving the SC and/or Lead Field State at Every Epoch
            if self.model.use_fit_gains:
                self.trainingStats.appendSC(self.model.sc_fitted.detach().cpu().numpy())
            """if self.model.use_fit_lfm:
                self.trainingStats.appendLF(self.model.lm.detach().cpu().numpy())"""

        self.trainingStats.updateOutputs(windListDict[self.model.output_names[0]], 'training')
        self.trainingStats.updateStates(windListDict['states'], 'training')

    def evaluate(self, u, empRec, TPperWindow: int, base_window_num: int = 0, transient_num: int = 10):
        """
        Parameters
        ----------
        u : int or Tensor
            external or stimulus
        empRec: list of Recording
            This is the ML "Training Labels"
        TPperWindow: int
            Number of Empirical Time Points per window. model.forward does one window at a time.
        base_window_num : int
            length of num_windows for resting
        transient_num : int
            The number of initial time points to exclude from some metrics
        -----------
        """
        method_arg_type_check(self.evaluate, exclude = ['u']) # Check that the passed arguments (excluding self) abide by their expected data types
        #TODO: Should be updated to take a list of u and empRec

        # initial state
        X = self.model.createIC(ver = 1)
        # initials of history of E
        hE = self.model.createDelayIC(ver = 1)

        # define mask for getting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)

        # Create placeholders for the simulated states and outputs of entire time series corresponding to one recording
        windListDict = {} # A Dictionary with a List of windowed time series
        for name in ['states', self.model.output_names[0]]:
            windListDict[name] = []

        num_windows = empRec.shape[1]
        u_hat = np.zeros(
            (self.model.node_size,self.model.steps_per_TR,
             base_window_num*self.model.TRs_per_window + num_windows*self.model.TRs_per_window, self.model.pop_size))
        u_hat[:, :, base_window_num * self.model.TRs_per_window:] = u

        # LOOP 1/2: The number of windows in a recording
        for win_idx in range(num_windows + base_window_num):

            # Get the input and output noises for the module.
            external = torch.tensor(
                (u_hat[:, :, win_idx * self.model.TRs_per_window:(win_idx + 1) * self.model.TRs_per_window]),
                dtype=torch.float32)

            # LOOP 2/2: The loop within the forward model (numerical solver), which is number of time points per windowed segment
            next_window, hE_new = self.model.forward(external, X, hE)

            # TIME SERIES: Put the window of simulated forward model.
            if win_idx > base_window_num - 1:
                for name in ['states', self.model.output_names[0]]:

                    windListDict[name].append(next_window[name].detach().cpu().numpy())

            # last update current state using next state...
            # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
            X = next_window['current_state'].detach().clone() # dtype=torch.float32
            hE = hE_new.detach().clone() #dtype=torch.float32

        windowedTS = empRec[-1]
        ts_emp = np.concatenate(list(windowedTS),1) #TODO: Check this code
        fc = np.corrcoef(ts_emp)

        # TIME SERIES: Concatenate all windows together to get one recording
        for name in ['states', self.model.output_names[0]]:
            print(windListDict[name][0].shape)
            windListDict[name] = np.concatenate(windListDict[name], axis=len(windListDict[name][0].shape)-1)

        ts_sim = windListDict[self.model.output_names[0]]
        fc_sim = np.corrcoef(ts_sim[:, transient_num:])

        print('FC_cor: ', np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1],
              'cos_sim: ', np.diag(cosine_similarity(ts_sim, ts_emp)).mean())

        # Saving the last recording of training as a Model_fitting attribute
        self.trainingStats.updateOutputs(windListDict[self.model.output_names[0]], 'testing')
        self.trainingStats.updateStates(windListDict['states'], 'testing')