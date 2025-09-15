import numpy as np  # for numerical operations
import torch
import torch.optim as optim
from ..datatypes import Timeseries as Recording # JG: rename this to just Timeseries
from ..datatypes import AbstractNeuralModel,AbstractFitting,AbstractLoss
from ..datatypes import TrainingStats
#from whobpyt.models.RWW.RWW_np import RWW_np #This should be removed and made general
from ..functions.arg_type_check import method_arg_type_check
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class Model_fitting_fq:
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
    Methods:
    train()
        train model
    test()
        using the optimal model parater to simulate the BOLD
    """

    # from sklearn.metrics.pairwise import cosine_similarity
    def __init__(self, psd, num_epoches, model: AbstractNeuralModel, cost: AbstractLoss,):
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
        self.fq = torch.tensor(psd['fq'], dtype=torch.float32)
        self.psd = torch.tensor(psd['psd'], dtype=torch.float32)
        self.cost = cost
        #placeholder for output(EEG and histoty of model parameters and loss)
        self.trainingStats = TrainingStats(self.model)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def train(self, u= 0, learningrate: float = 0.05, lr_2ndLevel: float = 0.05, lr_scheduler: bool = False):
        """
        Parameters
        ----------
        None
        Outputs: OutputRJ
        """

        # placeholders for the history of model parameters

        loss_main_th = 1000

        method_arg_type_check(self.train, exclude = ['u', 'empRec']) # Check that the passed arguments (excluding self) abide by their expected data types

        # Define two different optimizers for each group
        modelparameter_optimizer = optim.Adam(self.model.params_fitted['modelparameter'], lr=learningrate, eps=1e-7)
        hyperparameter_optimizer = optim.Adam(self.model.params_fitted['hyperparameter'], lr=lr_2ndLevel, eps=1e-7)




        loss_his = []

        # define constant 1 tensor

        con_1 = torch.tensor(1.0, dtype=torch.float32)

        for i_epoch in range(self.num_epoches):
            if (loss_main_th > 1e-10):


                psd_target = self.psd[i_epoch % self.fq.shape[0]]
                fq_target = self.fq[i_epoch % self.fq.shape[0]]
                # Create placeholders for the simulated EEG E I M Ev Iv and Mv of entire time series.




                # Reset the gradient to zeros after update model parameters.
                hyperparameter_optimizer.zero_grad()
                modelparameter_optimizer.zero_grad()


                # Use the model.forward() function to update next state and get simulated EEG in this batch.
                next_batch = self.model(fq_target)

                print(((torch.log(next_batch) - torch.log(psd_target))**2).mean())

                #loss, loss_main = 1*self.cost.cost_eff(torch.log10(next_batch), torch.log10(psd_target),self.model)

                loss, loss_main = 1*self.cost.loss(next_batch, psd_target)
                loss_main_th = loss_main.detach().numpy()
                loss_his.append(loss.detach().numpy())
                # print('epoch: ', i_epoch, 'batch: ', i_batch, loss.detach().numpy())

                # Calculate gradient using backward (backpropagation) method of the loss function.
                loss.backward(retain_graph=True)

                # Optimize the model based on the gradient method in updating the model parameters.
                hyperparameter_optimizer.step()
                modelparameter_optimizer.step()

                # Put the updated model parameters into the history placeholders.
                # sc_par.append(self.model.sc[mask].copy())
                trackedParam = {}
                exclude_param = ['gains_con'] #This stores SC and LF which are saved seperately
                if(self.model.track_params):
                    for par_name in self.model.track_params:
                        var = getattr(self.model.params, par_name)
                        if (var.fit_par):
                            trackedParam[par_name] = var.value().detach().cpu().numpy().copy()
                            if var.fit_hyper:

                                trackedParam[par_name + "_prior_mean"] = var.prior_mean.detach().cpu().numpy().copy()
                                trackedParam[par_name + "_prior_var_inv"] = var.prior_var_inv.detach().cpu().numpy().copy()
                for key, value in self.model.state_dict().items():
                    if key not in exclude_param:
                        trackedParam[key] = value.detach().cpu().numpy().ravel().copy()
                self.trainingStats.appendParam(trackedParam)



                self.trainingStats.appendLoss(loss_his)
                print('epoch: ', i_epoch, loss.detach().numpy(),  loss_main.detach().numpy())







    def test(self, input):
        """
        Parameters
        ----------
        None
        Outputs: OutputRJ
        """


        fq_target = torch.tensor(input, dtype=torch.float32)
        next_batch = self.model(fq_target)

        return fq_target, next_batch