# Code based on:
# github.com/GriffithsLab/whobpyt/blob/dev/whobpyt/run/customfitting.py


import torch
import numpy as np
from whobpyt.datatypes import Recording
from whobpyt.datatypes import TrainingStats
from whobpyt.datatypes.AbstractFitting import AbstractFitting

class Fitting_Batch(AbstractFitting):
    """
    Fitting Batch
    
    This is a specalized model fitting class to train using a batched approach.
    
    Attributes
    -----------
    model : AbstractNMM
        The model which has parameters to be trained.
    cost : AbstractLoss
        The objective function that will be used to calculate a loss.
    trainingStats : TrainingStats
        An object that will keep track of optimization/machine learning statistics during training.
    lastRec :  Recording
        The simulated data from the last batch run
    device : torch.device
        Whether the fitting is to run on CPU or GPU
    """
    
    def __init__(self, model, cost, device = torch.device('cpu')):
        
        self.model = model
        self.cost = cost
        
        self.device = device
        
        self.trainingStats = TrainingStats(self.model)
        self.lastRec = None
        
    def train(self, stim, empDatas, num_epochs, batch_size, learningrate = 0.05, staticIC = True, staticNoise = False):
        '''
        
        Method to train the model.
        
        Parameters
        -----------
        stim : Int or Tensor
            The input into the NMM (is 0 for the resting state case).
        empDatas : List of Recording or other object
            The empirical data compared to in the objective function. 
        num_epochs : Int
            Number of epochs for training.
        batch_size : Int
            The number of simulations run that will be backpropagated through simultaneously.    
        learningrate : Float
            Learning rate used by backpropagation optimizer.
        staticIC : Bool
            Whether to reset the models initial condition after every backpropagation
        staticNoise : Bool
            Whether to use the same noise for each epoch
        '''
        
        optim = torch.optim.Adam(self.model.params_fitted['modelparameter'], lr = learningrate)
                        
        self.model.setBlocks(batch_size) # Set Block Size to one before creating IC
        initCond = self.model.createIC(ver = 0) #initCond is not being used, but this method resets the next_start_state of the model directly
        stateHist = self.model.createDelayIC(ver = 0)
        [serialNoise, blockNoise] = self.model.genNoise(self.model.sim_len, batched = True)
        del serialNoise
        
        dummyVal = torch.tensor(0).to(self.device)
        
        for e in range(num_epochs):
            print("epoch: ", e)
            
            # TRAINING_STATS: placeholders for the history of trainingStats
            loss_his = []  # loss placeholder to take the average for the epoch at the end of the epoch
            
            for empData in empDatas:
            
                if not staticIC:
                    # initial state
                    firstIC = self.model.createIC(ver = 0)
                    # initials of history of E
                    delayHist = self.model.createDelayIC(ver = 0) #TODO: Delays are currently is not implemented in various places
                else:
                    firstIC = self.model.next_start_state
                    delayHist = dummyVal # TODO: Delays are currently is not implemented in various places

                # initial the external inputs
                external = dummyVal # TODO: Currenlty this code only works for resting state
                
                num_blocks = batch_size
                
                if not staticNoise:
                    [serialNoise, blockNoise] = self.model.genNoise(self.model.sim_len, batched = True)
                    del serialNoise

                sim_vals, delayHist = self.model.forward(external, firstIC, delayHist, blockNoise, batched = True)

                
                print("Batched Forward Finished")
                
                
                # calculating loss
                loss = self.cost.loss(sim_vals, empData)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                print("Batched Backwards Finished")
                                                
                # TRAINING_STATS: Adding Loss for every training backpropagation
                loss_his.append(loss.detach().cpu().numpy().copy())
                
            self.trainingStats.appendLoss(np.mean(loss_his))
            trackedParams = {}
            if(self.model.track_params):
                for parKey in self.model.track_params:
                    var = getattr(self.model.params, parKey)
                    if (var.fit_par):
                        trackedParams[parKey] = var.value().detach().cpu().numpy().copy()
            self.trainingStats.appendParam(trackedParams)
    
        # Saving the last recording of training as a Model_fitting attribute
        self.lastRec = {}
        for simKey in set(self.model.state_names + self.model.output_names):
            self.lastRec[simKey] = Recording(sim_vals[simKey].detach().cpu().numpy().copy(), step_size = self.model.step_size) #TODO: This won't work if different variables have different step sizes
        
        
    def evaluate(self, stim, empData):
        '''
        Not implemented yet. 
        '''
        pass
    
    
    def simulate(self, stim, numTP):
        '''
        Not implemented yet.
        '''
        pass
    