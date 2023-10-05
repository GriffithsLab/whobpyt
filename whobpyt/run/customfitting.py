# Code based on:
# github.com/GriffithsLab/whobpyt/blob/dev/whobpyt/run/modelfitting.py
# github.com/Andrew-Clappison/whobpyt/blob/parallel_idea/examples/Multimodal_Parallel_Simulation_Example_v6.ipynb


import torch
import numpy as np
from whobpyt.datatypes import Recording
from whobpyt.datatypes import TrainingStats
from whobpyt.datatypes.AbstractFitting import AbstractFitting

class Fitting_FNGFPG(AbstractFitting):
    """
    Fitting Forward No Gradient Forward Parallel Gradient (FNG-FPG)
    
    This is a specalized model fitting class to:
        - backpropagate through time of the duration of 10's of seconds of real time scale simulation
        - learn generalized NMM parameters such that given a SC the model predicts the corresponding FC
        - the label can be a fixed value (example a FC matrix, instead of a neuroimaging recording)
    
    Attributes
    -----------
    model : AbstractNMM
        The model which has parameters to be trained.
    cost : AbstractLoss
        The objective function that will be used to calculate a loss.
    trainingStats : TrainingStats
        An object that will keep track of optimization/machine learning statistics during training. 
    lastSerial : Recording
        The simulated data from the last serial run (FNG)
    lastRec :  Recording   
        The simulated data from the last parallel run (FPG)
    device : torch.device
        Whether the fitting is to run on CPU or GPU
    """
    
    def __init__(self, model, cost, device = torch.device('cpu')):
        
        self.model = model
        self.cost = cost
        
        self.device = device
        
        self.trainingStats = TrainingStats(self.model)
        self.lastSerial = None #The last FNG run
        self.lastRec = None # The last FPG run
        
    def train(self, stim, empDatas, num_epochs, block_len, learningrate = 0.05, resetIC = True):
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
        block_len : Int
            The number of simulation steps per block. Total number of steps should be divisable by this number.    
        learningrate : Float
            Learning rate used by backpropagation optimizer.
        resetIC : Bool
            Whether to reset the models initial condition after every backpropagation
        
        '''
        
        optim = torch.optim.Adam(self.model.params_fitted['modelparameter'], lr = learningrate)
        
        initCond = self.model.createIC(ver = 0) #initCond is not being used, but this method resets the next_start_state of the model directly
        stateHist = self.model.createDelayIC(ver = 0)
        
        for e in range(num_epochs):
            print("epoch: ", e)
            
            # TRAINING_STATS: placeholders for the history of trainingStats
            loss_his = []  # loss placeholder to take the average for the epoch at the end of the epoch
            
            for empData in empDatas:
            
                # STEP 1/2: FNG (Forward in serial no gradients)
                # Purpose to get initial conditions to run different segments of time in parallel (which will have the same numerical result but a different backwards graph for back propagation)
                
                # Set Block Size to one before creating IC
                self.model.setBlocks(1)
            
                if resetIC:
                    # initial state
                    firstIC = self.model.createIC(ver = 0)
                    # initials of history of E
                    delayHist = self.model.createDelayIC(ver = 0) #TODO: Delays are currently is not implemented in various places
                else:
                    firstIC = self.model.next_start_state
                    delayHist = torch.tensor(1.0, device = self.device) # TODO: Delays are currently is not implemented in various places

                # initial the external inputs
                external = torch.tensor([0], device = self.device) # TODO: Currenlty this code only works for resting state
                
                num_blocks = int(self.model.sim_len/block_len)
            
                ## Noise for the epoch (which has 1 batch)
                [serialNoise, blockNoise] = self.model.genNoise(block_len)

                with torch.no_grad():
                    sim_vals, delayHist = self.model.forward(external, firstIC, delayHist, serialNoise)
                
                # Saving last Serial Run for Confirming it matches the Block Run
                if e == (num_epochs - 1):
                    lastSerial = {}
                    for simKey in set(self.model.state_names + self.model.output_names):
                        lastSerial[simKey] = Recording(sim_vals[simKey].detach().cpu().numpy().copy(), step_size = self.model.step_size) 

                print("Serial Finished")
    
                # STEP 2/2 FPG (Forward in "Parallel" with Gradients)
                # Using initial conditions acquired for serial run and with the same noise
                
                #print(blockNoise['E'].shape) # Time x Nodes x Blocks
                newICs = torch.zeros(self.model.node_size, len(self.model.state_names), num_blocks).to(self.device) # nodes x state_variables x blocks
                idx = 0
                for name in (self.model.state_names): #WARNING: Cannot use set() here as it does not preserve order, also this code assumes order of model.state_names is correct
                    newICs[:, idx, :] = sim_vals[name][:,(int(block_len/self.model.step_size)-1)::int(block_len/self.model.step_size)]
                    idx += 1
                ICs = torch.cat((firstIC, newICs), dim = 2)
                newICs = ICs[:,:,0:num_blocks] #This gets rid of the last state, for which would be the IC of the next forward() call
                nextICs = ICs[:,:,[-1]] #Brackets are to keep the dimension

                self.model.setBlocks(num_blocks)
                self.model.next_start_state = newICs
                sim_vals, delayHist = self.model.forward(external, newICs, delayHist, blockNoise)
                self.model.next_start_state = nextICs #TODO: Update to deal with delays as well
                
                print("Blocked Finished")
                
                
                # calculating loss
                loss = self.cost.loss(sim_vals, empData)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                print("Params Updated")
                                                
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
        self.lastSerial = lastSerial
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
    