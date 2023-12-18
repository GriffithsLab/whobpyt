import torch
from whobpyt.datatypes.AbstractLoss import AbstractLoss


class CostsMean(AbstractLoss):
    '''
    Target Mean Value of a Variable
    
    This loss function calculates the mean value of a particular variable for every node across time, and then takes the Mean Squared Error of those means with the target value. 
    
    Attributes
    -----------------
    num_regions : Int
        The number of reagons in the model beign fit
    simKey : String 
        The name of the variable for which the mean is calculated
    targetValue : Tensor
        The target value either as single number or vector
    device : torch.device
        Whether the objective function is to run on GPU
    '''
        
    def __init__(self, num_regions, simKey, targetValue = None, empiricalData = None, batch_size = 1, device = torch.device('cpu')):
        '''
        Parameters
        -----------------
        num_regions : Int
            The number of regions in the model being fit
        simKey : String 
            The name of the variable for which the mean is calculated
        targetValue : Tensor
            The target value either as single number or vector
        '''
        super(CostsMean, self).__init__(simKey)
        
        self.num_regions = num_regions
        self.simKey = simKey # This is the key from the numerical simulation used to select the time series
        self.batch_size = batch_size
        
        self.device = device
        
        # Target can be specific to each region, or can have a single number that is repeated for each region
        if torch.numel(targetValue) == 1:
            if self.batch_size == 1:
                self.targetValue = targetValue.repeat(num_regions).to(device)
            else:
                self.targetValue = targetValue.repeat(num_regions, batch_size).to(device)
        else:
            self.targetValue = targetValue.to(device)
            
        if empiricalData != None:
            # In the future, if given empiricalData then will calculate the target value in this initialization function. 
            # That will possibly involve a time series of targets, for which then the loss would need a parameter to identify
            # which one to fit to.
            pass

        
    def loss(self, simData, empData = None):
        '''
        Method to calculate the loss
        
        Parameters
        --------------
        simData: dict of Tensor[ Nodes x Time ] or [ Nodes x Time x Blocks(Batch) ]
            The time series used by the loss function 
            
        Returns
        --------------
        Tensor
            The loss value 
        
        '''
        
        sim = simData[self.simKey]
        
        meanVar = torch.mean(sim, 1)
        
        return torch.nn.functional.mse_loss(meanVar, self.targetValue)
        