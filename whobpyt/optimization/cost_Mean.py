import torch


class CostsMean():
    def __init__(self, num_regions,  varIdx, targetValue = None, empiricalData = None):
        
        self.num_regions = num_regions
        self.varIdx = varIdx # This is the index in the data simulation to extract variable time series from
        
        # Target can be specific to each region, or can have a single number that is repeated for each region
        if torch.numel(targetValue) == 1:
            self.targetValue = targetValue.repeat(num_regions)
        else:
            self.targetValue = targetValue
            
        if empiricalData != None:
            # In the future, if given empiricalData then will calculate the target value in this initialization function. 
            # That will possibly involve a time series of targets, for which then the calcLoss would need a parameter to identify
            # which one to fit to.
            pass
        
    def calcLoss(self, simData):
        # simData assumed to be in the form [time_steps, regions]
        # Returns the sum of the MSE of each regions mean value to target value of each reagion
        
        meanVar = torch.mean(simData[:,:], 0)
        
        return torch.nn.functional.mse_loss(meanVar, self.targetValue)
        