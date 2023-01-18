"""
WhoBPyt Objective Function Classes
"""

import torch


class meanVariableLoss():
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
        # simData assumed to be in the form [time_steps, regions, state_vars (+ opt_params)]
        # Returns the sum of the MSE of each regions mean value to target value of each reagion 
        
        meanVar = torch.mean(simData[:,:, self.varIdx], 0)
        
        return torch.nn.functional.mse_loss(meanVar, self.targetValue)
        
    

class powerSpectrumLoss():
    # TODO: Deal with num_region vs. num_channels vs. num_parcels conflict with variable naming
    def __init__(self, num_regions, varIdx, sampleFreqHz, targetValue = None, empiricalData = None):
        self.num_regions = num_regions
        self.varIdx = varIdx  # This is the index in the data simulation to extract variable time series from
        
        self.sampleFreqHz = sampleFreqHz
        
        if targetValue != None:
            self.targetValue = targetValue
        
        if empiricalData != None:
            # In the future, if given empiricalData then will calculate the target value in this initialization function. 
            # That will possibly involve a time series of targets, for which then the calcLoss would need a parameter to identify
            # which one to fit to.
            pass
    
    def calcPSD(signal, sampleFreqHz, minFreq = None, maxFreq = None, axMethod = 2):
        # signal assumed to be in the form of [time_steps, regions or channels]
        # Returns the Power Spectrial Density with associated Hz values
    
        N = len(signal)
        
        # These defines the range of the PSD to return
        if minFreq == None:
            minFreq = 0
        if maxFreq == None:
            maxFreq = sampleFreqHz//2
        
        # Not sure which axis method is correct
        if axMethod == 1:
            fftAxis = torch.linspace(0,sampleFreqHz, N)
        elif axMethod == 2:
            fftAxis = torch.arange(N)*sampleFreqHz/N
        
        # Take the FFT of the Signal
        signalFFT = torch.fft.fft(signal, dim = 0)
        
        # Take the square value (it is complex) so following is equivalent:
        # Square of absolute, or
        # Sum of real squared and imag squared
        spectralDensity = torch.abs(signalFFT[:N//2])**2
        
        #Filter to the desired range
        minPoint = int((minFreq/(sampleFreqHz//2))*len(spectralDensity))
        maxPoint = int((maxFreq/(sampleFreqHz//2))*len(spectralDensity))
        sdAxis = fftAxis[minPoint:maxPoint]
        sdValues = spectralDensity[minPoint:maxPoint]
        
        return sdAxis, sdValues
    
    def downSmoothPSD(sdAxis, sdValues, numPoints = 512):
        #WARNING: This function will not necessarily achieve exactly numPoints number of points
        
        kernel_size = len(sdValues)//(numPoints//2)
        stride = kernel_size//2
        
        sdAxis_dS = torch.nn.functional.avg_pool1d(torch.unsqueeze(sdAxis,0), kernel_size,stride,0)[0]
        sdValues_dS = torch.transpose(torch.nn.functional.avg_pool1d(torch.unsqueeze(torch.transpose(sdValues, 0, 1), dim = 0), kernel_size,stride,0)[0], 0 , 1)
        
        return sdAxis_dS, sdValues_dS
    
    def scalePSD(sdAxis_dS, sdValues_dS):
        scale = torch.trapezoid(sdValues_dS, sdAxis_dS, dim = 0)
        # Alternative Method: 
        # spacing = (sdAxis_dS[-1] - sdAxis_dS[0])/len(sdAxis_dS)
        # scale = torch.trapezoid(sdValues_dS, dx = spacing, dim = 0)
        
        sdValues_dS_scaled = sdValues_dS / scale
        
        return sdAxis_dS, sdValues_dS_scaled

    
    def calcLoss(self, simData):
        # simData assumed to be in the form [time_steps, regions or channels, one or more variables]
        # Returns the MSE of the difference between the simulated and target power spectrum
        
        sdAxis, sdValues = powerSpectrumLoss.calcPSD(simData[:, :, self.varIdx], sampleFreqHz = self.sampleFreqHz, minFreq = 2, maxFreq = 40)
        sdAxis_dS, sdValues_dS = powerSpectrumLoss.downSmoothPSD(sdAxis, sdValues, numPoints = 32)
        sdAxis_dS, sdValues_dS_scaled = powerSpectrumLoss.scalePSD(sdAxis_dS, sdValues_dS)

        return torch.nn.functional.mse_loss(sdValues_dS_scaled, self.targetValue)
    


class functionalConnectivityLoss():
    # Right now this fit's to a fixed "empirical" FC matrix, but in the future
    # will change to fit to a time series of FC
    # Furthermore, for computational effeciency a batch of overlapping FC's will
    # be calculated to create a kind of mini-batch back propagation 
    
    def __init__(self, num_regions, varIdx, targetValue = None, empiricalData = None):
        self.num_regions = num_regions
        self.varIdx = varIdx  # This is the index in the data simulation to extract variable time series from
        
        if targetValue != None:
            self.targetValue = targetValue
        
        if empiricalData != None:
            # In the future, if given empiricalData then will calculate the target value in this initialization function. 
            # That will possibly involve a time series of targets, for which then the calcLoss would need a parameter to identify
            # which one to fit to.
            pass
        
    def calcFC(data):
        # data assumed to be in the form of [time_steps, regions or channels]
        # Returns the functional connectivity of the data
        return torch.corrcoef(torch.transpose(data, 0, 1))
    
    def calcLoss(self, simData):
        # simData assumed to be in the form [time_steps, regions or channels, one or more variables]
        # Returns the MSE of the difference between the simulated and target functional connectivity
        
        # Calculating the FC
        simFC = functionalConnectivityLoss.calcFC(simData[:, :, self.varIdx])
        
        # Removing the self-connection Identity
        for z in range(self.num_regions):
            simFC[z,z] = 0.0
        
        # Normalizing the matrix
        simFC_norm = (1/torch.linalg.matrix_norm(simFC, ord = 2)) * simFC
        
        return torch.nn.functional.mse_loss(simFC_norm, self.targetValue)

    def calcCorLoss(self, simData):
        # simData assumed to be in the form [time_steps, regions or channels, one or more variables]
        # Returns the MSE of the difference between the simulated and target functional connectivity
        
        # Calculating the FC
        simFC = functionalConnectivityLoss.calcFC(simData[:, :, self.varIdx])
        
        # Get lower triangle values for both simulated and target SC
        tril_indices = torch.tril_indices(simFC.shape[0], simFC.shape[1], -1)
        simFC_tril = simFC[tril_indices[0], tril_indices[1]]
        targetFC_tril = self.targetValue[tril_indices[0], tril_indices[1]]
        cor = torch.corrcoef(torch.stack((simFC_tril,targetFC_tril)))[0,1]
        
        return 0.5 - 0.5*cor
        
        
# zheng's version
    
class Costs:

    def __init__(self, method):
        self.method = method
    def cost_dist(self, sim, emp):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, the probability and negative log-likelihood.
        Parameters
        ----------
        logits_series_tf: tensor with node_size X datapoint
            simulated EEG
        labels_series_tf: tensor with node_size X datapoint
            empirical EEG
        """
        losses = torch.sqrt(torch.mean((sim - emp)**2))#
        return losses
    def cost_psd(self, sim, emp):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, the probability and negative log-likelihood.
        Parameters
        ----------
        logits_series_tf: tensor with node_size X datapoint
            simulated EEG
        labels_series_tf: tensor with node_size X datapoint
            empirical EEG
        """
        sp_sim = torch.fft.fftn(sim)
        sp_emp = torch.fft.fftn(emp)
        abs_sim = sp_sim.real**2 + sp_sim.imag**2
        abs_emp = sp_emp.real**2 + sp_emp.imag**2
        losses = torch.sqrt(torch.mean((abs_sim - abs_emp)**2))#
        return losses
    def cost_r(self, sim, emp):
        """
        Calculate the Pearson Correlation between the simFC and empFC.
        From there, the probability and negative log-likelihood.
        Parameters
        ----------
        logits_series_tf: tensor with node_size X datapoint
            simulated BOLD
        labels_series_tf: tensor with node_size X datapoint
            empirical BOLD
        """
        # get node_size(batch_size) and batch_size()
        node_size = sim.shape[0]
        truncated_backprop_length = sim.shape[1]
        mask = torch.tril_indices(node_size, node_size, -1)
        # fc for sim and empirical BOLDs
        fc_sim = torch.corrcoef(sim)
        fc_emp = torch.corrcoef(emp)
        # corr_coef
        corr_FC = torch.corrcoef(fc_sim[mask], fc_emp[mask])[0,1]
        # use surprise: corr to calculate probability and -log
        losses_corr = -torch.log(0.5000 + 0.5*corr_FC) #torch.mean((FC_v -FC_sim_v)**2)#
        return losses_corr
    def cost_eff(self, sim, emp):
        if self.method == 0:
            return self.cost_dist(sim,emp)
        elif self.method == 1:
            return self.cost_r(sim,emp)
        else:
            return self.cost_psd(sim,emp)
