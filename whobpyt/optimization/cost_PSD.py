import torch


class CostsPSD():
    # TODO: Deal with num_region vs. num_channels vs. num_parcels conflict with variable naming
    def __init__(self, num_regions, simKey, sampleFreqHz, targetValue = None, empiricalData = None):
        self.num_regions = num_regions
        self.simKey = simKey  # This is the index in the data simulation to extract variable time series from
        
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
        
        sdAxis, sdValues = powerSpectrumLoss.calcPSD(simData[:, :, self.simKey], sampleFreqHz = self.sampleFreqHz, minFreq = 2, maxFreq = 40)
        sdAxis_dS, sdValues_dS = powerSpectrumLoss.downSmoothPSD(sdAxis, sdValues, numPoints = 32)
        sdAxis_dS, sdValues_dS_scaled = powerSpectrumLoss.scalePSD(sdAxis_dS, sdValues_dS)

        return torch.nn.functional.mse_loss(sdValues_dS_scaled, self.targetValue)