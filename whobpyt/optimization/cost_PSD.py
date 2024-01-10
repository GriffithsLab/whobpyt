import torch
from warnings import warn
from whobpyt.datatypes.AbstractLoss import AbstractLoss

class CostsPSD(AbstractLoss):
    '''
    WARNING: This function is no longer supported.
    TODO: Needs to be updated. 
    '''
    # TODO: Deal with num_region vs. num_channels vs. num_parcels conflict with variable naming
    def __init__(self, num_regions, simKey, sampleFreqHz, targetValue = None, empiricalData = None):
        super(CostsPSD, self).__init__(simKey)
        self.num_regions = num_regions
        self.simKey = simKey  # This is the index in the data simulation to extract variable time series from
        
        self.sampleFreqHz = sampleFreqHz
        
        if targetValue != None:
            self.targetValue = targetValue
        
        if empiricalData != None:
            # In the future, if given empiricalData then will calculate the target value in this initialization function. 
            # That will possibly involve a time series of targets, for which then the loss would need a parameter to identify
            # which one to fit to.
            pass
        warn(f'{self.__class__.__name__} will be deprecated.', DeprecationWarning, stacklevel=2)


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

    
    def loss(self, simData, empData = None):
        # simData assumed to be dict with values in the form [time_steps, regions or channels, one or more variables]
        # Returns the MSE of the difference between the simulated and target power spectrum
        sim = simData[self.simKey]
        
        sdAxis, sdValues = powerSpectrumLoss.calcPSD(sim[:, :, self.simKey], sampleFreqHz = self.sampleFreqHz, minFreq = 2, maxFreq = 40)
        sdAxis_dS, sdValues_dS = powerSpectrumLoss.downSmoothPSD(sdAxis, sdValues, numPoints = 32)
        sdAxis_dS, sdValues_dS_scaled = powerSpectrumLoss.scalePSD(sdAxis_dS, sdValues_dS)

        return torch.nn.functional.mse_loss(sdValues_dS_scaled, self.targetValue)


class CostsFixedPSD(AbstractLoss):
    """
    Updated Code that fits to a fixed PSD
    
    The simulated PSD is generated as the square of the Fast Fourier Transform (FFT). A particular range to fit on is selected. The mean is not removed, so it is recommended to set the range such as to disclude the first point of the PSD. Removing an initial transient period before calculating the PSD is also recommended. 
    
    Designed for Fitting_Batch, where the model output has an extra dimension for batch. TODO: Generalize further to work in case without this dimension as well. 
     
    NOTE: If using batching, the batches will be averaged before calculating the error (as opposed to having an error for each time series in the batch). 
    
    Has GPU support.
    
    Attributes
    ----------
    simKey: String
        Name of the state variable or modality to be used as input to the cost function. 
    num_regions: Int
        The number of nodes in the model. 
    batch_size: Int
        The number of elements in the batch. 
    rmTransient: Int
        The number of initial time steps of simulation to remove as the transient. Default: 0
    device: torch.device
        Whether to run the objective function on CPU or GPU.
    sampleFreqHz: Int
        The sampling frequency of the data.        
    targetValue: torch.tensor
        The target PSD. This is assumed to be created with the same frequency range and density as that of the PSD generated from the simulated data. 
    empiricalData: torch.tensor
        NOT IMPLEMENTED: This is a placeholder for the case of getting an empirical timeseries as input, which would be applicable if doing a windowed fitting paradigm
       
    
    """
    
    def __init__(self, num_regions, simKey, sampleFreqHz, minFreq, maxFreq, targetValue = None, empiricalData = None, batch_size = 1, rmTransient = 0, device = torch.device('cpu')):
        """
        
        
        Parameters
        ----------
        num_regions; Int
            The number of nodes in the model.             
        simKey: String
            Name of the state variable or modality to be used as input to the cost function.         
        sampleFreqHz: Int
            The sampling frequency of the data.
        minFreq: Int
            The minimum frequnecy of the PSD to return.
        maxFreq: Int
            The maximum frequency of the PSD to return.            
        targetValue: torch.tensor
            The sampling frequency of the data.
        empiricalData: torch.tensor
            NOT IMPLEMENTED: This is a placeholder for the case of getting an empirical timeseries as input, which would be applicable if doing a windowed fitting paradigm            
        batch_size: Int
            The number of elements in the batch. 
        rmTransient: Int
            The number of initial time steps of simulation to remove as the transient. Default: 0        
        device: torch.device
            Whether to run the objective function on CPU or GPU.        
        
        """
        super(CostsFixedPSD, self).__init__(simKey)
        
        self.num_regions = num_regions
        self.simKey = simKey  # This is the index in the data simulation to extract variable time series from
        self.batch_size = batch_size
        self.rmTransient = rmTransient
        
        self.device = device
        
        self.sampleFreqHz = sampleFreqHz #
        self.minFreq = minFreq #
        self.maxFreq = maxFreq #
        
        if targetValue != None:
            if self.batch_size == 1:
                self.targetValue = targetValue.repeat(num_regions, 1).to(device)
            else:
                self.targetValue = targetValue.repeat(num_regions, 1).to(device) #TODO: Currently taking mean before MSE from all batches, need to document this
        
        if empiricalData != None:
            # In the future, if given empiricalData then will calculate the target value in this initialization function. 
            # That will possibly involve a time series of targets, for which then the loss would need a parameter to identify
            # which one to fit to.
            pass
    
    def calcPSD(self, signal, sampleFreqHz, minFreq = None, maxFreq = None, axMethod = 2):
        """
        This method calculates the Power Spectral Density (PSD) as the square of the Fast Fourier Transform (FFT). 
        
        Tested when working with the default simulation frequency of 10,000Hz. 
        
        Parameters
        ----------
        signal: dict of torch.tensor
            The timeseries outputted by a model. Dimensions: [nodes, time, batch]
        sampleFreqHz: Int
            The sampling frequency of the data.
        minFreq: Int
            The minimum frequnecy of the PSD to return.
        maxFreq: Int
            The maximum frequency of the PSD to return.
        axMethod: Int
            Either 1 or 2 depending on the approach to calculate the PSD axis.
        
        Returns
        -------
        sdAxis: torch.tensor
            The axis values of the PSD
        sdValues: torch.tensor
            The PSD values [____, ____]
        """
        
        
        # signal assumed to be in the form of [time_steps, regions or channels] <- This is being changed
        # Returns the Power Spectrial Density with associated Hz values
    
        if (self.rmTransient > 0):
            signal = signal[:,self.rmTransient:,:]
    
        N = signal.shape[1]
        
        # These defines the range of the PSD to return
        if minFreq == None:
            minFreq = 0
        if maxFreq == None:
            maxFreq = sampleFreqHz//2
        
        # Not sure which axis method is correct
        if axMethod == 1:
            fftAxis = torch.linspace(0,sampleFreqHz, N).to(self.device)
        elif axMethod == 2:
            fftAxis = (torch.arange(N)*sampleFreqHz/N).to(self.device)
        
        # Take the FFT of the Signal
        signalFFT = torch.fft.fft(signal, dim = 1)
        
        # Take the square value (it is complex) so following is equivalent:
        # Square of absolute, or
        # Sum of real squared and imag squared
        spectralDensity = torch.abs(signalFFT[:,:N//2,:])**2
        
        #Filter to the desired range
        minPoint = int((minFreq/(sampleFreqHz//2))*(N//2))
        maxPoint = int((maxFreq/(sampleFreqHz//2))*(N//2))+1
        sdAxis = fftAxis[minPoint:maxPoint]
        sdValues = spectralDensity[:,minPoint:maxPoint,:]
        
        return sdAxis, sdValues
               
    def loss(self, simData, empData = None):
        """
        
        NOTE: If using batching, the batches will be averaged before calculating the error (as opposed to having an error for each simulated time series in the batch).
        
        Parameters
        ----------
        simData: torch.tensor
            Simulated Data in the form [regions, time_steps, block/batch]
        empData: torch.tensor
            NOT IMPLEMENTED: This is a placeholder for the case of getting an empirical timeseries as input, which would be applicable if doing a windowed fitting paradigm
        
        
        Returns
        -------
        psdMSE:
            The MSE of the difference between the simulated and target power spectrum within the specified range
        
        """
        sim = simData[self.simKey]
        
        psdAxis, psdValues = self.calcPSD(sim, sampleFreqHz = self.sampleFreqHz, minFreq = self.minFreq, maxFreq = self.maxFreq) # TODO: Sampling frequency of simulated data and target time series is currently assumed to be the same.
        
        meanValue = torch.mean(psdValues, 2)
        
        psdMSE = torch.nn.functional.mse_loss(meanValue, self.targetValue)
        
        return psdMSE
