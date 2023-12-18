import torch
import numpy as np

class Recording():
    '''    
    This class is responsible for holding timeseries of empirical and simulated data. It is: 
        - Part of the input and output of Model_fitting and Model_simulation[future]. 
        - It is the format expected by the visualization function of whobpyt. 
    
    However, the Recording class is not used within and across the NMM and Modals, as a simpler dictionary of tensors is used in those contexts. 

    Numerical simulation internals don't necessarily use this data structure, as calculations may be more efficient as the transpose: ts_length x num_regions.
    
    
    Attributes
    -------------
    data : Numpy Array or Tensor of dimensions num_regions x ts_length
        The time series data, either empirical or simulated
    step_size : Float
        The step size of the time points in the data class
    modality : String
        The name of the modality of the time series   
    numNodes : Int
        The number of nodes it time series.
    length : Int 
        The number of time points in the time series. 
    
    
    '''
        
    def __init__(self, data, step_size, modality = ""):
        '''
        
        Parameters
        -----------
        
        data : Numpy Array or Tensor of dimensions num_regions x ts_length
            The time series data, either empirical or simulated
        step_size : Float
            The step size of the time points in the data class
        modality : String
            The name of the modality of the time series
        
        '''
        
        
        if not(torch.is_tensor(data)):
            data = torch.tensor(data) # Store as Tensor
        
        self.data = data
        self.step_size = step_size
        self.modality = modality
        self.numNodes = self.data.shape[0]
        self.length = self.data.shape[1]
    
    def pyTS(self):
        '''
        Returns
        --------
        Tensor of num_regions x ts_length
        
        '''
        
        return self.data
    
    def npTS(self):
        '''
        Returns
        ---------
        Numpy Array of num_regions x ts_length
        
        '''
        
        return self.data.cpu().numpy()
        
    def npNodeByTime(self):
        '''
        Returns
        ---------
        Numpy Array of num_regions x ts_length        
        
        '''
        
        return self.data.cpu().numpy()
        
    def npTimeByNodes(self):
        '''
        Returns
        ---------
        Numpy Array of ts_length x num_regions        
        
        '''
        
        return self.data.cpu().numpy().T
        
    def length(self):
        '''
        Returns
        ---------
        The time series length
        
        '''   
        
        return self.length
        
    def npResample(self):
        '''
        This outputs resampled data used for figures (TODO: Not yet implemented)        
        
        '''
        
        pass
    
    def windowedTensor(self, TPperWindow):
        '''
        This method is called by the Model_fitting Class during training to reshape the data into windowed segments (adds another dimension).
        
        Parameters
        -----------
        TPperWindow : Int
            The number of time points in the window that will be back propagated
        
        Returns
        ---------
        Tensor: num_windows x num_regions x window_length
            The time series data in a windowed format
        '''
        
        node_size = self.data.shape[0]
        length_ts = self.data.shape[1]
        num_windows = int(length_ts / TPperWindow)
        data_out = np.zeros((num_windows, node_size, TPperWindow))
    
        for i_win in range(num_windows):
            data_out[i_win, :, :] = self.data[:, i_win * TPperWindow:(i_win + 1) * TPperWindow]
    
        return data_out
