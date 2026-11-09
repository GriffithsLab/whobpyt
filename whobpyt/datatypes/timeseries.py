import torch
import numpy as np

class Timeseries():
    """A recording or simulation: a regions (or channels) x time array.

    Empirical data are passed to the fitting classes as ``Timeseries``
    objects, and simulation results come back as ``Timeseries`` (in
    ``fit.lastRec``). The plotting functions in ``whobpyt.visualization``
    also take them. Inside the models, a plain dictionary of tensors is used
    instead.

    Parameters
    ----------
    data : ndarray or torch.Tensor, shape (num_regions, ts_length)
        The time series. NumPy arrays are converted to a tensor.
    step_size : float
        Time between samples, in seconds.
    modality : str
        Name of the modality, for example ``'eeg'``.

    Attributes
    ----------
    data : torch.Tensor, shape (num_regions, ts_length)
        The time series.
    step_size : float
        Time between samples, in seconds.
    modality : str
        Name of the modality.
    numNodes : int
        Number of regions or channels.
    length : int
        Number of time points.
    """

    def __init__(self, data, step_size, modality = ""):
        if not(torch.is_tensor(data)):
            data = torch.tensor(data) # Store as Tensor
        
        self.data = data
        self.step_size = step_size
        self.modality = modality
        self.numNodes = self.data.shape[0]
        self.length = self.data.shape[1]
    
    def pyTS(self):
        """Return the data as a tensor, shape ``(num_regions, ts_length)``."""
        
        return self.data
    
    def npTS(self):
        """Return the data as a NumPy array, shape ``(num_regions, ts_length)``."""
        return self.data.cpu().numpy()
        
    def npNodeByTime(self):
        """Return the data as a NumPy array, shape ``(num_regions, ts_length)``."""
        
        return self.data.cpu().numpy()
        
    def npTimeByNodes(self):
        """Return the transposed data as a NumPy array, shape ``(ts_length, num_regions)``."""
        
        return self.data.cpu().numpy().T
        
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
        
        data = self.data.detach().cpu().numpy() if torch.is_tensor(self.data) else np.asarray(self.data)
        node_size = data.shape[0]
        length_ts = data.shape[1]
        num_windows = int(length_ts / TPperWindow)
        data_out = np.zeros((num_windows, node_size, TPperWindow))
    
        for i_win in range(num_windows):
            data_out[i_win, :, :] = data[:, i_win * TPperWindow:(i_win + 1) * TPperWindow]
    
        return data_out
