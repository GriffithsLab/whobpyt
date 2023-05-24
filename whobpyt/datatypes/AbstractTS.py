# A fundamental datastructure that is a common output and input format to various whobpyt classes
# This can contain either empirical or simulated time series

class AbstractTS:
    
    def __init__(self):
        self.modality = ""
        self.numTS = 0
        self.step_size = 0
        self.length = 0
    
    def resample():
        pass
        
    def iterator():
        pass
        
    def getTS():
        pass
        
    def npTS():
        pass
        
    