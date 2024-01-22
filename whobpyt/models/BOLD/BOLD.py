import torch
from whobpyt.datatypes import AbstractNMM
from whobpyt.models.BOLD import ParamsBOLD
import numpy as np
from whobpyt.functions.arg_type_check import method_arg_type_check
   
class RNNBOLD(AbstractNMM):
    '''
    Balloon-Windkessel Hemodynamic Response Function Forward Model
    
    Equations & Biological Variables From:
    
    Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.  
    
    Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
    '''

    def __init__(self, params: ParamsBOLD, node_size = 68, TRs_per_window = 20, step_size = 0.05,  \
                   tr=1.0):
        """
        Parameters
        ----------
            
        node_size: int
            The number of ROIs
        TRs_per_window: int
            The number of BOLD TRs to simulate in one forward call    
        step_size: float
            Integration step for forward model
        tr : float
            tr of fMRI image. That is, the spacing betweeen images in the time series. 
        
            Whether to fit the structural connectivity, will fit via connection gains: exp(gains_con)*sc
        params: ParamsRWW
            A object that contains the parameters for the RWW nodes.
        """        
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types
        
        super(RNNBOLD, self).__init__(params)
        
        self.state_names = np.array(['x', 'f', 'v', 'q'])
        self.output_names = ["bold"]
        
        
        self.model_name = "RNNBOLD"
        self.state_size = 4  # 6 states WWD model
        # self.input_size = input_size  # 1 or 2
        self.tr = tr  # tr fMRI image
        self.step_size = step_size  # integration step 0.05
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
       
        
        self.output_size = node_size
        
        self.setModelParameters()
        
    
    
    
    def createIC(self, ver):
        """
        
            A function to return an initial state tensor for the model.    
        
        Parameters
        ----------
        
        ver: int
            Ignored Parameter
        
        
        Returns
        ----------
        
        Tensor
            Random Initial Conditions for RWW & BOLD 
        
        
        """
        
        # initial state
        return torch.tensor(0.2 * np.random.uniform(0, 1, (self.node_size, self.pop_size, self.state_size)) + np.array(
                [1.0, 1.0, 1.0]), dtype=torch.float32)

    
    
    
        
    def forward(self, external, hx):
        """
        
        Forward step in simulating the BOLD signal.
        
        Parameters
        ----------
        
        
        hx: tensor with node_size x state_size
            states of Ballon model
        
        Returns
        -------
        next_state: dictionary with Tensors
            Tensor dimension [Num_Time_Points, Num_Regions]
        
        with keys: 'current_state''states_window''bold_window'
            record new states and BOLD
            
        """
        # Generate the ReLU module for model parameters gEE gEI and gIE
        m = torch.nn.ReLU()
    
    
        
    
        # Output (BOLD signal)
        alpha = self.params.alpha.value()
        rho = self.params.rho.value()
        k1 = self.params.k1.value()
        k2 = self.params.k2.value()
        k3 = self.params.k3.value()  # adjust this number from 0.48 for BOLD fluctruate around zero
        V = self.params.V.value()
        E0 = self.params.E0.value()
        tau_s = self.params.tau_s.value()
        tau_f = self.params.tau_f.value()
        tau_0 = self.params.tau_0.value()
        
    
    
        next_state = {}
    
        # hx is current state (6) 0: E 1:I (neural activities) 2:x 3:f 4:v 5:f (BOLD)
    
        x = hx[:,:,0]
        f = hx[:,:,1]
        v = hx[:,:,2]
        q = hx[:,:,3]
    
        dt = torch.tensor(self.step_size, dtype=torch.float32)
    
        
        bold_window = []
        states_window = []
        
        
        # Use the forward model to get neural activity at ith element in the window.
        
        

        for TR_i in range(self.TRs_per_window):

            for step_i in range(self.steps_per_TR):
                x_next = x + 1 * dt * (external[:, :, TR_i, step_i] - torch.reciprocal(tau_s) * x \
                         - torch.reciprocal(tau_f) * (f - 1))
                f_next = f + 1 * dt * x
                v_next = v + 1 * dt * (f - torch.pow(v, torch.reciprocal(alpha))) * torch.reciprocal(tau_0)
                q_next = q + 1 * dt * (f * (1 - torch.pow(1 - rho, torch.reciprocal(f))) * torch.reciprocal(rho) \
                         - q * torch.pow(v, torch.reciprocal(alpha)) * torch.reciprocal(v)) * torch.reciprocal(tau_0)
    
                x = torch.tanh(x_next)
                f = (1 + torch.tanh(f_next - 1))
                v = (1 + torch.tanh(v_next - 1))
                q = (1 + torch.tanh(q_next - 1))
            
            

            # Put the BOLD signal each tr to the placeholder being used in the cost calculation.
            bold_window.append((0.01 * torch.randn(self.node_size, 1) +
                                    100.0 * V * torch.reciprocal(E0) *
                                    (k1 * (1 - q) + k2 * (1 - q * torch.reciprocal(v)) + k3 * (1 - v))))
        
            states_window.append(torch.cat([x[:,:,np.newaxis] , f[:,:,np.newaxis], v[:,:,np.newaxis],\
                                            q[:,:,np.newaxis]], dim =len(x.shape))[:,:,:,np.newaxis])
                                            
        # Update the current state.
        current_state = torch.cat([x[:,:, np.newaxis],\
                  f[:,:, np.newaxis], v[:,:, np.newaxis], q[:,:, np.newaxis]], dim=len(x.shape))
        next_state['current_state'] = current_state
        next_state['bold'] = torch.cat(bold_window, dim =len(x.shape)-1)
        next_state['states'] = torch.cat(states_window, dim=len(current_state.shape))
        
        
        return next_state

