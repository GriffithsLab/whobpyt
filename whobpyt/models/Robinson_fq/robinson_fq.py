"""
Authors: Zheng Wang, John Griffiths, Davide Momi, Kevin Kadak, Parsa Oveisi, Taha Morshedzadeh, Sorenza Bastiaens
Neural Mass Model fitting
module for Robinson with forward backward and lateral connection for EEG
"""

# @title new function PyTepFit

# Pytorch stuff


"""
Importage
"""
import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes.AbstractParams import AbstractParams
from whobpyt.datatypes.AbstractNMM import AbstractNMM
from whobpyt.models.Robinson_fq.ParamsCT_fq import ParamsCT_fq
from whobpyt.datatypes.parameter import par
import numpy as np  # for numerical operations


class RNNROBINSON_FQ(AbstractNMM):
    """
    A module for forward model (Robinson) to simulate PSD of EEG signals
    Attibutes
    ---------
   
    node_size: int
        the number of ROIs
    output_size: int
        the number of eeg channels
    Methods
    -------
    forward(input, noise_out, hx)
        forward model (Robinson) for generating a number of EEG signals with current model parameters
    """

    def __init__(self, node_size: int, output_size: int, param: ParamsCT_fq, use_fit_gains=False, use_fit_lfm=False) -> None:
        """
        Parameters
        ----------

        param from ParamJR
        """
        super(RNNROBINSON_FQ, self).__init__()

        self.param = param
        self.node_size = node_size
        self.output_size = node_size
        self.use_fit_gains = use_fit_gains
        self.use_fit_lfm = use_fit_lfm

    
       
    
    def setModelParameters(self):

        vars_name = [a for a in dir(self.param) if not a.startswith('__') and not callable(getattr(self.param, a))]
        for var in vars_name:
            if np.any(getattr(self.param, var)[1] > 0):
                if var == 'lm':
                    size = getattr(self.param, var)[1].shape
                    setattr(self, var, Parameter(
                        torch.tensor(getattr(self.param, var)[0] -np.ones((size[0], size[1])),
                            dtype=torch.float32)))
                    print(getattr(self, var))
                else:
                    setattr(self, var, Parameter(torch.tensor(getattr(self.param, var)[0] + getattr(self.param, var)[1] * np.random.randn(1, )[0],
                                 dtype=torch.float32)))

                if var not in ['std_in']:
                    dict_nv = {'m': getattr(self.param, var)[0], 'v': 1 / (getattr(self.param, var)[1]) ** 2}

                    dict_np = {'m': var + '_m', 'v': var + '_v_inv'}

                    for key in dict_nv:
                        setattr(self, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
            else:
                setattr(self, var, torch.tensor(getattr(self.param, var)[0], dtype=torch.float32))





    
    def forward(self, input):
        """
        Forward step in simulating the EEG signal.
        Parameters
        ----------
        input: list of frequencey

        Outputs
        -------
        next_state: pws with given frequence same size as input

        """

        # define some constants


        next_state = []


        for i_fq in range(input.shape[0]):
            #print(i_fq)
            omega = i_fq * 2*np.pi*torch.ones((self.node_size,1))
            j = complex(0, 1) # imaginary number
            s = omega * j
            tf = (self.alpha*self.beta) / ((s + self.alpha)*(s + self.beta))

            closed_loop_ei = ((self.eis*tf) / (1 - self.ii*tf) + self.es)
            closed_loop_rs = (self.sn * tf**3 *torch.exp(-s * torch.exp(self.t0_2))) / (1 - (self.sr* tf**2))

            q2r2 = (1 + s/ self.gamma)**2+self.K -self.ee*tf-tf**2/(1 - self.ii * tf) \
            *(self.eie + ((self.g_ese + self.g_esre * tf) * tf) \
                                  / (1 - self.srs * tf**2)*torch.exp(-s * torch.exp(self.t0_2)*2)) \
                -(self.g_ese + self.g_esre * tf)/(1 - self.srs * tf**2)*torch.exp(-s * torch.exp(self.t0_2)*2)*tf**2



            closed_loop_g = closed_loop_ei * closed_loop_rs * (1 / q2r2)
            #print(torch.abs(closed_loop_g))
            lm_n = self.lm/torch.sqrt((self.lm**2).sum())
            next_state.append(torch.exp(self.gain_tune)*torch.abs(torch.matmul(lm_n+0*j, closed_loop_g)))




        return torch.cat(next_state, dim=1)

