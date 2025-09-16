# PyTorch stuff
import torch
from torch.nn.parameter import Parameter as ptParameter
from torch.nn import ReLU as ptReLU
from torch.linalg import norm as ptnorm
from torch import (tensor as pttensor, float32 as ptfloat32, sum as ptsum, exp as ptexp, diag as ptdiag, 
                   transpose as pttranspose, zeros_like as ptzeros_like, int64 as ptint64, randn as ptrandn, 
                   matmul as ptmatmul, tanh as pttanh, matmul as ptmatmul, reshape as ptreshape, sqrt as ptsqrt,
                   ones as ptones, cat as ptcat)

# Numpy stuff
from numpy.random import uniform 
from numpy import ones,zeros
import numpy as np

# WhoBPyT stuff
from ...datatypes import AbstractNeuralModel, AbstractParams, Parameter as par
from ...functions.arg_type_check import method_arg_type_check# ...




class ParamsLinearFreqs(AbstractParams):

    def __init__(self, **kwargs):
        """
        Initializes the ParamsLinearFreqs object.

        Args:
            **kwargs: Keyword arguments for the model parameters.

        Returns:
            None
        """

        super(ParamsLinearFreqs, self).__init__(**kwargs)
        param = {
            "std_in": par(0.1),

            "eigvals": par(1),
            "mu": par(5)
        }
        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])

class LINEAR_FQ(AbstractNeuralModel):
    """
    A module for Robinson model from freqency to power spectrum
    Attibutes
    ---------
    """
    model_name = "LINEAR_FQ"

    def __init__(self, params: ParamsLinearFreqs, node_size = 200, mode_size = 20, output_size = 64,  sc_eigvecs =np.ones((200,200)), \
                 dist =np.ones((200,200)), use_fit_gains=False, use_fit_lfm=False):
        """
        Parameters
        ----------

        param from ParamJR
        """
        super(LINEAR_FQ, self).__init__(params)

        self.params = params
        self.node_size = node_size
        self.mode_size = mode_size
        self.output_size = node_size
        self.use_fit_gains = use_fit_gains
        self.use_fit_lfm = use_fit_lfm
        self.sc_eigvecs = sc_eigvecs
        self.dist = dist

        self.setModelParameters()






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
        # Generate the ReLU module
        m = torch.nn.ReLU()
        # define some constants
        std_in = 0.00001 + m(self.params.std_in.value())
        g = 0.00001 + m(self.params.g.value())
        a = 0.00001 + m(self.params.a.value())
        b = 0.00001 + m(self.params.b.value())
        A = 0.00001 + m(self.params.A.value())
        B = 0.00001 + m(self.params.B.value())
        C2 = 0.00001 + m(self.params.C2.value())
        C1 = 0.00001 + m(self.params.C1.value())
        c = 0.00001 + m(self.params.c.value())
        n_mode = self.mode_size
        eigvals = 0  + m(self.params.eigvals.value()[:n_mode])/m(self.params.eigvals.value()[:n_mode]).max()
        mu = 0.02  + m(self.params.mu.value())
        dist = torch.tensor(self.dist, dtype=torch.float32)
        u_sc = torch.tensor(self.sc_eigvecs, dtype=torch.float32)
        lm = self.params.lm.value()
        tau_mode = m(u_sc.T @ dist/mu @ u_sc)
        sc = u_sc[:,:n_mode] @ torch.diag(eigvals[:,0]) @ (u_sc[:,:n_mode]).T



        next_state = []


        for i_fq in range(input.shape[0]):
            #print(i_fq)
            omega = input[i_fq] * 2*np.pi
            j = complex(0, 1) # imaginary number
            s = omega * j
            tf_e = A*a/(s**2 +2*a*s +a**2 )
            tf_i = B*b/(s**2 +2*b*s +b**2 )
            tf_ei = (1-0*C1*tf_i)*tf_e/(1+ C1*C2*tf_e*tf_i)
            tf_close = (1/(s+c))*tf_ei*torch.linalg.inv(1+g* torch.exp(-s*tau_mode)*sc*tf_ei)

            """lap = torch.diag((u_sc @ (torch.diag(eigvals[:,0]) ) @ u_sc.T).sum(1)) \
                - (u_sc @ (torch.diag(eigvals) ) @ u_sc.T)
            u_l, d_l, l_l = torch.svd(lap)"""
            tf = std_in * tf_close.sum(0)[:,np.newaxis]




            #print(torch.abs(closed_loop_g))
            lm_n = lm/torch.sqrt((lm**2).sum())
            next_state.append(torch.abs(torch.matmul((lm_n + 0*j), tf)))




        return torch.cat(next_state, dim=1)