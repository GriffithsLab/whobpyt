import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes import AbstractNMM, AbstractParams, par
from whobpyt.models.HGF import ParamsHGF

class HGF(AbstractNMM):
    def __init__(self, paramsHGF, TRperWindow = 20, node_size =1, tr=1, step_size = .05,  use_fit_gains = False, output_size=1) -> None:
        super(HGF, self).__init__(paramsHGF)

        self.model_name = 'HGF'
        self.output_names =['x1']
        self.state_names = np.array(['x2', 'x3'])
        self.pop_size = 1  # 3 populations JR
        self.state_size = 2  # 2 states in each population
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = torch.tensor(step_size, dtype=torch.float32)  # integration step 0.1 ms
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRperWindow  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.output_size = output_size  # num of EEG channels
        self.use_fit_gains = use_fit_gains
        #self.params = paramsHGF
        self.setModelParameters()
        self.setHyperParameters()


    def forward(self, externa=None,  X=None, hE=None):
        omega3 = self.params.omega_3.value()
        omega2 = self.params.omega_2.value()
        kappa = self.params.kappa.value()
        x2mean = self.params.x2mean.value()
        deca2 = self.params.deca2.value()
        deca3 = self.params.deca3.value()
        c = self.params.c.value()
        g_x2_x3 = self.params.g_x2_x3.value()
        g_x3_x2 = self.params.g_x3_x2.value()
        dt = self.step_size
        next_state = {}

        state_windows = []
        x1_windows = []
        x2=X[:,0:1]
        x3=X[:,1:2]
        for TR_i in range(self.TRs_per_window):
            x2_tmp =[]
            x3_tmp =[]
            # Since tr is about second we need to use a small step size like 0.05 to integrate the model states.
            for step_i in range(self.steps_per_TR):


                x3_new = x3 -dt*(deca3*x3-g_x2_x3*x2)+torch.sqrt(dt)*torch.randn(self.node_size,1)*omega3
                x2_new = x2 -dt*(deca2*x2- g_x3_x2*x3)+ torch.sqrt(dt)*torch.randn(self.node_size,1)*omega2*torch.exp(kappa*x3)

                x2_tmp.append(x2_new)
                x3_tmp.append(x3_new)
                x2 = 10*torch.tanh(x2_new/10)
                x3 = 10*torch.tanh(x3_new/10)
            #x2 = sum(x2_tmp)/self.steps_per_TR#10*torch.tanh(x2_new/10)
            #x3 = sum(x3_tmp)/self.steps_per_TR#10*torch.tanh(x3_new/10)
            state_windows.append(torch.cat([x2, x3], dim =1)[:,:,np.newaxis])
            x1_windows.append(x2-x2mean)
            #x1_windows.append(1/(1+torch.exp(-c*(x2-k))))
        next_state['x1'] = torch.cat(x1_windows, dim =1)
        next_state['states'] = torch.cat(state_windows, dim =2)
        next_state['current_state'] = torch.cat([x2, x3], dim =1)
        return next_state, hE


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
        return torch.tensor(0 * np.random.uniform(-0.02, 0.02, (self.node_size, self.state_size))+np.array([0.0,0.5]), dtype=torch.float32)


    def setHyperParameters(self):
        """
        Sets the parameters of the model.
        """



        # Set w_bb, w_ff, and w_ll as attributes as type Parameter if use_fit_gains is True
        self.mu2 = Parameter(torch.tensor(0.0*np.ones((self.node_size, 1)), # the lateral gains
                                                dtype=torch.float32))
        self.mu3 = Parameter(torch.tensor(1.0*np.ones((self.node_size, 1)), # the lateral gains
                                                dtype=torch.float32))
        self.var_inv_2 = Parameter(torch.tensor(.1*np.ones((self.node_size, 1)), # the lateral gains
                                                dtype=torch.float32))
        self.var_inv_3 = Parameter(torch.tensor(.1*np.ones((self.node_size, 1)), # the lateral gains
                                                dtype=torch.float32))
        self.params_fitted['hyperparameter'].append(self.mu2)
        self.params_fitted['hyperparameter'].append(self.mu3)
        self.params_fitted['hyperparameter'].append(self.var_inv_2)
        self.params_fitted['hyperparameter'].append(self.var_inv_3)