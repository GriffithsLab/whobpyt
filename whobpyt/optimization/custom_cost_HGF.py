import torch
from whobpyt.datatypes.parameter import par
from whobpyt.datatypes.AbstractLoss import AbstractLoss

class CostsHGF(AbstractLoss):
    def __init__(self, model):
        self.mainLoss = CostsTS("x1")
        self.simKey = "x1"
        self.model = model

    def loss(self, simData: dict, empData: torch.Tensor):


        sim = simData
        emp = empData

        model = self.model

        # define some constants
        m = torch.nn.ReLU()

        loss_main = self.mainLoss.loss(sim, emp)#0.5-0.5*torch.corrcoef(torch.cat([sim[self.simKey], emp], dim=0))[0,1]#

        loss_EI = 0
        x2_window = sim['states'][:,0]
        x3_window = sim['states'][:,1]
        loss_EI = torch.mean(self.model.var_inv_2 * (x2_window - self.model.mu2) ** 2) \
                      + torch.mean(-torch.log(self.model.var_inv_2)) + \
                      torch.mean(self.model.var_inv_3 * (x3_window - self.model.mu3) ** 2) \
                      + torch.mean(-torch.log(self.model.var_inv_3))
       
        loss_prior = []

        variables_p = [a for a in dir(model.params) if (type(getattr(model.params, a)) == par)]

        for var_name in variables_p:
            
            var = getattr(model.params, var_name)
            if var.fit_hyper:
                loss_prior.append(torch.sum(( m(var.prior_var_inv)) * \
                                            (m(var.val) - m(var.prior_mean)) ** 2) \
                                  + torch.sum(-torch.log( m(var.prior_var_inv))))

        # total loss
        loss = 1*loss_main +  1*sum(loss_prior) + 0* loss_EI