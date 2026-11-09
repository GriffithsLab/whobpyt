"""
Authors: Andrew Clappison, John Griffiths, Zheng Wang, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita
"""

import torch
from .parameter import Parameter as par
from torch.nn.parameter import Parameter as ptParameter
from torch.nn import Module as ptModule
import os
import subprocess

def get_git_commit_hash():
    """Return the git commit of the WhoBPyT source tree, or None.

    Only reports a hash when WhoBPyT itself is running from a git checkout
    (for example an editable install). It looks in the package's own
    directory, not the current working directory, so it never reports
    the commit of an unrelated repository, and it stays silent when git
    is unavailable or the package is not in a checkout.
    """
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        toplevel = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'], cwd=pkg_root,
            stderr=subprocess.DEVNULL).strip().decode('utf-8')
        if os.path.realpath(toplevel) != os.path.realpath(pkg_root):
            return None  # the package sits inside some other repository
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=pkg_root,
            stderr=subprocess.DEVNULL).strip().decode('utf-8')
    except (OSError, subprocess.CalledProcessError):
        # git not installed, or not a git checkout
        return None



class AbstractNeuralModel(ptModule):
    """Base class for models (typically neural mass models) that can be fitted.

    A model is a :class:`torch.nn.Module` that simulates one window of
    activity per call to :meth:`forward`. The measurement model (for example
    a lead field for EEG) may be part of the model.

    Subclasses should set ``state_names`` and ``output_names``, call
    :meth:`setModelParameters` in their ``__init__``, and implement
    :meth:`createIC`, :meth:`createDelayIC` and :meth:`forward`.

    Parameters
    ----------
    params : AbstractParams
        The model's parameters.

    Attributes
    ----------
    state_names : list of str
        Names of the model's state variables.
    output_names : list of str
        Names of the model's outputs. The first is used by the objective
        functions by default.
    track_params : list of str
        Names of the learnable parameters, recorded during training.
    params_fitted : dict
        ``'modelparameter'`` and ``'hyperparameter'`` lists of tensors that the
        optimizers update. Set by :meth:`setModelParameters`.
    commit_hash : str or None
        WhoBPyT's git commit, if running from a git checkout.
    """

    def __init__(self, params):
        super(AbstractNeuralModel, self).__init__() # May not want to enherit from torch.nn.Module in the future 
        self.params = params
        self.state_names = ["None"] # The names of the state variables of the model
        self.output_names = ["None"] # The variable to be used as output from the NMM, for purposes such as the input to an objective function
        self.track_params = [] # Which NMM Parameters to track over training
        self.commit_hash = get_git_commit_hash()
        
        
    def info(self):
        """Return the model's state names, output names, tracked parameters and commit."""
        
        return {"state_names": self.state_names, 
                "output_names": self.output_names,
                "track_params": self.track_params,
                "commit_hash": self.commit_hash}
            
    def setModelParameters(self):
        """Register the learnable parameters with PyTorch.

        Every :class:`~whobpyt.datatypes.Parameter` in ``params``
        with ``fit_par=True`` becomes a ``torch.nn.Parameter`` and is added to
        ``params_fitted['modelparameter']``; learnable priors
        (``fit_hyper=True``) are added to ``params_fitted['hyperparameter']``.
        Call this in the subclass's ``__init__``.
        """

        param_reg = []
        param_hyper = []

        
        var_names = [a for a in dir(self.params) if (type(getattr(self.params, a)) == par)]
        for var_name in var_names:
            var = getattr(self.params, var_name)
            if (var.fit_par):
                if var_name == 'lm':
                    size = var.val.shape
                    var.val = ptParameter(- 1 * torch.ones((size[0], size[1])) + 0.1*torch.randn(size[0], size[1])) 
                    var.prior_mean = ptParameter(var.prior_mean)
                    var.prior_precision = ptParameter(var.prior_precision)
                    param_reg.append(var.val)
                    if var.fit_hyper:
                        param_hyper.append(var.prior_mean)
                        param_hyper.append(var.prior_precision)
                    self.track_params.append(var_name)
                else:
                    var.val = ptParameter(var.val) # TODO: This is not consistent with what user would expect giving a variance
                    var.prior_mean = ptParameter(var.prior_mean)
                    var.prior_precision = ptParameter(var.prior_precision)
                    param_reg.append(var.val)
                    if var.fit_hyper:
                        param_hyper.append(var.prior_mean)
                        param_hyper.append(var.prior_precision)
                    self.track_params.append(var_name)



        self.params_fitted = {'modelparameter': param_reg,'hyperparameter': param_hyper}
        
    def createIC(self, ver):
        """Return initial conditions for the model's state variables."""
        pass
        
    def createDelayIC(self, ver):
        """Return the initial history of state variables needed for delays.

        The default returns a dummy tensor, for models without delays.
        """
        return torch.tensor(1.0) #Dummy variable if delays are not used
    
    def forward(self, external, hx, hE):
        """Simulate one window.

        Parameters
        ----------
        external : torch.Tensor
            External input for the window.
        hx : torch.Tensor
            Current state.
        hE : torch.Tensor
            History used for delayed inputs.

        Returns
        -------
        next_state : dict
            Simulated time series of each state variable and output, plus
            ``'current_state'``, the state at the end of the window.
        hE : torch.Tensor
            Updated history.
        """
        pass
