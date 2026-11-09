import torch
from .parameter import Parameter as par

class AbstractParams:
    """Base class for a model's parameter container.

    Subclasses set each model parameter as an attribute holding a
    :class:`~whobpyt.datatypes.Parameter`, typically with defaults
    that keyword arguments can override (see
    :class:`~whobpyt.models.jansen_rit.JansenRitParams`). The parameters may
    belong to the neural mass model and/or the measurement model.
    """

    def __init__(self, **kwargs):
        pass
    
    def getFittedNames(self):
        """Return the names of the learnable parameters.

        Returns
        -------
        list of str
            The name of every parameter with ``fit_par=True``, plus
            ``<name>_m`` and ``<name>_v_inv`` for every parameter whose
            prior mean and precision are learned (``fit_hyper=True``).
        """
        fp = []
        vars_names = [a for a in dir(self) if not a.startswith('__')]
        for var_name in vars_names:
            var = getattr(self, var_name)
            if isinstance(var, par):
                if var.fit_par:
                    fp.append(var_name)
                if var.fit_hyper:
                    fp.append(var_name + "_m")
                    fp.append(var_name + "_v_inv")
        return fp
        
    def to(self, device):
        """Move all parameters to ``device`` (a :class:`torch.device`)."""
        
        vars_names = [a for a in dir(self) if not a.startswith('__')]
        for var_name in vars_names:
            var = getattr(self, var_name)
            if (type(var) == par):
                var.to(device)
