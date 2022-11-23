# WhoBPyT Code Base Architecture


## Module structure

`data.py` temporarily serves as artificial empirical EEG/fMRI data file, which defines a parameter class (`ParamsJR`) to simulate timeseries data.  

`models.py` constructs a feed-forward JansenRitt RNN architectures, each of which simulate a batch of EEG and fMRI signal data.  

`objective.py` compares modelled data from `models.py` with empirical data (presently from data.py), then calls `fits.py` to update model parameters based on loss function value.  

`viz.py` takes either a model object and may graph functional connectivity or PSDs, or simply plot empirical data.


Run a simulation --> compare model data with empirical data --> fit model data to empirical data.



## API Usage

```python
from whobpyt.models import RNNJANSEN,ParamsJR 
from whobpyt.fit import fit_model

data_tofit = ''
connectivity = ''  
delays = '' 


# Simulate
params = ParamsJR('JR', A = [3.25, 0])
model = RNNJANSEN(params=params,cmat=connectivity=conn_mat,dmat=delayst))
sim_res = model.run()

# Fit
fitter = fit_model(model, erp_timeseries_data, num_epochs=50, use_priors=0)
fitter.run(num_epochs=50)
fitted_res = fitter.test()


## Notes
```

## System, Model, and Biophysical Parameters
### RWW_Params
```
G = 1
Lambda = 0 #1 or 0 depending on using long range feed forward inhibition (FFI)

#Excitatory Gating Variables
a_E # nC^(-1)
b_E # Hz
d_E # s
tau_E = tau_NMDA # ms
W_E 

#Inhibitory Gating Variables
a_I # nC^(-1)
b_I # Hz
d_I # s
tau_I # ms
W_I

#Setting other variables
w_plus # Local excitatory recurrence
J_NMDA # Excitatory synaptic coupling in nA
J # Local feedback inhibitory synaptic coupling. 1 in no-FIC case, different in FIC case #TODO: Currently set to J_NMDA but should calculate based on paper
gamma # a kinetic parameter in ms
sig # Noise amplitude at node in nA
#v_of_T # Uncorrelated standard Gaussian noise # NOTE: Now defined at time of running forward model
I_0 # The overall effective external input in nA

I_external # External input current 

#Starting Condition
#S_E # The average synaptic gating variable of excitatory 
#S_I # The average synaptic gating variable of inhibitory

#############################################
## Model Additions/modifications
#############################################

self.gammaI = 1/1000 # Zheng suggested this to get oscillations
```


### RWW_Layer
```
## EQUATIONS & BIOLOGICAL VARIABLES FROM:
# Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitation-inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
# Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
# Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
# Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.
```


