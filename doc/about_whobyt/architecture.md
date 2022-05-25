# WhoBPyT Code Base Architecture


## Module structure

data.py temporarily serves as artificial empirical EEG/fMRI data file, which defines a parameter class (ParamsJR) to simulate timeseries data.  models.py constructs a feed-forward JansenRitt RNN architectures, each of which simulate a batch of EEG and fMRI signal data.  objective.py compares modelled data from models.py with empirical data (presently from data.py), then calls fits.py to update model parameters based on loss function value.  viz.py takes either a model object and may graph functional connectivity or PSDs, or simply plot empirical data.

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




