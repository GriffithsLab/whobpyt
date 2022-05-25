# WhoBPyT Code Base Architecture


## Module structure



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




