# WhoBPyT Code Base Architecture


## Module structure



## API Usage

```python
from whobpyt.models import RNNJANSEN,ParamsJR 

data = '' # 
params = ParamsJR('JR', A = [3.25, 0])
model = RNNJANSEN(input_size, node_size, batch_size, step_size, output_size, tr, sc, lm, dist, True, False, params)

fitter = fit_model(model, data, num_epochs=50, 0)
trained = fitter.train(u=u)
test = fitter.test(X0, hE0, base_batch_num, u=u)



## Notes








