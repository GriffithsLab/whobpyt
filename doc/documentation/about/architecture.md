Code Architecture
===================================

The package is a collection of interchangable Whole Brain Models, Objective Functions Components, and Parameter Fitting Paradigms. This code base is still in alpha phase and not all combinations of these components are currently supported. 

## Simplified Usage Pseudo Code

```python
import whobpyt

# Making the training data
training_input = [stimulus1, ..., stimulusN] # This can be 0 for the resting-state case
empirical_data = [Recording(data1), ..., Recording(dataN)]

# Create the CNMM from whobpyt.models
params = AbstractParams(P = par(value, fit = True)) # specify which parameters will be fit
model = AbstractNMM(params)

#Define an objective function from whobpyt.optimization
objective = AbstractLoss(key = "variable") # specify which variable to use in objective function

# Train the Model using a paradigm from whobpyt.run
fitting = AbstractFitting(model, objective)
fitting.train(training_input, empirical_data, epochs = 100, lr = 0.1)

# Evaluate the Training
plot(fitting.trainingStats)

# Evaluate the Found Parameters
verify_model = NumPyNMM(model.params)
simulated_data = verify_model.simulate()

# Preform Analysis
...

```

## Whole Brain Models: 

These models implement the numerical simulation of a CNMM (or modified CNMM). They can be combined with modalities, or may have an integrated modality. 

The built in models are:

- RWWExcInb - Two variations are avaliable
- JansenRit - With Lead Field, Delays, Laplacian Connections
- Linear (needs updating)
- Robinson (Future Addition)

## Objective Function Components:

Some objective function components can be used individually, others such as biological priors are addons. 

The built in objective functions are:

- Functional Connectivity Correlation
- Time Series Correlation
- Power Spectral Density Difference 
- Target Mean Value of a State Variable
- Biological Priors of Parameters

## Parameter Fitting Paradigms: 

Paradigms for fitting model parameters.

The built in parameter fitting paradigms are:

- Model Fitting - Uses a approch to train on windowed sections of neuroimaging recordings
- Fitting FNGFPG - A technique to run true time scale BOLD 


## Package Expansions

Further extensions are encouraged, and abstract classes have been created to help facilitate that. In addition to providing some free functionality, the abstract classes act as a template for what methods to implement in order for the new addition to work with the other classes in WhoBPyT.

Abstract classes to inherit from:

- AbstractParams
- AbstractNMM
- AbstractMode
- AbstractLoss
