# WhoBPyT Data Types

This folder contains abstract classes for implementing new NMM and Modality Models, Objective Functions.

This folder also contains classes for par, Recording, and TrainingStats.

## Abstract Data Classes

Abstract classes for NMM Parameteters, NMM Models, Modalities, and Loss functions. (future to also have a abstract classes for training/simulation classes).

These classes provide some functionality that can be inherited, but also a template for attributes and methods that are expected to be present to function correctly with the rest of WhoBPyT.

## par Class

The class to hold one model parameter.
* Can be a single value or have one value per brain region.
* Feature for also having prior mean and variance for the parameter.
* Feature to store the value as the log(value) so the parameter will not go negative.

## Recording Class

The class to hold empirical or simulated time series.
* Has attributes for time series metadata.
* Will have methods for returning resampled and restructured versions of the time series.

## TrainingStats Class

This class is for storing statistics during model training.
* Model performance over training windows/epochs.
* Model parameters over training windows/epochs.