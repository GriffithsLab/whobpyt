# Reduced Wong Wang Neural Mass Model (Version from Deco et al., 2014)

## Description:

	The Reduced Wang Wong (RWW) model as defined in (Deco et al., 2014). The model is combined with a baloon-windkessel hemodynamic model to output simulated BOLD. 
	
	There are many default and optional modifications that deviate from simply simulating the original model, so after parameters have been fit they should be validated using an unaltered version.


## Features and Modifications:

* Fitting RWW Weights
* Fitting Structural Connectivity Weights
* Negative Laplacian of the Structual Connectivity Matrix used
* Boundary Functions on State Variables
* ReLU functions applied to parameters to prevent certian parameters from changing sign in the equations
* A kind of downsampling between the RWW and the BOLD dynamics
* Faster BOLD dynamics to reduce computer memory requirement
* Custom objective function components involving hyperparameters
* Fitting of hyperparameters


## Usage:

See https://griffithslab.github.io/whobpyt/


## Equations & Biological Variables From:

- Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitationâ€“inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
- Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
- Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
- Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302. 