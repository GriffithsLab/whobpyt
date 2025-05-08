Background Information
===================================

The Whole Brain Models are of the form of Connectome-based Neural Mass Models (CNMM), with additional modality equations to produce simulated neuroimaging data. 


## Neuroimaging Data

The following items may be requried (in a consistent parcellated format), depending on the CNMM model and empirical data:

- Structural Connectivity Matrix : For connection strengths
- Distance Matrix : For connection delays
- Lead Field Matrix : To convert from CNMM source space to EEG channel space (if empirical EEG data in channel space is used)
- One or more neuroimaging time series modalities, or a derivative information such as functional connectivity



## System, Model, and Biophysical Parameters

### RWWExcInb

This model is typically used for simulating fMRI BOLD. 

```
## EQUATIONS & BIOLOGICAL VARIABLES FROM:
# Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M. How local excitation-inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience. 2014 Jun 4;34(23):7886-98.
# Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M. Resting-state functional connectivity emerges from structurally and dynamically shaped slow linear fluctuations. Journal of Neuroscience. 2013 Jul 3;33(27):11239-52.
# Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006 Jan 25;26(4):1314-28.
# Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage. 2003 Aug 1;19(4):1273-302.
```

### Jansen-Rit

This model is typically used for simulating EEG.

```
## EQUATIONS & BIOLOGICAL VARIABLES FROM:
# Jansen BH, Rit VG. Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. Biological cybernetics. 1995 Sep;73(4):357-66.
```

