# Reproducing Abeysuriya 2015 outputs and BrainTrak equations with the forward function

# Importage:

# os stuff
import os
import sys
sys.path.append('..')

# whobpyt stuff
import whobpyt
from whobpyt.datatypes import Parameter as par
from whobpyt.models.robinson import RobinsonLinModel, RobinsonLinParams

# python stuff
import numpy as np
import pandas as pd
import scipy.io
import pickle
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
#neuroimaging packages
import mne

# viz stuff
import matplotlib.pyplot as plt

def parameter_set(pars):
  if pars == 'EC':
    # Eyes closed
    params = RobinsonLinParams( Gei = par(-4.11),
                                Gee = par(2.070),
                                Ges = par(0.77),
                                Gsn = par(8.10),
                                Gse = par(7.77),
                                Gsr = par(-3.3),
                                Grs = par(0.2),
                                Gre = par(0.66),
                                alpha = par(83),
                                beta = par(769),
                                gammae = par(116),
                                t0 = par(0.085),
                                re = par(0.086),
                                phin = par(1e-5),
                                EMG = par(0),
                                kmax = par(4),
                                k0 = par(10),
                                Lx = par(0.5)
                                )

  if pars == 'EO':
    # Eyes open
    params = RobinsonLinParams( Gei = par(-13.22),
                                Gee = par(10.50),
                                Ges = par(1.21),
                                Gsn = par(14.23),
                                Gse = par(5.78),
                                Gsr = par(-2.83),
                                Grs = par(0.25),
                                Gre = par(0.85),
                                alpha = par(83),
                                beta = par(769),
                                gammae = par(116),
                                t0 = par(0.085),
                                re = par(0.086),
                                phin = par(1e-5),
                                EMG = par(0),
                                kmax = par(4),
                                k0 = par(10),
                                Lx = par(0.5)
                                )      
  if pars == 'REM':
    # REM
    params = RobinsonLinParams( Gei = par(-6.61),
                                Gee = par(5.87),
                                Ges = par(0.21),
                                Gsn = par(0.68),
                                Gse = par(0.66),
                                Gsr = par(-0.28),
                                Grs = par(4.59),
                                Gre = par(2.08),
                                alpha = par(45),
                                beta = par(185),
                                gammae = par(116),
                                t0 = par(0.085),
                                re = par(0.086),
                                phin = par(1e-5),
                                EMG = par(0),
                                kmax = par(4),
                                k0 = par(10),
                                Lx = par(0.5)
                                )   
  if pars == 'S1':
    # S1
    params = RobinsonLinParams( Gei = par(-8.30),
                                Gee = par(7.45),
                                Ges = par(0.31),
                                Gsn = par(3.90),
                                Gse = par(1.67),
                                Gsr = par(-0.40),
                                Grs = par(4.44),
                                Gre = par(7.47),
                                alpha = par(45),
                                beta = par(185),
                                gammae = par(116),
                                t0 = par(0.085),
                                re = par(0.086),
                                phin = par(1e-5),
                                EMG = par(0),
                                kmax = par(4),
                                k0 = par(10),
                                Lx = par(0.5)
                                ) 
  if pars == 'S2':
    # S2
    params = RobinsonLinParams( Gei = par(-17.93),
                                Gee = par(16.8),
                                Ges = par(3.89),
                                Gsn = par(2.38),
                                Gse = par(0.07),
                                Gsr = par(-0.14),
                                Grs = par(8.33),
                                Gre = par(4.96),
                                alpha = par(45),
                                beta = par(185),
                                gammae = par(116),
                                t0 = par(0.085),
                                re = par(0.086),
                                phin = par(1e-5),
                                EMG = par(0),
                                kmax = par(4),
                                k0 = par(10),
                                Lx = par(0.5)
                                )  
  if pars == 'SWS':
    # SWS
    params = RobinsonLinParams( Gei = par(-19.74),
                                Gee = par(19.52),
                                Ges = par(5.30),
                                Gsn = par(1.70),
                                Gse = par(0.22),
                                Gsr = par(-0.22),
                                Grs = par(1.35),
                                Gre = par(1.90),
                                alpha = par(45),
                                beta = par(185),
                                gammae = par(116),
                                t0 = par(0.085),
                                re = par(0.086),
                                phin = par(1e-5),
                                EMG = par(0),
                                kmax = par(4),
                                k0 = par(10),
                                Lx = par(0.5)
                                )   
  if pars == 'Spindles':
    # Spindles
    params = RobinsonLinParams( Gei = par(-18.96),
                                Gee = par(18.52),
                                Ges = par(2.55),
                                Gsn = par(2.78),
                                Gse = par(0.73),
                                Gsr = par(-0.26),
                                Grs = par(16.92),
                                Gre = par(4.67),
                                alpha = par(45),
                                beta = par(185),
                                gammae = par(116),
                                t0 = par(0.085),
                                re = par(0.086),
                                phin = par(1e-5),
                                EMG = par(0),
                                kmax = par(4),
                                k0 = par(10),
                                Lx = par(0.5)
                                )   
  return params

freq = np.linspace(1,60,200)
w1 = 2*math.pi*freq
options = ['EC','EO', 'REM', 'S1', 'S2', 'SWS', 'Spindles']
for option in options:
    params = parameter_set(option)
    model = RobinsonLinModel(params)
    P = model.forward(external=0, w1=w1)
    plt.plot(freq,P/sum(P))
plt.yscale("log")
plt.xscale("log")
plt.ylabel('Normalized Power (A.U.)')
plt.xlabel('Frequency (Hz)')
plt.legend(options)
plt.title('Power spectra for different parameter sets')
