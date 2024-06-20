import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from whobpyt.optimization import CostsPSD
import torch


def plot_fc(recording, skip_dur=500):
    """
    This function takes a Recording object and plots the functional connectivity based on its timeseries data.

    Parameters:
    recording: Recording object containing the activity data.
    skip_dur: Initial transient duration to skip (in seconds).
    """

    step_size = recording.step_size
    skip_trans = int(skip_dur/step_size)
    num_regions = recording.data.shape[0]
    
    # Compute functional connectivity
    sim_FC = np.corrcoef(recording.npTS()[:,skip_trans:])

    plt.figure(figsize = (8, 8))
    plt.title("Simulated BOLD FC: After Training")

    # Create a mask to ignore self-connections
    mask = np.eye(num_regions)

    # Heatmap of functional connectivity
    sns.heatmap(sim_FC, mask = mask, center=0, cmap='RdBu_r', vmin=-1.0, vmax = 1.0)
    plt.show()


def plot_timeseries(recording, pop_label):
    """
    Takes a Recording object and plots the timeseries 
    activity of a specific population.

    Parameters:
    recording: Recording object containing the activity data.
    pop_label: String representing the population label.
    """
    num_regions = recording.data.shape[0]
    step_size = recording.step_size
    
    plt.figure(figsize = (16, 8))
    plt.title(f"Activity of {pop_label}")

    for n in range(num_regions):
        plt.plot(recording.npTS()[n, :], label = f"{pop_label} Node = {n}")

    plt.xlabel(f'Time Steps (multiply by step_size to get msec), step_size = {step_size}')
    plt.ylabel('Activity')
    plt.legend()
    plt.show()






def plot_psd(recording, minFreq=2, maxFreq=40):
    """
    This function takes a Recording object and plots the power spectral density (PSD) based on its timeseries data.

    Parameters:
    recording: Recording object containing the activity data.
    minFreq: Minimum frequency to plot (in Hz).
    maxFreq: Maximum frequency to plot (in Hz).
    """
    step_size = recording.step_size
    sampleFreqHz = 1000*(1/step_size)
    sdAxis, sdValues = CostsPSD.calcPSD((recording.npTS().T), sampleFreqHz, minFreq, maxFreq)
    sdAxis_dS, sdValues_dS = CostsPSD.downSmoothPSD(sdAxis, sdValues, 32)
    sdAxis_dS, sdValues_dS_scaled = CostsPSD.scalePSD(sdAxis_dS, sdValues_dS)

    plt.figure()
    plt.plot(sdAxis_dS, sdValues_dS_scaled.detach())
    plt.xlabel('Hz')
    plt.ylabel('PSD')
    plt.title("Simulated EEG PSD: After Training")
    plt.show()
