.. _design_philosophy:

Design philosophy
=================

WHOBPYT is a general-purpose brain modeling framework designed for modular neural mass and field models, enabling efficient simulation and inversion of EEG, PSD, and BOLD data. It is optimized for scientific transparency, biophysical interpretability, and extensibility across multiple neural model classes.

🧠 Supported Model Families
WHOBPYT integrates multiple canonical models of brain dynamics:

Module	Description
jansen_rit	Simulates cortical column dynamics (EEG) via Jansen-Rit equations
CBNet	Implements the Robinson neural field model for PSD/ECoG with saturation
RWW	Realizes the Wong-Wang–Deco mean-field model for fMRI BOLD dynamics

Each model inherits a shared interface for:

forward(): forward dynamic simulation

Parames: define the model parameter


⚙️ Core Design Principles
1. Unified Inference Pipeline
All models use a common pipeline structure:

State-space simulation

Observation model (EEG/PSD/BOLD)

Likelihood construction (e.g. MSE, KL, free energy)

Inversion (gradient-based, variational) modelfitting Clase

This enables the same tools to fit EEG alpha rhythms, PSD slopes, or resting-state BOLD with minimal code changes.

2. Separation of Biophysics and Numerics
Each model encapsulates its biophysics in a standalone class:

Dynamics defined by biologically interpretable parameters

Observation models project hidden states to data modalities

Priors and likelihoods modularly combined for flexible inference

Numerical tools (integration, differentiation, convolution) are abstracted in reusable utility modules.

3. Multi-Scale Brain Modeling
WHOBPYT supports modeling across multiple spatial and temporal scales:

Fast oscillatory EEG (JR, Robinson)

Large-scale spectral dynamics (Robinson)

Slow fMRI BOLD fluctuations (Wong-Wang–Deco)

This makes it suitable for comparing models across modalities, or building hybrid pipelines (e.g., EEG-constrained fMRI modeling).

4. Neurobiological Interpretability
All parameters (e.g., excitatory/inhibitory gain, conductances, saturation thresholds) are physiologically grounded. Inversion targets meaningful biomarkers such as:

E:I balance

Alpha peak frequency

Time constants

Dynamic stability (Lyapunov exponents)

5. Flexible, Extensible Infrastructure
Each model module (jansen_rit, CBNet, RWW) is independent but interoperable, enabling:

Easy integration of new model types

Extension to new data modalities

Head-to-head model comparison for model selection

🔄 Model-Specific Capabilities
Model	Data Type	Inference	Use Case
Jansen-Rit (jansen_rit)	EEG (time)	Variational, UKF, gradient	Cortical dynamics, TMS planning
Robinson Field (CBNet)	PSD (freq)	Spectral matching, variational	ECoG/EEG biomarker inference
Wong-Wang–Deco (RWW)	BOLD (fMRI)	Gradient-based, DCM-style	Resting-state connectivity, psychiatric modeling

🚀 Summary
WHOBPYT is built to be:

✅ Scientifically rigorous: grounded in neurophysiology

🧩 Modular and extensible: plug in your own model or solver

🔬 Inference-ready: invert EEG, PSD, or BOLD with interpretable outputs

🧪 Application-focused: designed for mental health, brain stimulation, and biomarker discovery
