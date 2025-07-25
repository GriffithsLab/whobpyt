.. _implementation:

Algorithms and other implementation details
===========================================

🔁 Model Inversion Overview

The model is reformulated as a state-space model:

- **Hidden States**: Represent neural population dynamics
- **Observations**: EEG signals modeled as projections of pyramidal or excitatory activity

State Equation:

.. math::
   \dot{x}_t = f(x_t, \theta) + w_t

Observation Equation:

.. math::
   y_t = g(x_t, \theta) + v_t

---

 🔍 Inference Techniques

1. Variational Bayesian Inference (VBI)

- Estimates posterior distributions of parameters and states
- Minimizes **variational free energy** \( \mathcal{F} \)
- Captures uncertainty and enables Bayesian model comparison

2. Gradient-Based Optimization

- Implements automatic differentiation (PyTorch)
- Minimizes reconstruction loss:

.. math::
   \mathcal{L} = \text{MSE}(y_{\text{true}}, y_{\text{sim}}) + \text{KL}(q(\theta) || p(\theta))

3. Kalman Filters (Optional)

- Includes **Extended Kalman Filter (EKF)** and **Unscented Kalman Filter (UKF)**
- Real-time parameter and state estimation
- Ideal for closed-loop applications

---

🧪 Tools Supported

- PyTorch backend for differentiable simulation
- Modular inversion API
- EEG compatibility through MNE and NumPy


