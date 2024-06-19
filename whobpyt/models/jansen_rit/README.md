# Jansen-Rit Neural Mass Model

## Description
The Jansen-Rit model is  a neural mass model representing macroscopic electrophysiological activity within a cortical column. More specifically, it can be used to reproduce a broad range of EEG oscillation frequencies, as well as evoked response waveform. The circuit is composed of three interconnected neural populations: pyramidal projection neurons, excitatory interneurons and inhibitory interneurons, with two feedback loops (a (fast) excitatory feedback loop and a slow inhibitory feedback loop). The output, representing the net PSP on the pyramidal cell dendrites, is defined as the difference between the EPSP from the excitatory population and the IPSP from the inhibitory population. This quantity corresponds to the membrane potential of pyramidal neurons which can also be understood as the output of the columnar microcircuit that is transmitted to other adjacent and distal brain areas.

Each neural population is decribed with two operators: a rate-to-potential operator describing the dynamics between synapses and dendritic trees, and a potential-to-rate operator representing the output firing rate produced at the soma performing the inverse operation. The populations are described in these two steps to capture then the dynamics of the circuit. More details are provided in the equations section below.

## Modifications
- Added noise input term to the populations
- All populations receive input from other populations
- The external input is added to the pyramidal population

## Equations & Biological Variables From:
The first mathematical operator converts the mean pulse densities of incoming action potentials into an excitatory or inhibitory postsynaptic membrane potential (EPSP or IPSP, respectively). This PSP operator linearly transforms an incoming impulses, and is given by

$$h(t)=Aate^{-at} \qquad \text{for t} > 0 $$
for excitatory cases, and
$$h(t)=Bbte^{-bt} \qquad \text{for t} > 0 $$
for inhibitory cases. A and B are the maximum EPSP and IPSP amplitudes, respectively; while 'a' and 'b' represent the combined effects of the membrane's reciprocal time constant and other distributed delays in the dendritic network. The second mathematical operator, takes the mean membrane potential of the neuronal population (i.e., the output from the neuronal population) and converts it into an average pulse density of action potentials. This is given by a nonlinear function, in the form of a sigmoid,
$$Sigm(v) = \frac{2e_{0}}{1 + e^{r(v_{0}-v)}}$$
The maximum firing rate on the neuronal population is given by $e_{0}$. The PSP at half the maximum firing rate is given by $v_{0}$, and $r$ is the slope of the sigmoid transform (Jansen & Rit, 1995).
Four connectivity constants characterize the interaction between the three neuron subtypes in the JR model (i.e., pyramidal cells, excitatory, and inhibitory interneurons). These constants are $C_{1}$, $C_{2}$, $C_{3}$, and $C_{4}, and they account for the total number of synapses between the interneurons and the axons and dendrites of the cortical column neurons. Including these connectivity constants, the following six differential equations describe the JR model:
```math
\dot{y}_{0}(t) = y_{3}(t)
```
```math
\dot{y}_{3}(t) = AaS[y_{1}(t)-y_{2}(t)] - 2ay_{3}(t) - a^{2}y_{0}(t)$$
```
```math  
    \dot{y}_{1}(t) = y_{4}(t)
```
```math
    \dot{y}_{4}(t) = Aa(p(t) + C_{2}S[C_{1}y_{0}(t)]) - 2ay_{4}(t) - a^{2}y_{1}(t)
```
```math
    \dot{y}_{2}(t) = y_{5}(t)
```
```math
    \dot{y}_{5}(t) = BbC_{4}S[C_{3}y_{0}] - 2by_{5}(t) - b^{2}y_{2}(t)
```
where $y_{0}$, $y_{1}$, and $y_{2}$ are the outputs of the PSP block from the pyramidal cells, the excitatory interneurons, and the inhibitory interneurons, and $y_{3}$, $y_{4}$, and $y_{5}$ are their first-order derivatives, respectively.

