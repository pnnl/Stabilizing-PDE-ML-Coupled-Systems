# Stabilizing PDE-ML Coupled Systems

This repository contains the codes and data-files to accompany the publication: 

**Saad Qadeer, Panos Stinis, and Hui Wan. "Stabilizing PDE-ML Coupled Systems." CMAME (2026).**

In summary, PDE-ML coupled systems are notoriously inaccurate when numerically integrated in time. This issue acts as a bottleneck to the use of ML architectures in the online sense in complex systems (e.g., engineering systems, earth systems, etc). We investigated this phenomenon for the viscous Burgers' equation by replacing the diffusion term with a feedforward neural network. The combined system was trained in-the-loop by driving its predictions to be close to the original system. As expected, the coupled system was unstable when integrated in time. We found that the ML component suffered from spectral bias and developed a filtration strategy to stabilize the system. This however leads to an under-resolved simulation, which by its nature is not very accurate. We next developed a strategy based on the Mori-Zwanzig formalism to improve the accuracy of the combined system. This entails appending memory-based terms to the stabilized systems, which take the form of convolutional integrals with memory kernel functions. Our scripts enable the calculation of these these kernels accurately and (in principle) to an arbitrarily high order. The additional terms were then shown to considerably reduce the errors in the under-resolved stabilized system.

The descriptions of the important scripts are as follows:

+ surr_trainer.jl: script for training the surrogate in place of the diffusion term in the viscous Burgers' equation
+ surrogate_example: weights and biases for a trained surrogate
+ ML_stab.m: integrates the coupled system forward in time and demonstrates the instability. Applying low-pass filters allows us to stabilize the calculations
+ kernel_calc.m: calculates the memory kernels for the viscous Burgers' equation, with the order of the correction terms specified by D
+ KvsR_linear and KvsR_cubic: example memory kernels calculated up to T = 20, for D = 1 and D = 3 respectively
+ ML_stab_ML_sim.m: integrates the coupled system with memory corrections added for improved accuracy

The remaining scripts contain implementations of various auxiliary tasks such as Runge-Kutta schemes and Gauss-Hermite quadrature rules
