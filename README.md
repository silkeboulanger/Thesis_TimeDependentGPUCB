# Thesis\_TimeDependentGPUCB

This repository contains the code and resources supporting the master's thesis titled **"As flexible as humans? Towards more temporally adaptive models of value-based decision-making"**.

## Overview

The goal of this project is to implement and evaluate various time-dependent extensions of the Gaussian Process Upper Confidence Bound (GP-UCB) model, including:

* **Linear schedules** for the exploration parameter β (broad and narrow variants)
* **Exponential schedules** (broad and narrow variants)
* **Reward-dependent schedules** (broad and narrow variants)
* **Baseline invariant β** for comparison
* **Control Variance Model** control where model uncertainty is heightened 

We compare these time-varying adaptations on simulated data, and apply them to empirical datasets involving spatial multi-armed bandit tasks.
