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

## Repository Structure

```
MODEL_MASTERPROEF/
├── Data/
│   ├── estimates/            # Model estimation output
│   └── SCMAB_data/           # Simulated multi-armed bandit task data
├── estimation_utils.py       # Utility functions for parameter estimation
├── grid_solving_simulator.py # Spatial bandit task simulator
├── model_behavior_visualisations.ipynb  # Jupyter notebook for simulation behavioral plots
├── model_comparison.ipynb    # Notebook for model fit comparisons
├── parameter_estimator.py    # Parameter estimation class and wrappers
├── preprocessing.py          # Data preprocessing routines
├── stats.R                   # R scripts for statistical analyses on real participant data
└── utils.py                  # General utility functions
```
