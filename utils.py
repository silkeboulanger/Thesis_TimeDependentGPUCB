#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 6 20:51:11 2024

@author: silkedesmedt
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt

def GP(observed_rewards, grid_size, l_fit, revealed, epsilon = 0.0001):
    '''
    This function starts from the observations,
    for which the coordinates are saved in xlist
    and the observed rewards in rewardlist
    since the GP assumes that not observed states have as default a value of 0,
    we rescale the reward to have a mean 0 and vary between max bounds of -0.5, 0.5
   
    Parameters
    ----------
    observed_bandit : the observations
    W : width of the bandit.
    L : lenght of the bandit.
    l_fit: the generalization strength with which the participant smooths out the observed rewards
    hidden : list of which cells are hidden. True if hidden, False if the reward is known.
    epsilon: the assumed noise,
            measured as the variance of the reward around the mean
            std is 1 up 100,
            var is 0.01**2 = 0.0001
   
    Returns
    -------
    Returns the mean function m(x) and the uncertainty function s(x) per tile.
    '''
    revealed2 = revealed.reshape(grid_size, grid_size)
    xlist = [] #list with the coordinates of the data points (list of doubles)
    rewardlist = []
    for i in range(0, len(revealed2)):
        for j in range(0, len(revealed2[0])):
            if revealed2[i][j] == True: #then we have an observation for this cell
                for observation in observed_rewards[i*grid_size + j]:
                    xlist.append([i, j])
                    rewardlist.append(observation)
                   
    cells =[[i,j] for i in range(0, grid_size) for j in range(0, grid_size)]    
   
    kernel = RBF(l_fit, "fixed") + WhiteKernel(epsilon, "fixed")
       
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(xlist, [(reward-40)/90 for reward in rewardlist])
    mlist, sigmalist = gp.predict(cells, return_std=True)
             
    return mlist, sigmalist

def UCB(mlist, sigmalist, beta):
    # Calculate upper confidence bounds
    UCB_values = mlist + beta * sigmalist

    return UCB_values

def softmax(UCB_values, tau):
    # Subtract max for numerical stability
    scaled_UCB = (UCB_values - np.max(UCB_values)) / tau

    exp_values = np.exp(scaled_UCB)
    sum_exp = np.sum(exp_values)

    # Avoid division by zero
    if sum_exp == 0 or np.isnan(sum_exp):
        raise ValueError("Sum of exponential values is zero or NaN. Check UCB_values and tau.")

    probabilities = exp_values / sum_exp
    return probabilities


def plot_grid(revealed, grid_size, rewards, action):
    plt.figure(figsize=(8, 6))
    plt.imshow(np.where(revealed.reshape((grid_size, grid_size)), np.array(rewards).reshape((grid_size, grid_size)), 0), cmap='Reds', vmin=0, vmax=100) 
    plt.colorbar().set_label('Reward')
    plt.title('Screenshot')
    
    # Annotate revealed tiles with rewards
    for i in range(grid_size):
        for j in range(grid_size):
            if revealed[i * grid_size + j]:  # Check if the tile is revealed
                reward = rewards[i * grid_size + j]
                plt.text(j, i, f'{reward:.2f}', ha='center', va='center', color='white')
    
    if action:
        plt.scatter(action % grid_size, action // grid_size, color='black', marker='o') #indicate the new tile
    
    plt.show()

def localizer(UCB, prior_choice, W, L):
    # Create a grid of coordinates
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(L), indexing='ij')
    
    # Compute inverse Manhattan distance (IMD)
    distance = np.abs(x_coords - prior_choice[0]) + np.abs(y_coords - prior_choice[1])
    with np.errstate(divide='ignore'):
        IMD = 1 / distance
    IMD[prior_choice[0], prior_choice[1]] = 1  # Handle division by zero at prior_choice

    # Reshape UCB and apply IMD weighting
    UCB_reshaped = np.array(UCB).reshape(W, L)
    UCBloc = IMD * UCB_reshaped

    return UCBloc.ravel()  # Return as flat array to match original output shape
