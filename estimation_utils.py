# -*- coding: utf-8 -*-
"""
script to estimate the model parameters of one (behavioural/computational) participant
"""
import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import scipy
from scipy.optimize import minimize
"""
Prior with which we want to estimate the model parameters:
"""
time_variance = 'time-invariant' # Options: time-invariant, time-linear, control_variance, time-exponential and reward_variant
broad = True # Options: True or False 

#prior values of model parameters, used to start the search and to bias the optimization
x0 = [10, 0.5, 0.01]
#std of the priors on the model parameters, used to bias the optimization
xstd = [8, 0.2, 0.2] 
#covariance matrix of the priors on the model parameters
xcov = np.diag(xstd)

bounds = np.exp(np.array([(-5, 6), (-5, 3), (-5, 3)]))
localized = True #to change if we want to use the localized model or not

def iqr_dev(arr): # Interquartile range deviation
    return (np.percentile(arr, 75) - np.percentile(arr, 25)) / 2

def estimate(W, L, nr_trials, nr_blocks, data, time_variance, broad):
    """
    Laat telkens 1 blok uit de data weg en schat de parameters van de andere blokken, en test deze
    op het weggelaten blok.

    """
    #Initial opened calculation
    datalast = data[data.trial_nr == (nr_trials - 1)]  #Only last trial
    initial_opened = [value for value in datalast.initial_opened] # initial opened cell
    
    l_fit_est_list, beta_est_list, tau_est_list, NLL_list = [], [], [], []
    
    for i in range(nr_blocks):
        training_data = data.query(f'block_nr != {i}')
        training_args = (W, L, nr_trials,
                initial_opened[:i] + initial_opened[i + 1:], # Initial opened van alles behalve de weggelaten blok
                list(training_data.selected_choice),
                list(training_data.reward),
                list(training_data.average_reward)
            )

        # Optimaliseert parameters voor deze fold in log schaal, en maakt bounds zodat de 'search space' niet te groot is
        # Dit is nodig omdat de optimizer niet goed werkt met een te grote search space
        result = minimize(fun=wrapper, x0 = np.log(x0), args=training_args, method='SLSQP', bounds=np.log(np.array(bounds)))
        # Exponentieert de parameters terug naar de originele schaal
        (l_fit_est, beta_est, tau_est) = np.exp(result.x)
        
        l_fit_est_list.append(l_fit_est)
        beta_est_list.append(beta_est)
        tau_est_list.append(tau_est)
        
        # Valideer deze schattingen met de weggelaten blok
        test_data = data.query(f'block_nr == {i}')
        # Bereken de NLL voor de weggelaten blok met de geschatte parameters van de testing data
        cross_val = NLL(W, L, nr_trials, l_fit_est, beta_est, tau_est, 
                        [initial_opened[i]], 
                        [value for value in test_data.selected_choice],
                        [value for value in test_data.reward], 
                        [value for value in test_data.average_reward],
                        time_variance, 
                        broad)
        
        NLL_list.append(cross_val)
        
    print("Median parameters with interquartile deviations:")
    print(f"lambda:      {np.median(l_fit_est_list):.3f} ± {iqr_dev(l_fit_est_list):.3f}")
    print(f"beta:        {np.median(beta_est_list):.3f} ± {iqr_dev(beta_est_list):.3f}")
    print(f"tau:         {np.median(tau_est_list):.3f} ± {iqr_dev(tau_est_list):.3f}")
    print(f"Total NLL:   {np.sum(NLL_list):.3f}")
    resx = (np.median(l_fit_est_list), np.median(beta_est_list), np.median(tau_est_list))
    resfun = np.sum(NLL_list)
    
    return (resx, resfun)


def wrapper(par, *args):
    """
    this function is just a wrapper function that brings the NLL function into the correct format
    to be called by the optimization function
    
    used for the 1 l, 1 beta and 1 tau per data set
    
    Parameters
    ----------
    par : array of floats
        the array of the model parameters that need to be optimized
    *args : tuple of ...
        all the other arguments needed for NLL

    Returns
    -------
    returns the function NLL
    """
    l_fit = np.exp(par[0])
    if (l_fit == 0):
        l_fit = 10e-8
    
    beta = np.exp(par[1])
    if (beta == 0):
        beta = 10e-8
    
    tau = np.exp(par[2])
    if (tau == 0):
        tau = 10e-8    

    (W, L, nr_trials, initial_opened, selected_choice, reward, average_reward) = args
    
    '''
    The probability that the data is generated 
    by a model with the given model parameters
    is 
    prob = L * bias
    so 
    NLprob = NLL + NLbias = NLL - Lbias
    '''
    term1 = NLL(W, L, nr_trials, l_fit, beta, tau, initial_opened, selected_choice, reward, average_reward) 
    term2 = NLbias((l_fit, beta, tau), x0, xcov) # Straf als de parameters te ver van de prior afwijken
    NLprob = term1 + term2
    return NLprob
    

'''
functions calculating the probability of each choice given a parameter set
'''
def NLL(W, L, nr_trials, l_fit, beta, tau, initial_opened, selected_choice, reward, average_reward):
    """
    Simulatie van de NLL van de data, gegeven de parametersx
    """
    LL = 0
    
    for round_nr in range(0, len(initial_opened)):
        opened_cells = [initial_opened[round_nr]]
        #we need to change the format to a list of coordinates
        opened_cells2D = [[math.floor(value/W), value%W] for value in opened_cells]
        
        first_observation = 2*average_reward[round_nr*nr_trials] - reward[round_nr*nr_trials] # the first observation is the average reward of the first trial, omdat de initial_reward niet in de data zit
        observations = [first_observation]

        reward_adjustment = False
        
        for trial_nr in range(0, nr_trials):
            choice = selected_choice[round_nr*nr_trials + trial_nr]
            prior_choice_number = opened_cells[-1] 
            LL += log_probability(W, L, choice, opened_cells2D, observations, l_fit, beta, tau, prior_choice_number, nr_trials, trial_nr, reward, reward_adjustment, time_variance, broad)
            
            #update the observations before moving on to the next 
            opened_cells.append(choice)
            opened_cells2D = [[math.floor(value/W), value%W] for value in opened_cells]
            observations.append(reward[round_nr*nr_trials + trial_nr])
       
    return -LL


'''
functions needed to calculate the NLL
'''
#more stable if we use the log_softmax function of scipy instead of calculing P ourselves and then logging it
def log_probability(W, L, choice, opened_cells2D, observations, l_fit, beta, tau, tile_number, nr_trials, trial_nr, reward, reward_adjustment):
    # MODEL STEP 1: GAUSSIAN PROCESS
    m, s = GP(observations, W, L, l_fit, opened_cells2D)
    
    # MODEL STEP 2: UCB
    # Adjust beta depending on the time variance setting
    if time_variance == 'time-invariant':
        current_beta = beta
        
    if time_variance == 'control_variance':
        s = s**2
        current_beta = beta 
        
    # The beta is constant over time, so we don't need to change it
    elif time_variance == 'time-linear':
        if broad:
            current_beta = beta + 0.5 * beta * (1 - 2 * (trial_nr + 1) / nr_trials)
        else:
            current_beta = beta + 0.33 * beta * (1 - 2 * (trial_nr + 1) / nr_trials)
            
    elif time_variance == 'time-exponential':
        if broad:
            current_beta = (1.7 * beta) * np.exp(-np.log(3) * trial_nr / (nr_trials - 1))
        else:
            current_beta = (1.33 * beta) * np.exp(-np.log(2) * trial_nr / (nr_trials - 1))

    elif time_variance == 'reward_variant': 
        if not reward_adjustment:
            if broad:
                current_beta = beta * (6/4)
            else:
                current_beta = beta * (4/3)
        if reward[trial_nr] >= 63 or reward_adjustment:
            if broad:
                current_beta = beta * (2/4)
            else:
                current_beta = beta * (2/3)
            reward_adjustment = True

    UCB = [m[i] + current_beta * s[i] for i in range(0, W*L)]
    
    if localized:
        #UCB gets weighted
        prior_choice = [(tile_number-(tile_number%W-1))//W, tile_number%W]
        UCB = localizer(UCB, prior_choice, W, L)
    
    # MODEL STEP 3: SOFTMAX
    UCBtau = [value/tau for value in UCB]
    log_P = scipy.special.log_softmax(UCBtau)
    
    return log_P[int(choice)]
    

def GP(observations, W, L, l_fit, opened_cells2D, epsilon = 0.001):   
    '''
    This function starts from the observations
    since the GP assumes that not observed states have as default a value of 0,
    we rescale the reward to have a mean 0 and vary between max bounds of -0.5, 0.5
    
    Parameters
    ----------
    observations : the observations
    W : width of the bandit.
    L : lenght of the bandit.
    l_fit: the generalization strength with which the participant smooths out the observed rewards
    opened_cells2D : list of which cells are opened.
    noise: True if the participant assumes noise in their observations
    epsilon: the assumed noise, 
            measured as the variance of the reward around the mean
            std is 1 up 100, 
            var is 0.01**2 = 0.0001
             
    Returns
    -------
    Returns the mean function m(x) and the uncertainty function s(x) per tile.
    '''
    
    cells =[[i,j] for i in range(0, W) for j in range(0, L)]    
    
    kernel = RBF(l_fit, "fixed") + WhiteKernel(epsilon, "fixed")
             
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(opened_cells2D, [(reward-40)/90 for reward in observations]) 
    
    mlist, sigmalist = gp.predict(cells, return_std=True)
    
    return mlist, sigmalist

def NLbias(par, x0, xcov):
    """
    Straf als de parameters te ver van de prior afwijken
    """
    NLbias = 0
    for i in range(len(par)):
        NLbias += (par[i]-x0[i])**2/xcov[i][i]
   
    return NLbias/2

def localizer(UCB, prior_choice, W, L):
    """
    Applies inverse Manhattan distance weighting to the UCB values
    based on the location of the prior choice.
    """

    # Initialize the updated UCB values
    UCBloc = []

    for i in range(W):
        for j in range(L):
            # Compute Manhattan distance from the prior choice
            manhattan_distance = abs(i - prior_choice[0]) + abs(j - prior_choice[1])
            
            # Avoid division by zero at the prior choice
            if manhattan_distance == 0:
                weight = 1.0
            else:
                weight = 1.0 / manhattan_distance

            # Get the index of the current tile in the flat UCB list
            index = i * L + j

            # Apply the weight to the UCB value
            weighted_value = UCB[index] * weight
            UCBloc.append(weighted_value)

    return UCBloc
