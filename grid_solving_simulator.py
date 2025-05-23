import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from bandits11_l2 import bandits
from utils import GP, UCB, softmax, localizer

## SET WD ##
directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(directory)

folder = './Data/SCMAB_data/'

### SETTINGS ###
localized = True

time_variance = 'reward_variant'  # 'time-invariant', 'time-linear', 'time-exponential', 'reward_variant', 'control_variance'
broad = False # Broad or narrow variant

print(f'Sim_{time_variance}_{broad}')

# Define bandit grid size
grid_size = 11
num_tiles = grid_size ** 2

# Experiment size in time:
n_trials = 25
n_blocks = 10
n_participants = 500

# Noise standard deviation for clicked rewards
noise_sd = 1

## INITIALISATION ##
clicked_tile = []
clicked_reward = []
initial_opened = []
initial_reward = []
trial_nr = []
block_nr = []
selected_bandit = []
subject_id_list = []
subject_beta = []
subject_lambda = []
subject_tau = []

for p in range(n_participants):
    # Random parameter initialization
    l_fit = np.random.uniform(0.3, 1.26)
    beta0 = np.random.uniform(0.17, 0.78)
    tau = np.random.uniform(0.03, 0.14)

    print(f"Participant {p + 1}/{n_participants}")

    for b in range(n_blocks):
        # Select reward map
        bandit_index = np.random.choice(len(bandits))
        rewards = np.array(bandits[bandit_index], dtype=float).reshape((grid_size, grid_size))

        # Revealed tiles
        revealed = np.full(num_tiles, False)
        random_index = np.random.choice(num_tiles, 1, replace=False)
        revealed[random_index] = True

        random_coords = np.unravel_index(random_index, (grid_size, grid_size))
        random_reward = rewards[random_coords]

        # Initialize reward history
        observed_rewards = [[] for _ in range(num_tiles)]
        for idx, r in zip(random_index, random_reward):
            observed_rewards[idx].append(r)

        prior_choice = [random_index[0] // grid_size, random_index[0] % grid_size]
        reward_adjustment = False
        reward = random_reward[0]  # Use first initial reward for logic

        for t in range(n_trials):
            # GP
            mlist, sigmalist = GP(observed_rewards, grid_size, l_fit, revealed, epsilon=0.0001)

            mlist = np.array(mlist).reshape((grid_size, grid_size))
            sigmalist = np.array(sigmalist).reshape((grid_size, grid_size))

            # Beta schedule
            if time_variance == 'time-invariant':
                beta = beta0
            elif time_variance == 'control_variance':
                beta = beta0
            elif time_variance == 'time-linear':
                if broad:
                    beta = beta0 + 0.5 * beta0 * (1 - 2 * (t + 1) / n_trials)
                else:
                    beta = beta0 + 0.33 * beta0 * (1 - 2 * (t + 1) / n_trials)
            elif time_variance == 'time-exponential':
                if broad:
                    beta = (1.7 * beta0) * np.exp(-np.log(3) * t / (n_trials - 1))
                else:
                    beta = (1.33 * beta0) * np.exp(-np.log(2) * t / (n_trials - 1))
            elif time_variance == 'reward_variant':
                if reward < 63 and not reward_adjustment:
                    if broad:
                        beta = beta0 * (6 / 4)
                    else:
                        beta = beta0 * (4 / 3)
                else:
                    if broad:
                        beta = beta0 * (2 / 4)
                    else:
                        beta = beta0 * (2 / 3)
                    reward_adjustment = True

            # UCB and optional localization
            UCB_values = UCB(mlist, sigmalist, beta)
            UCB_values /= np.max(UCB_values)

            if localized:
                localized_ucb = localizer(UCB_values.ravel(), prior_choice, grid_size, grid_size)
                UCB_values = np.array(localized_ucb).reshape((grid_size, grid_size))

            # Softmax decision
            softmax_values = softmax(UCB_values, tau)
            action = np.random.choice(np.arange(num_tiles), p=softmax_values.ravel())

            # Update prior choice
            prior_choice = [action // grid_size, action % grid_size]

            # Reward
            reward = rewards[prior_choice[0], prior_choice[1]] + np.random.normal(0, noise_sd)
            observed_rewards[action].append(reward)
            revealed[action] = True

            # Append results
            clicked_tile.append(action)
            clicked_reward.append(reward)
            trial_nr.append(t)
            block_nr.append(b)
            subject_id_list.append(p)
            subject_beta.append(beta)
            subject_lambda.append(l_fit)
            subject_tau.append(tau)
            selected_bandit.append(bandit_index)
            initial_opened.append(random_index.tolist())
            initial_reward.append(np.round(random_reward, 2).tolist())

# Compile DataFrame
df = pd.DataFrame({
    'subjectID': subject_id_list,
    'subject_beta': subject_beta,
    'subject_lambda': subject_lambda,
    'subject_tau': subject_tau,
    'selected_bandit': selected_bandit,
    'block_nr': block_nr,
    'trial_nr': trial_nr,
    'initial_opened': initial_opened,
    'initial_reward': initial_reward,
    'selected_choice': clicked_tile,
    'reward': clicked_reward
})

print("Done!")
# Save CSV
df.to_csv(f'{folder}Sim_{time_variance}_{broad}.csv', index=False)
