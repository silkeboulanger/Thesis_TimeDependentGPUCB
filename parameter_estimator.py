"""
Estimate parameters for all participants in the dataset, using parallel processing.
"""
import pandas as pd
from utils_exponential import estimate
from joblib import Parallel, delayed

test = True # Set to True to test the function on one participant

# Specific variables
nr_trials = 25
nr_blocks = 10
W = L = 11
data_name = "/Data/datacontrols_preprocessed" #Data van alle controleparticipanten (geen ASS)

# Read in data
data = pd.read_csv(data_name + '.csv', delimiter=',')
participants = data.subjectID.unique()

# Prepare the function for estimation
def estimate_one(participant):
    data_p = data[data.subjectID == participant]
    est = estimate(W, L, nr_trials, nr_blocks, data_p)
    
    return [
        participant, 
        est[0][0],  # l_fit
        est[0][1],  # beta
        est[0][2],  # tau
        est[1],     # NLL
        2 * 4 + 2 * est[1]  # AIC (3 estimated parameters)
    ]

if test:
    one_participant = participants[8] # For the first pp, change index for other pps
    participant = one_participant
    results = estimate_one(participant)
    print(results)
else: 
    # Parallelize with joblib
    results = Parallel(n_jobs=-1)(delayed(estimate_one)(participant) for participant in participants)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=["Participant", "l_fit", "beta", "tau", "NLL", "AIC"])

    # Save results to CSV at the end
    results_df.to_csv("estimates.csv", index=False)
