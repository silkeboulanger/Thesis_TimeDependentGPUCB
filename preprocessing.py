import os
import pandas as pd 
import numpy as np
os.chdir(os.path.dirname(os.path.realpath(__file__)))

grid_size = 11

# Take all csv files in the folder
folder = './Data/SCMAB_data/'

files = [f for f in os.listdir(folder) if f.endswith('.csv')]

for file in files:
    data = pd.read_csv(folder + file)
    print(f'Processing {file}')
    
    # Ensure numeric
    data['selected_choice'] = pd.to_numeric(data['selected_choice'], errors='coerce')
    data['initial_opened'] = pd.to_numeric(data['initial_opened'], errors='coerce')

    # Convert tile variable to x,y coordinates (vectorized)
    data['Clicked y'] = data['selected_choice'] // grid_size
    data['Clicked x'] = data['selected_choice'] % grid_size
    data['Initial y'] = data['initial_opened'] // grid_size
    data['Initial x'] = data['initial_opened'] % grid_size

    
    distance_to_top10_Manhattan = []
    high_value_clicks = []
    novel_click = []
    consecutive_distance_Manhattan = []

    # Iterate over each subject
    for subject in np.unique(data['subjectID']):
        # Filter the data for the current subject
        data_subject = data[data['subjectID'] == subject]

        for block in np.unique(data_subject['block_nr']):
            # Filter the data for the current block
            data_block = data_subject[data_subject['block_nr'] == block]

            for trial in range(len(data_block)):
                current_trial = data_block.iloc[trial]
                x, y = current_trial['Clicked x'], current_trial['Clicked y']
                
                if trial == 0:
                    distance_to_top10_Manhattan.append(abs(x - current_trial['Initial x']) + abs(y - current_trial['Initial y']))
                    consecutive_distance_Manhattan.append(np.nan)
                    threshold = 0  # just set to 0 to avoid reward filtering
                else:
                    until_current_trial = data_block.iloc[:trial]
                    last_trial = data_block.iloc[trial - 1]
                    rewards_list = until_current_trial['reward'].tolist()
                    max_reward = max(rewards_list)
                    threshold = 0.9 * max_reward

                    consecutive_distance_Manhattan.append(abs(x - last_trial['Clicked x']) + abs(y - last_trial['Clicked y']))
                    top10x = until_current_trial.loc[until_current_trial['reward'] >= threshold, 'Clicked x'].tolist()
                    top10y = until_current_trial.loc[until_current_trial['reward'] >= threshold, 'Clicked y'].tolist()
                    
                    if len(top10x) > 0:
                        distances_list = [abs(x - top10x[i]) + abs(y - top10y[i]) for i in range(len(top10x))]
                        distance_to_top10_Manhattan.append(min(distances_list))
                    else:
                        distance_to_top10_Manhattan.append(np.nan)

                # Novel click
                if trial == 0 or current_trial['selected_choice'] not in data_block.iloc[:trial]['selected_choice'].tolist():
                    novel_click.append(1)
                    high_value_clicks.append(0)
                else:
                    novel_click.append(0)
                    high_value_clicks.append(1 if current_trial['reward'] >= threshold else 0)

    # Add the computed values back to the DataFrame
    data['Distance to Top10 Manhattan'] = distance_to_top10_Manhattan
    data['Consecutive Distance Manhattan'] = consecutive_distance_Manhattan
    data['High Value Click'] = high_value_clicks
    data['Novel Click'] = novel_click

    # Take log of the distance to top10 and consecutive distance
    data['log_distancetotop10'] = np.log(data['Distance to Top10 Manhattan'].replace(0, 1e-6))
    data['log_consecutivedistance'] = np.log(data['Consecutive Distance Manhattan'].replace(0, 1e-6))

    # Previous reward
    data['Previous reward'] = data['reward'].shift(1)
    # Round the previous reward
    data['Previous reward'] = data['Previous reward'].round(0)

    # Check the updated DataFrame
    data.to_csv(f'{folder}/preprocessed{file}', index=False)