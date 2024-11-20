import os
import pandas as pd
import numpy as np

# Directory containing the CSV files
directory = '/h/321/ashmita/Gradients-Look-Alike-Sensitivity-is-Often-Overestimated-in-DP-SGD/Gradients-Look-Alike-Sensitivity-is-Often-Overestimated-in-DP-SGD/res'

# List to store DataFrames
dfs = []

# Read each CSV file and store the DataFrame in the list
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        dfs.append(df[['distance (sum)', 'sigma', 'point']])

# Concatenate all DataFrames along the columns
combined_df = pd.concat(dfs, axis=1)

# Calculate the average for 'distance (sum)', 'sigma', and 'point' columns
average_df = combined_df.groupby(combined_df.columns, axis=1).mean()

# Read the first file to extract other values
res0_df = pd.read_csv(os.path.join(directory, 'res0.csv'))

# Create a new DataFrame with the required format
new_data = {
    'distance (sum)': average_df['distance (sum)'],
    'step': res0_df['step'],
    'real batch size': res0_df['real batch size'],
    'p': res0_df['p'],
    'point': average_df['point'],
    'sigma': average_df['sigma'],
    'correct': res0_df['correct'],
    'accuracy': res0_df['accuracy'],
    'points': res0_df['points'],
    'batch_size': res0_df['batch_size'],
    'num_iters': res0_df['num_iters'],
    'alpha': res0_df['alpha'],
    'num_batches': res0_df['num_batches'],
    'lr': res0_df['lr'],
    'cn': res0_df['cn'],
    'epochs': res0_df['epochs'],
    'dp': res0_df['dp'],
    'eps': res0_df['eps'],
    'optimizer': res0_df['optimizer'],
    'dataset': res0_df['dataset'],
    'model': res0_df['model'],
    'norm_type': res0_df['norm_type'],
    'save_freq': res0_df['save_freq'],
    'save_name': res0_df['save_name'],
    'res_name': res0_df['res_name'],
    'gamma': res0_df['gamma'],
    'dec_lr': res0_df['dec_lr'],
    'id': res0_df['id'],
    'seed': res0_df['seed'],
    'overwrite': res0_df['overwrite'],
    'poisson_train': res0_df['poisson_train'],
    'stage': res0_df['stage'],
    'reduction': res0_df['reduction'],
    'exp': res0_df['exp'],
    'less_point': res0_df['less_point']
}

new_df = pd.DataFrame(new_data)

# Ensure the heading order is exactly as specified
columns_order = [
    'distance (sum)', 'step', 'real batch size', 'p', 'point', 'sigma', 'correct', 'accuracy', 'points', 'batch_size', 
    'num_iters', 'alpha', 'num_batches', 'lr', 'cn', 'epochs', 'dp', 'eps', 'optimizer', 'dataset', 'model', 'norm_type', 
    'save_freq', 'save_name', 'res_name', 'gamma', 'dec_lr', 'id', 'seed', 'overwrite', 'poisson_train', 'stage', 
    'reduction', 'exp', 'less_point'
]

# Reorder the columns
new_df = new_df[columns_order]

# Save the new DataFrame to a CSV file
new_df.to_csv(os.path.join(directory, 'average_results.csv'), index=False)
