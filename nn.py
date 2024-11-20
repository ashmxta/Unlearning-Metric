import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import scipy.special

# Define the neural network structure
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Adjust width as necessary
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot privacy cost as a function of step using NN interpolation.')
    parser.add_argument('--input_folder', type=str, help='Input folder with compo results', default='compo_results')
    parser.add_argument('--alpha', type=int, help='Renyi alpha parameter', default=10)
    parser.add_argument('--sigma', type=float, help='Noise scale sigma', default=1.0)
    parser.add_argument('--q', type=float, help='Sampling rate q', default=0.1)
    parser.add_argument('--cn', type=float, help='Constant cn', default=1.0)
    return parser.parse_args()

def binom(n, k):
    """Calculates binomial coefficient"""
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

def renyi_baseline(alpha, sigma, q, cn):
    """Calculates the Renyi divergence baseline"""
    res = []
    for k in range(alpha + 1):
        coeff = np.log(binom(alpha, k) * math.pow(1 - q, alpha - k) * math.pow(q, k))
        expect = math.pow(cn, 2) * k * (k - 1) / (2 * math.pow(sigma, 2))
        res.append(coeff + expect)
    return scipy.special.logsumexp(res) / (alpha - 1)

def train_network(x_train, y_train, model, criterion, optimizer, epochs=500):
    """Trains the neural network"""
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

def main():
    args = parse_arguments()

    input_folder = args.input_folder
    if not os.path.exists('nn'):
        os.makedirs('nn')

    alpha = args.alpha
    sigma = args.sigma
    q = args.q
    cn = args.cn

    baseline = renyi_baseline(alpha, sigma, q, cn)
    all_data = {}
    mae_per_point = {}

    # Load and process data
    for filename in os.listdir(input_folder):
        if filename.startswith('compo_res') and filename.endswith('.csv'):
            filepath = os.path.join(input_folder, filename)
            df = pd.read_csv(filepath)
            grouped = df.groupby('point')
            for point, group in grouped:
                if point not in all_data:
                    all_data[point] = []
                all_data[point].append(group['Privacy cost'] / baseline)

    # Calculate and plot for each point
    for point, runs in all_data.items():
        average_data = pd.concat(runs).groupby(level=0).mean()
        mae_list = []
        
        for run_index, run_data in enumerate(runs):
            subsampled_run_data = run_data[::3]  # Take every third data point
            x_train = torch.tensor(subsampled_run_data.index.values.reshape(-1, 1)).float()
            y_train = torch.tensor(subsampled_run_data.values.reshape(-1, 1)).float()

            # Initialize and train the neural network
            model = SimpleNet()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            train_network(x_train, y_train, model, criterion, optimizer)

            # Predict using the trained model
            x_pred = torch.tensor(average_data.index.values.reshape(-1, 1)).float()
            predictions = model(x_pred).detach().numpy().flatten()

            # Calculate Mean Absolute Error (MAE)
            mae = np.mean(np.abs(predictions - average_data.values))
            mae_list.append(mae)

            datapoint_folder = os.path.join('nn', f'point_{point}')
            if not os.path.exists(datapoint_folder):
                os.makedirs(datapoint_folder)

            # Plot the results
            fig, axs = plt.subplots(1, 2, figsize=(20, 6))
            axs[0].plot(x_train.numpy(), y_train.numpy(), 'o', color='green', label='Subsampled Data')
            axs[0].plot(x_pred.numpy(), predictions, '--', color='blue', label='NN Predicted')
            axs[0].set_title(f'Run {run_index+1}, Datapoint index={point}')
            
            axs[1].scatter(average_data.index, average_data, color='red', label='Average Data')
            axs[1].plot(x_pred.numpy(), predictions, '--', color='blue', label='NN Predicted')
            axs[1].set_title(f'Full Average vs. NN, Run {run_index+1}, Datapoint index={point} - MAE: {mae:.4f}')
            
            for ax in axs:
                ax.set_xlabel('Step')
                ax.set_ylabel('Privacy/Baseline cost')
                ax.set_yscale('log')
                ax.grid(True)
                ax.legend()

            # Save the plot
            plot_filename = os.path.join(datapoint_folder, f'privacy_cost_nn_run_{run_index+1}.png')
            plt.savefig(plot_filename)
            plt.close()
            print(f'Plot saved as {plot_filename}')

        # Calculate average MAE for this point
        average_mae = np.mean(mae_list)
        mae_per_point[point] = average_mae

    # Calculate the overall average MAE across all points
    overall_average_mae = np.mean(list(mae_per_point.values()))
    
    # Save MAE data to CSV including the overall average
    mae_df = pd.DataFrame(list(mae_per_point.items()), columns=['Point', 'Average MAE'])
    mae_df.loc[len(mae_df.index)] = ['Overall Average', overall_average_mae]
    mae_csv_path = os.path.join('nn', 'mae_summary.csv')
    mae_df.to_csv(mae_csv_path, index=False)
    print(f'MAE summary saved as {mae_csv_path}')

if __name__ == "__main__":
    main()
