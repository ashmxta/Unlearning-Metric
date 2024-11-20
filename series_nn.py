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
    parser = argparse.ArgumentParser(description='Approximate average using neural network interpolation on subsampled data.')
    parser.add_argument('--input_folder', type=str, help='Input folder with compo results', default='compo_res')
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
    if not os.path.exists('series_nn'):
        os.makedirs('series_nn')

    alpha = args.alpha
    sigma = args.sigma
    q = args.q
    cn = args.cn

    baseline = renyi_baseline(alpha, sigma, q, cn)
    sum_error_per_point_run = {}
    sum_error_per_point_avg = {}

    # Load and process data
    for filename in os.listdir(input_folder):
        if filename == 'compo_resA.csv':
            filepath = os.path.join(input_folder, filename)
            df = pd.read_csv(filepath)
            grouped = df.groupby('point')
            all_data = {point: group['Privacy cost'] / baseline for point, group in grouped}

    # Process each run for every point
    for point, run_data in all_data.items():
        true_average = run_data.groupby(level=0).mean()
        point_errors = []  # To store errors for all runs of a single point

        subsampling_offsets = [0, 1, 2]  # Start at first, second, and third data points
        nn_results = []

        fig, axs = plt.subplots(2, 2, figsize=(20, 12))

        # Store neural network predictions for final plot
        all_nn_predictions = []

        for i, offset in enumerate(subsampling_offsets):
            subsampled_run_data = run_data[offset::3]  # Take every third data point starting at offset
            x_train = torch.tensor(subsampled_run_data.index.values.reshape(-1, 1)).float()
            y_train = torch.tensor(subsampled_run_data.values.reshape(-1, 1)).float()

            # Initialize and train the neural network
            model = SimpleNet()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            train_network(x_train, y_train, model, criterion, optimizer)

            # Predict using the trained model
            x_pred = torch.tensor(true_average.index.values.reshape(-1, 1)).float()
            predictions = model(x_pred).detach().numpy().flatten()
            nn_results.append(predictions)
            all_nn_predictions.append(predictions)

            # Plot each subsampling's neural network result in its own subplot
            axs[i // 2, i % 2].plot(x_train.numpy(), y_train.numpy(), 'o', label='Subsampled Points')
            axs[i // 2, i % 2].plot(x_pred.numpy(), predictions, '--', label='NN Prediction')
            axs[i // 2, i % 2].set_title(f'Offset {offset} Subsampling, Point {point}')
            axs[i // 2, i % 2].set_xlabel('Step')
            axs[i // 2, i % 2].set_ylabel('Privacy/Baseline cost')
            axs[i // 2, i % 2].set_yscale('log')
            axs[i // 2, i % 2].grid(True)
            axs[i // 2, i % 2].legend()

        # Calculate average approximation from neural network results
        average_approximation = np.mean(nn_results, axis=0)

        # Compute sum error between true average and approximated average
        sum_error = np.sum(true_average.values - average_approximation)
        sum_error_per_point_run[point] = sum_error
        point_errors.append(sum_error)

        # Plot true average, average approximation, and exact copies of neural network predictions
        axs[1, 1].scatter(true_average.index, true_average, color='red', label='True Average Over Runs', zorder=3)
        axs[1, 1].plot(true_average.index, average_approximation, color='blue', linewidth=2, label='Average Approximation')

        # Add each neural network prediction exactly as in individual plots
        for j, nn_prediction in enumerate(all_nn_predictions):
            axs[1, 1].plot(true_average.index, nn_prediction, '--', label=f'Offset {subsampling_offsets[j]} Prediction')

        axs[1, 1].set_title(f'Comparison of True vs. Approximated Average, Point {point}')
        axs[1, 1].set_xlabel('Step')
        axs[1, 1].set_ylabel('Privacy/Baseline cost')
        axs[1, 1].set_yscale('log')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        plt.tight_layout()

        # Save the plot
        datapoint_folder = os.path.join('series_nn', f'point_{int(point)}')
        if not os.path.exists(datapoint_folder):
            os.makedirs(datapoint_folder)
        plot_filename = os.path.join(datapoint_folder, f'comparison_point_{int(point)}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f'Plot saved as {plot_filename}')

        # Compute the average sum error for this point across all runs
        average_sum_error_for_point = np.mean(point_errors)
        sum_error_per_point_avg[point] = average_sum_error_for_point

    # Calculate the overall average sum error across all points and runs
    overall_average_sum_error = np.mean(list(sum_error_per_point_avg.values()))

    # Save sum error data to CSV including the average per point and the overall average
    sum_error_df = pd.DataFrame(list(sum_error_per_point_run.items()), columns=['Point', 'Sum Error'])
    for point in sum_error_per_point_avg:
        sum_error_df = sum_error_df.append({'Point': f'Point {point} Average', 'Sum Error': sum_error_per_point_avg[point]}, ignore_index=True)
    sum_error_df = sum_error_df.append({'Point': 'Overall Average', 'Sum Error': overall_average_sum_error}, ignore_index=True)

    sum_error_csv_path = os.path.join('series_nn', 'sum_error_summary.csv')
    sum_error_df.to_csv(sum_error_csv_path, index=False)
    print(f'Sum error summary saved as {sum_error_csv_path}')

if __name__ == "__main__":
    main()
