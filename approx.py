import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import scipy.special

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot privacy cost as a function of step.')
    parser.add_argument('--file', type=str, default='compo_results', help='Name of the CSV file containing the data')
    parser.add_argument('--ave_file', type=str, default='ave_costs', help='Name of the average CSV file')
    parser.add_argument('--alpha', type=int, help='Renyi alpha parameter', default=8)
    parser.add_argument('--sigma', type=float, help='Noise scale sigma', default=1.0)
    parser.add_argument('--q', type=float, help='Sampling rate q', default=0.1)
    parser.add_argument('--cn', type=float, help='Constant cn', default=1.0)
    parser.add_argument('--method', type=str, choices=['linear', 'polynomial', 'nn', 'all'], default='all', help='Method to approximate the trend in the data')
    return parser.parse_args()

def binom(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

def renyi_baseline(alpha, sigma, q, cn):
    res = []
    for k in range(alpha + 1):
        coeff = np.log(binom(alpha, k) * math.pow(1 - q, alpha - k) * math.pow(q, k))
        expect = math.pow(cn, 2) * k * (k - 1) / (2 * math.pow(sigma, 2))
        res.append(coeff + expect)
    return scipy.special.logsumexp(res) / (alpha - 1)

def linear_interpolation(x, y, x_full):
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
    return f(x_full)

def moving_window_polynomial_interpolation(x, y, x_full, degree=2, window_size=5):
    y_interp = np.zeros_like(x_full)
    half_window = window_size // 2
    
    for i in range(len(x_full)):
        start = max(0, i - half_window)
        end = min(len(x) - 1, i + half_window)
        
        x_window = x[start:end + 1]
        y_window = y[start:end + 1]
        
        if len(x_window) < degree + 1:
            continue
        
        poly = PolynomialFeatures(degree)
        x_poly = poly.fit_transform(x_window.reshape(-1, 1))
        model = LinearRegression().fit(x_poly, y_window)
        
        x_full_poly = poly.fit_transform(x_full[i].reshape(-1, 1))
        y_interp[i] = model.predict(x_full_poly)[0]
        
    return y_interp

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x

def neural_network_interpolation(x, y, x_full):
    model = NeuralNet()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    x_train = torch.FloatTensor(x).view(-1, 1)
    y_train = torch.FloatTensor(y).view(-1, 1)
    x_test = torch.FloatTensor(x_full).view(-1, 1)

    model.train()
    for epoch in range(5000):  # Increased the number of epochs
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    model.eval()
    with torch.no_grad():
        predictions = model(x_test).view(-1).tolist()

    return predictions

def plot_interpolation(df, point, method, ave_df, baseline):
    point_data = df[df['point'] == point]
    x_values = point_data['step'].values
    y_values = point_data['Privacy cost'].values
    
    # Calculate privacy over baseline
    privacy_over_baseline = y_values / baseline
    
    # Sample every third value
    x_sampled = x_values[::3]
    y_sampled = privacy_over_baseline[::3]
    
    if method == 'linear':
        y_interp = linear_interpolation(x_sampled, y_sampled, x_values)
    elif method == 'polynomial':
        y_interp = moving_window_polynomial_interpolation(x_sampled, y_sampled, x_values)
    elif method == 'nn':
        y_interp = neural_network_interpolation(x_sampled, y_sampled, x_values)
    
    # Calculate differences and average difference for every point
    differences = y_interp - privacy_over_baseline
    avg_difference = np.mean(np.abs(differences))
    
    # Ensure the plots directory exists
    eval_folder = f'eval_point{point}'
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    
    # Plot sampled points and interpolation
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_sampled, y_sampled, 'o', label='Sampled Points (Every Third)', color='red')
    plt.plot(x_values, y_interp, '-', label=f'{method.capitalize()} Interpolation', color='blue')
    plt.title(f'{method.capitalize()} Interpolation for Point {point}')
    plt.xlabel('Step')
    plt.ylabel('Privacy Cost over Baseline')
    plt.legend()
    plt.grid(True)

    # Evaluation with average over runs
    if ave_df is not None:
        x_ave = ave_df['step'].values
        y_ave = ave_df['Privacy cost'].values
        y_ave_privacy_over_baseline = y_ave / baseline
        y_interp_ave = linear_interpolation(x_sampled, y_sampled, x_ave) if method == 'linear' else (
                       moving_window_polynomial_interpolation(x_sampled, y_sampled, x_ave) if method == 'polynomial' else
                       neural_network_interpolation(x_sampled, y_sampled, x_ave))

        # Calculate average error with the average run data
        eval_differences = y_interp_ave - y_ave_privacy_over_baseline
        eval_avg_difference = np.mean(np.abs(eval_differences))

        plt.subplot(1, 2, 2)
        plt.plot(x_ave, y_ave_privacy_over_baseline, 'o', label='Average over Runs', color='black')
        plt.plot(x_ave, y_interp_ave, '-', label=f'{method.capitalize()} Interpolation', color='blue')
        plt.title(f'{method.capitalize()} Interpolation Evaluation\nwith Average for Point {point}\nAverage Error: {eval_avg_difference:.2e}')
        plt.xlabel('Step')
        plt.ylabel('Privacy Cost over Baseline')
        plt.legend()
        plt.grid(True)

    plt.savefig(f'{eval_folder}/{method}_interpolation_point_{point}.png')
    plt.close()

    return avg_difference, np.sum(np.abs(differences)), len(x_values), len(x_sampled), eval_avg_difference

def main():
    args = parse_arguments()

    df = pd.read_csv(args.file, delimiter=',')
    ave_df = pd.read_csv(args.ave_file, delimiter=',')

    if df.empty or ave_df.empty:
        raise ValueError("One of the provided CSV files is empty.")

    alpha = args.alpha
    sigma = args.sigma
    q = args.q
    cn = args.cn

    baseline = renyi_baseline(alpha, sigma, q, cn)
    
    point = 0  # Only consider data point 0
    methods = ['linear', 'polynomial', 'nn'] if args.method == 'all' else [args.method]
    results = {method: [] for method in methods}
    total_points = 0
    total_sampled_points = 0
    total_errors = {method: 0 for method in methods}
    sampled_errors = {method: 0 for method in methods}
    eval_errors = {method: [] for method in methods}
    
    for method in methods:
        avg_difference, total_error, num_points, num_sampled, eval_avg_difference = plot_interpolation(df, point, method, ave_df, baseline)
        results[method].append(avg_difference)
        total_errors[method] += total_error
        sampled_errors[method] += num_sampled * avg_difference
        total_points += num_points
        total_sampled_points += num_sampled
        eval_errors[method].append(eval_avg_difference)

    non_sampled_errors = {
        method: (total_errors[method] - sampled_errors[method]) / (total_points - total_sampled_points)
        for method in methods
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(non_sampled_errors.keys(), non_sampled_errors.values(), color=['blue', 'orange', 'green'])
    plt.title('Average Errors for Different Interpolation Methods (Non-Sampled Points)')
    plt.xlabel('Method')
    plt.ylabel('Average Error')
    plt.xticks(rotation=45)
    for method, error in non_sampled_errors.items():
        plt.text(method, error, f'{error:.2e}', ha='center', va='bottom')
    plt.grid(True)
    plt.savefig(f'eval_point{point}/summary_average_errors.png')
    plt.close()

    # Save evaluation average errors to a CSV file
    eval_errors_df = pd.DataFrame.from_dict(eval_errors)
    eval_errors_df.to_csv(f'eval_point{point}/evaluation_average_errors.csv', index=False)

if __name__ == '__main__':
    main()
   