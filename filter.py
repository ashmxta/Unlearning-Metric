import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.special
import math
from scipy.signal import savgol_filter

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot privacy cost as a function of step using Savitzky-Golay filter.')
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

def main():
    args = parse_arguments()

    input_folder = args.input_folder
    if not os.path.exists('filter'):
        os.makedirs('filter')

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
            x_coords = subsampled_run_data.index
            y_coords = subsampled_run_data.values

            # Apply Savitzky-Golay filter
            # Window length should be odd and at least the size of the polynomial order + 2
            window_length = min(11, len(y_coords) if len(y_coords) % 2 == 1 else len(y_coords) - 1)
            polyorder = 2  # Polynomial order
            smoothed_values = savgol_filter(y_coords, window_length, polyorder)

            # Interpolate smoothed values to match average data indices
            f_interp = np.interp(average_data.index, x_coords, smoothed_values)
            
            # Calculate MAE
            mae = np.mean(np.abs(f_interp - average_data.values))
            mae_list.append(mae)

            datapoint_folder = os.path.join('filter', f'point_{point}')
            if not os.path.exists(datapoint_folder):
                os.makedirs(datapoint_folder)

            fig, axs = plt.subplots(1, 2, figsize=(20, 6))
            
            # Subsampled and line-connected individual run plot
            axs[0].plot(x_coords, y_coords, 'o', color='green', label='Subsampled Data')
            axs[0].plot(x_coords, smoothed_values, '--', color='blue', label='Savgol Filter')
            axs[0].set_title(f'Run {run_index+1}, Datapoint index={point}')
            axs[0].set_xlabel('Step')
            axs[0].set_ylabel('Privacy/Baseline cost')
            axs[0].set_yscale('log')
            axs[0].grid(True)

            # Full average plot with the Savgol filter line
            axs[1].scatter(average_data.index, average_data, color='red', label='Average Data')
            axs[1].plot(average_data.index, f_interp, '--', color='blue', label='Savgol Filter')
            axs[1].set_title(f'Full Average vs. Savgol, Run {run_index+1}, Datapoint index={point} - MAE: {mae:.4f}')
            axs[1].set_xlabel('Step')
            axs[1].set_ylabel('Privacy/Baseline cost')
            axs[1].set_yscale('log')
            axs[1].grid(True)
            axs[1].legend()

            # Ensure both plots have the same scale for x and y axes
            x_min, x_max = min(min(x_coords), average_data.index.min()), max(max(x_coords), average_data.index.max())
            y_min, y_max = min(min(y_coords), average_data.min()), max(max(y_coords), average_data.max())
            axs[0].set_xlim(x_min, x_max)
            axs[0].set_ylim(y_min, y_max)
            axs[1].set_xlim(x_min, x_max)
            axs[1].set_ylim(y_min, y_max)

            plot_filename = os.path.join(datapoint_folder, f'privacy_costs_comparison_run_{run_index+1}.png')
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
    mae_csv_path = os.path.join('filter', 'mae_summary.csv')
    mae_df.to_csv(mae_csv_path, index=False)
    print(f'MAE summary saved as {mae_csv_path}')

if __name__ == "__main__":
    main()
