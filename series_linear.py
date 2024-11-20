import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.special
import math
from scipy.interpolate import interp1d

def parse_arguments():
    parser = argparse.ArgumentParser(description='Approximate average using linear interpolation on subsampled data.')
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
    if not os.path.exists('series_lin'):
        os.makedirs('series_lin')

    alpha = args.alpha
    sigma = args.sigma
    q = args.q
    cn = args.cn

    baseline = renyi_baseline(alpha, sigma, q, cn)
    all_data = {}
    mae_per_point_run = {}
    mae_per_point_avg = {}

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

    # Process each run for every point
    for point, runs in all_data.items():
        true_average = pd.concat(runs).groupby(level=0).mean()
        point_maes = []  # To store MAEs for all runs of a single point

        for run_index, run_data in enumerate(runs):
            subsampling_offsets = [0, 1, 2]  # Start at first, second, and third data points
            interpolated_results = []

            fig, axs = plt.subplots(2, 2, figsize=(20, 12))

            # Store interpolated data for final plot
            all_interpolated_values = []

            for i, offset in enumerate(subsampling_offsets):
                subsampled_run_data = run_data[offset::3]  # Take every third data point starting at offset
                x_coords = subsampled_run_data.index
                y_coords = subsampled_run_data.values

                # Perform linear interpolation
                f_interp = interp1d(x_coords, y_coords, kind='linear', fill_value='extrapolate')
                interpolated_values = f_interp(true_average.index)
                interpolated_results.append(interpolated_values)
                all_interpolated_values.append(interpolated_values)

                # Plot each subsampling's linear interpolation line in its own subplot
                axs[i // 2, i % 2].plot(x_coords, y_coords, 'o', label='Subsampled Points')
                axs[i // 2, i % 2].plot(true_average.index, interpolated_values, '--', label='Linear Interpolation')
                axs[i // 2, i % 2].set_title(f'Offset {offset} Subsampling, Run {run_index + 1}, Point {point}')
                axs[i // 2, i % 2].set_xlabel('Step')
                axs[i // 2, i % 2].set_ylabel('Privacy/Baseline cost')
                axs[i // 2, i % 2].set_yscale('log')
                axs[i // 2, i % 2].grid(True)
                axs[i // 2, i % 2].legend()

            # Calculate average approximation from subsampled series
            average_approximation = np.mean(interpolated_results, axis=0)

            # Compute MAE between true average and approximated average
            mae = np.mean(np.abs(true_average.values - average_approximation))
            mae_per_point_run[(point, run_index)] = mae
            point_maes.append(mae)

            # Plot true average, average approximation, and exact copies of subsampled interpolations
            axs[1, 1].scatter(true_average.index, true_average, color='red', label='True Average Over Runs', zorder=3)
            axs[1, 1].plot(true_average.index, average_approximation, color='blue', linewidth=2, label='Average Approximation')

            # Add each subsampled series exactly as in individual plots
            for j, interpolated_values in enumerate(all_interpolated_values):
                axs[1, 1].plot(true_average.index, interpolated_values, '--', label=f'Offset {subsampling_offsets[j]} Interpolation')

            axs[1, 1].set_title(f'Comparison of True vs. Approximated Average, Run {run_index + 1}, Point {point}')
            axs[1, 1].set_xlabel('Step')
            axs[1, 1].set_ylabel('Privacy/Baseline cost')
            axs[1, 1].set_yscale('log')
            axs[1, 1].grid(True)
            axs[1, 1].legend()

            plt.tight_layout()

            # Save the plot
            datapoint_folder = os.path.join('series_lin', f'point_{point}')
            if not os.path.exists(datapoint_folder):
                os.makedirs(datapoint_folder)
            plot_filename = os.path.join(datapoint_folder, f'comparison_run_{run_index + 1}.png')
            plt.savefig(plot_filename)
            plt.close()
            print(f'Plot saved as {plot_filename}')

        # Compute the average MAE for this point across all runs
        average_mae_for_point = np.mean(point_maes)
        mae_per_point_avg[point] = average_mae_for_point

    # Calculate the overall average MAE across all points and runs
    overall_average_mae = np.mean(list(mae_per_point_avg.values()))
    
    # Save MAE data to CSV including the average per point and the overall average
    mae_df = pd.DataFrame(list(mae_per_point_run.items()), columns=['Point_Run', 'MAE'])
    for point in mae_per_point_avg:
        mae_df = mae_df.append({'Point_Run': f'Point {point} Average', 'MAE': mae_per_point_avg[point]}, ignore_index=True)
    mae_df = mae_df.append({'Point_Run': 'Overall Average', 'MAE': overall_average_mae}, ignore_index=True)

    mae_csv_path = os.path.join('series_lin', 'mae_summary.csv')
    mae_df.to_csv(mae_csv_path, index=False)
    print(f'MAE summary saved as {mae_csv_path}')

if __name__ == "__main__":
    main()