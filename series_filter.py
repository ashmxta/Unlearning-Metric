import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.special
import math
from scipy.signal import savgol_filter

def parse_arguments():
    parser = argparse.ArgumentParser(description='Approximate average using Savitzky-Golay filter on subsampled data.')
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

def main():
    args = parse_arguments()

    input_folder = args.input_folder
    if not os.path.exists('series_filter'):
        os.makedirs('series_filter')

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
        interpolated_results = []

        fig, axs = plt.subplots(2, 2, figsize=(20, 12))

        # Store interpolated data for final plot
        all_interpolated_values = []

        for i, offset in enumerate(subsampling_offsets):
            subsampled_run_data = run_data[offset::3]  # Take every third data point starting at offset
            x_coords = subsampled_run_data.index
            y_coords = subsampled_run_data.values

            # Apply Savitzky-Golay filter
            window_length = min(11, len(y_coords) if len(y_coords) % 2 == 1 else len(y_coords) - 1)
            polyorder = 2
            smoothed_values = savgol_filter(y_coords, window_length, polyorder)

            # Interpolate smoothed values to match true average indices
            f_interp = np.interp(true_average.index, x_coords, smoothed_values)
            interpolated_values = f_interp
            interpolated_results.append(interpolated_values)
            all_interpolated_values.append(interpolated_values)

            # Plot each subsampling's Savitzky-Golay filter line in its own subplot
            axs[i // 2, i % 2].plot(x_coords, y_coords, 'o', label='Subsampled Points')
            axs[i // 2, i % 2].plot(true_average.index, interpolated_values, '--', label='Savgol Filter')
            axs[i // 2, i % 2].set_title(f'Offset {offset} Subsampling, Point {point}')
            axs[i // 2, i % 2].set_xlabel('Step')
            axs[i // 2, i % 2].set_ylabel('Privacy/Baseline cost')
            axs[i // 2, i % 2].set_yscale('log')
            axs[i // 2, i % 2].grid(True)
            axs[i // 2, i % 2].legend()

        # Calculate average approximation from subsampled series
        average_approximation = np.mean(interpolated_results, axis=0)

        # Compute sum error between true average and approximated average
        sum_error = np.sum(true_average.values - average_approximation)
        sum_error_per_point_run[point] = sum_error
        point_errors.append(sum_error)

        # Plot true average, average approximation, and exact copies of subsampled interpolations
        axs[1, 1].scatter(true_average.index, true_average, color='red', label='True Average Over Runs', zorder=3)
        axs[1, 1].plot(true_average.index, average_approximation, color='blue', linewidth=2, label='Average Approximation')

        # Add each subsampled series exactly as in individual plots
        for j, interpolated_values in enumerate(all_interpolated_values):
            axs[1, 1].plot(true_average.index, interpolated_values, '--', label=f'Offset {subsampling_offsets[j]} Interpolation')

        axs[1, 1].set_title(f'Comparison of True vs. Approximated Average, Point {point}')
        axs[1, 1].set_xlabel('Step')
        axs[1, 1].set_ylabel('Privacy/Baseline cost')
        axs[1, 1].set_yscale('log')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        plt.tight_layout()

        # Save the plot
        datapoint_folder = os.path.join('series_filter', f'point_{int(point)}')
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

    sum_error_csv_path = os.path.join('series_filter', 'sum_error_summary.csv')
    sum_error_df.to_csv(sum_error_csv_path, index=False)
    print(f'Sum error summary saved as {sum_error_csv_path}')

if __name__ == "__main__":
    main()
