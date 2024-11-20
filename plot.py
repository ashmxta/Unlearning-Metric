import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.special
import math

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot privacy cost as a function of step.')
    parser.add_argument('--input_folder', type=str, help='Input folder with compo results', default='compo_results')
    parser.add_argument('--alpha', type=int, help='Renyi alpha parameter', default=10)
    parser.add_argument('--sigma', type=float, help='Noise scale sigma', default=1.0)
    parser.add_argument('--q', type=float, help='Sampling rate q', default=0.1)
    parser.add_argument('--cn', type=float, help='Constant cn', default=1.0)
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

def main():
    args = parse_arguments()

    input_folder = args.input_folder
    if not os.path.exists('plots'):
        os.makedirs('plots')

    ave_costs_folder = 'ave_costs'
    if not os.path.exists(ave_costs_folder):
        os.makedirs(ave_costs_folder)

    alpha = args.alpha
    sigma = args.sigma
    q = args.q
    cn = args.cn

    baseline = renyi_baseline(alpha, sigma, q, cn)
    all_data = {}

    for filename in os.listdir(input_folder):
        if filename.startswith('compo_res') and filename.endswith('.csv'):
            filepath = os.path.join(input_folder, filename)
            run_index = filename.split('compo_res')[1].split('.')[0]
            
            df = pd.read_csv(filepath)
            
            grouped = df.groupby('point')
            for point, group in grouped:
                if point not in all_data:
                    all_data[point] = []
                all_data[point].append(group)

                normalized_privacy_cost = group['Privacy cost'] / baseline

                # Create subfolder for the datapoint
                datapoint_folder = os.path.join('plots', f'point_{point}')
                if not os.path.exists(datapoint_folder):
                    os.makedirs(datapoint_folder)

                fig, axs = plt.subplots(1, 2, figsize=(20, 6))

                axs[0].scatter(group['step'], group['Privacy cost'], color='blue')
                axs[0].set_title(f'Raw Privacy Values, Run {run_index}, Datapoint index={point}')
                axs[0].set_xlabel('Step')
                axs[0].set_ylabel('Privacy cost')
                axs[0].set_yscale('log')
                axs[0].grid(True)
                axs[0].text(0.95, 0.01, f'Baseline: {baseline:.3f}', verticalalignment='bottom', horizontalalignment='right', transform=axs[0].transAxes, fontsize=10, color='gray')

                axs[1].scatter(group['step'], normalized_privacy_cost, color='green')
                axs[1].set_title(f'Privacy/Baseline Values, Run {run_index}, Datapoint index={point}')
                axs[1].set_xlabel('Step')
                axs[1].set_ylabel('Privacy/Baseline cost')
                axs[1].set_yscale('log')
                axs[1].grid(True)
                axs[1].text(0.95, 0.01, f'Baseline: {baseline:.3f}', verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes, fontsize=10, color='gray')

                plot_filename = os.path.join(datapoint_folder, f'privacy_costs_comparison_run_{run_index}.png')
                plt.savefig(plot_filename)
                print(f'Plot saved as {plot_filename}')

                plt.close()

    # Generate final graphs and CSV for each datapoint
    for point, data in all_data.items():
        combined_df = pd.concat(data)
        mean_df = combined_df.groupby('step')['Privacy cost'].mean().reset_index()

        # Save the average to CSV
        csv_filename = os.path.join(ave_costs_folder, f'ave_cost{point}.csv')
        mean_df.to_csv(csv_filename, index=False)
        print(f'CSV saved as {csv_filename}')

        # Plot the average
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))

        axs[0].scatter(mean_df['step'], mean_df['Privacy cost'], color='blue')
        axs[0].set_title(f'Average Privacy Values, Datapoint index={point}')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Privacy cost')
        axs[0].set_yscale('log')
        axs[0].grid(True)
        axs[0].text(0.95, 0.01, f'Baseline: {baseline:.3f}', verticalalignment='bottom', horizontalalignment='right', transform=axs[0].transAxes, fontsize=10, color='gray')

        normalized_mean = mean_df['Privacy cost'] / baseline

        axs[1].scatter(mean_df['step'], normalized_mean, color='green')
        axs[1].set_title(f'Average Privacy/Baseline Values, Datapoint index={point}')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Privacy/Baseline cost')
        axs[1].set_yscale('log')
        axs[1].grid(True)
        axs[1].text(0.95, 0.01, f'Baseline: {baseline:.3f}', verticalalignment='bottom', horizontalalignment='right', transform=axs[1].transAxes, fontsize=10, color='gray')

        plot_filename = os.path.join('plots', f'point_{point}', 'average_plot.png')
        plt.savefig(plot_filename)
        print(f'Plot saved as {plot_filename}')

        plt.close()

if __name__ == "__main__":
    main()
