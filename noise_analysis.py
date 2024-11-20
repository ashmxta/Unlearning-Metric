import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_noise_data(point_folder):
    """
    Load all the difference data for a specific point across all runs.
    """
    noise_data = {}
    for filename in os.listdir(point_folder):
        if filename.startswith('diff_run_') and filename.endswith('.csv'):
            run_index = int(filename.split('diff_run_')[1].split('_')[0])
            filepath = os.path.join(point_folder, filename)
            df = pd.read_csv(filepath)
            noise_data[run_index] = df['Privacy cost difference'].values
    return noise_data

def compute_correlation_matrix(noise_data):
    """
    Compute the correlation matrix of the noise at different steps for the same point.
    """
    df_noise = pd.DataFrame(noise_data)
    correlation_matrix = df_noise.corr()
    return correlation_matrix

def compute_summary_statistics(correlation_matrix):
    """
    Compute summary statistics for the correlation matrix.
    """
    correlations = correlation_matrix.values.flatten()
    correlations = correlations[~np.isnan(correlations)]  # Remove NaN values

    summary_stats = {
        'mean_correlation': np.mean(correlations),
        'std_dev_correlation': np.std(correlations),
        'median_correlation': np.median(correlations),
        'proportion_significant': np.mean(np.abs(correlations) > 0.5)
    }

    return summary_stats

def save_summary_stats_image(summary_statistics_results, output_folder):
    """
    Save the summary statistics for all points as an image.
    """
    summary_df = pd.DataFrame(summary_statistics_results).T
    summary_df.loc['average'] = summary_df.mean()

    fig, ax = plt.subplots(figsize=(12, len(summary_df) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, rowLabels=summary_df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.savefig(os.path.join(output_folder, 'summary_stats.png'), bbox_inches='tight')
    plt.close()

def main():
    noise_folder = 'noise'
    
    # Dictionary to store correlation results and summary stats for each point
    correlation_results = {}
    summary_statistics_results = {}

    for point_folder_name in os.listdir(noise_folder):
        point_folder = os.path.join(noise_folder, point_folder_name)
        if os.path.isdir(point_folder):
            point_index = int(point_folder_name.split('_')[1])
            
            # Load the noise data for the current point
            noise_data = load_noise_data(point_folder)
            
            # Compute the correlation matrix for the noise at different steps
            correlation_matrix = compute_correlation_matrix(noise_data)
            
            # Store the correlation matrix in the results dictionary
            correlation_results[point_index] = correlation_matrix
            
            # Compute and store summary statistics
            summary_stats = compute_summary_statistics(correlation_matrix)
            summary_statistics_results[point_index] = summary_stats
            
            # Save the correlation matrix to a CSV file in the appropriate subfolder
            correlation_csv_filename = os.path.join(point_folder, f'correlation_matrix_point_{point_index}.csv')
            correlation_matrix.to_csv(correlation_csv_filename)
            print(f'Correlation matrix saved for point {point_index} as {correlation_csv_filename}')
            
            # Save the summary statistics to a CSV file in the appropriate subfolder
            summary_stats_csv_filename = os.path.join(point_folder, f'summary_stats_point_{point_index}.csv')
            summary_stats_df = pd.DataFrame([summary_stats])
            summary_stats_df.to_csv(summary_stats_csv_filename, index=False)
            print(f'Summary statistics saved for point {point_index} as {summary_stats_csv_filename}')
    
    # Save the combined summary statistics as an image
    save_summary_stats_image(summary_statistics_results, noise_folder)
    print(f'Summary statistics image saved as summary_stats.png in the {noise_folder} folder')
    
    return correlation_results, summary_statistics_results

if __name__ == "__main__":
    correlation_results, summary_statistics_results = main()
