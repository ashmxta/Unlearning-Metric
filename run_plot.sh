#!/bin/bash
# Create plots for all res_compoX.csv files in the compo_results folder

# Ensure the plotting script is executable
chmod +x plot.py

# Run the plotting script for the compo_results folder
python plot.py --input_folder compo_results

echo "All plots have been generated."

# to make the script executable: chmod +x run_plot.sh
# to run: ./run_plot.sh
# to run in background: nohup ./run_plot.sh > output_plots.log 2>&1 &
# tail -f output_plots.log --> check output
