#!/bin/bash

# Directory where the Python script is located
script_directory="/h/321/ashmita/Gradients-Look-Alike-Sensitivity-is-Often-Overestimated-in-DP-SGD/Gradients-Look-Alike-Sensitivity-is-Often-Overestimated-in-DP-SGD"

# Loop through each set of files from 0 to 9
for i in {0..9}
do
    echo "Processing set $i"
    python $script_directory/approx.py --file "$script_directory/compo_results/compo_res${i}.csv" --ave_file "$script_directory/ave_costs/ave_cost${i}.csv"
done

echo "All processing complete."

# Instructions for script usage:
# To make the script executable:
# chmod +x run_approx.sh

# To run the script:
# ./run_approx.sh

# To run the script in the background and redirect output:
# nohup ./run_approx.sh > output_approx.log 2>&1 &

# To check the background process output:
# tail -f output_approx.log
