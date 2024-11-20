#!/bin/bash
if [ -n "$wait_pid" ]; then
    echo "Waiting for run.sh to complete..."
    while kill -0 "$wait_pid" 2>/dev/null; do
        sleep 1
    done
fi

echo "run.sh has completed. Starting composition calculations..."

# Compute composition for resX.csv files
X = A
do
    python renyi_per_instance_sum_compo.py "res/res${X}.csv"
done

echo "Composition calculations completed."

# to make the script executable: chmod +x run_compo.sh
# to run: ./run_compo.sh
# to run in background: nohup ./run_compo.sh > output_compo.log 2>&1 &
# tail -f output_compo.log --> check output
