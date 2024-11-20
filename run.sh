#!/bin/bash
# Iterate from run value 4 to 9

run=3
for i in {0..40}
do
    stage=$(echo "scale=3; $i / 40" | bc) # beginning of each epoch
    seed=$RANDOM # generate a random seed value
    python compute_sensitivity.py --stage $stage --save-name "ckpt${run}" --res-name "res${run}" --seed $seed
done

# to make the script executable: chmod +x run.sh
# to run: ./run.sh
# to run in background: nohup ./run.sh > output.log 2>&1 &
