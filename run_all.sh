#!/bin/bash

# run_all_experiments.sh - Batch run experiment script

set -e  # Exit on error

# Path to the inner script
INNER_SCRIPT="./run.sh"

# Check if the inner script exists
if [ ! -f "$INNER_SCRIPT" ]; then
    echo "Error: Inner script $INNER_SCRIPT not found"
    exit 1
fi

# Parameter arrays
DATASET_NAMES=('math' 'college_math' 'gsm8k')
CUT_COEFFS=(0.03 0.1 0.3 0.5)
# CUT_COEFFS=(0.3 0.5 0.7 0.9 1.0)

# Fixed parameters
SMALL_COEFF=0.5
BIG_COEFF=1.5
BALANCE_COEFF=0.5
NUM_SAMPLES=100
R=0.5

echo "Starting batch experiments..."
echo "Datasets: ${DATASET_NAMES[@]}"
echo "CUT_COEFF values: ${CUT_COEFFS[@]}"
echo "================================"

# Record start time
start_time=$(date +%s)
total_experiments=$((${#DATASET_NAMES[@]} * ${#CUT_COEFFS[@]}))
current_experiment=0

# Double loop through all combinations
for dataset in "${DATASET_NAMES[@]}"; do
    for cut_coeff in "${CUT_COEFFS[@]}"; do
        current_experiment=$((current_experiment + 1))
        
        echo "[$current_experiment/$total_experiments] Running experiment:"
        echo "  DATASET_NAME: $dataset"
        echo "  CUT_COEFF: $cut_coeff"
        echo "  Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        
        # Set environment variables and call the inner script
        export DATASET_NAME="$dataset"
        export CUT_COEFF="$cut_coeff"
        export SMALL_COEFF="$SMALL_COEFF"
        export BIG_COEFF="$BIG_COEFF"
        export BALANCE_COEFF="$BALANCE_COEFF"
        export NUM_SAMPLES="$NUM_SAMPLES"
        export R="$R"
        
        # Run the inner script
        if bash "$INNER_SCRIPT"; then
            echo "Experiment $current_experiment finished"
        else
            echo "Experiment $current_experiment failed"
            # Optionally stop on failure
            # exit 1  # Uncomment to stop on failure
        fi
        
        echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "--------------------------------"
    done
done

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "All experiments finished!"
echo "Total experiments: $total_experiments"
echo "Total time: ${total_time} seconds"
