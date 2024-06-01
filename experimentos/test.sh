#!/bin/bash

CONFIG_YAML=$1

# Base directory calculation
BASE_DIR=$(dirname "$(realpath "$0")")

# Correctly source the conda.sh script to initialize Conda
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the correct Conda environment
conda activate tesis_conda

# Read configurations from the provided YAML file
MODEL=$(yq e '.model' "$CONFIG_YAML")
BASE_MODEL=$(yq e '.based-model' "$CONFIG_YAML")  # Corrected to match the YAML field
DATA=$(yq e '.data' "$CONFIG_YAML")
ADAPTER=$(yq e '.adapter-path' "$CONFIG_YAML" || echo "")
echo "base model" $BASE_MODEL
# Construct the data directory path which is at the same level as the script directory
DATA_DIR=$(realpath "$BASE_DIR/../data/processed/$DATA")
LORA_LAYERS=$(yq e '.lora-layers' "$CONFIG_YAML")
BATCH_SIZE=$(yq e '.batch-size' "$CONFIG_YAML")
LEARNING_RATE=$(yq e '.learning-rate' "$CONFIG_YAML")
ITERS=$(yq e '.iters' "$CONFIG_YAML")

echo "Learning rate" $LEARNING_RATE
echo "Data directory: $DATA_DIR", 

if [ "$BASE_MODEL" = true ]; then
    EXPERIMENT_DIR="$BASE_DIR/output/${MODEL}/${DATA}/Base/"
else
    # Change directly to the directory where the YAML file is located
    EXPERIMENT_DIR=$(dirname "$CONFIG_YAML")

cd "$EXPERIMENT_DIR" || exit 1
echo "Changed to directory of the YAML file: $EXPERIMENT_DIR"

# Intenta crear el directorio si no existe
if [ ! -d "$EXPERIMENT_DIR" ]; then
    mkdir -p "$EXPERIMENT_DIR"
    if [ $? -eq 0 ]; then
        echo "Experiment directory created successfully"
    else
        echo "Failed to create experiment directory"
        exit 1
    fi
else
    echo "Experiment directory already exists"
fi
# Change to the experiment directory
cd "$EXPERIMENT_DIR" || exit 1
echo "Changed to experiment directory successfully"

# Properly quote the command components to handle paths with spaces
CMD="python -m memory_profiler mlx_lm.lora \
  --model $MODEL \
  --base_model \"$BASE_MODEL\" \
  --test \
  --adapter \"adapters_best_val\" \
  --data \"$DATA_DIR\""
#  --base_model \"$BASE_MODEL\" \
# Execute the experiment command
echo "Running command: $CMD"
eval "$CMD"
exit 0
