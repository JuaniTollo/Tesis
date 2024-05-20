#!/bin/bash

# Check if a YAML file was passed as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_yaml>"
    exit 1
fi

CONFIG_YAML=$1

# Base directory calculation
BASE_DIR=$(dirname "$(realpath "$0")")

# Use explicitly the Python interpreter from the virtual environment
source "$BASE_DIR/../../env/bin/activate" || { echo "Failed to activate virtual environment"; exit 1; }
echo "Virtual environment activated at: $VIRTUAL_ENV"

# Read configurations from the provided YAML file
MODEL=$(yq e '.model' "$CONFIG_YAML")
BASE_MODEL=$(yq e '.base-model' "$CONFIG_YAML")  # Ensure correct key is used, not 'based-model'
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
    EXPERIMENT_DIR="${BASE_DIR}/output/${MODEL}/$(basename ${DATA})/${LORA_LAYERS}/${BATCH_SIZE}/${ITERS}/${LEARNING_RATE}//"
fi

echo "Experiment directory: $EXPERIMENT_DIR"

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
CMD="python -m mlx_lm.lora \
  --model \"$MODEL\" \
  --test \
  --adapter \"adapters\" \
  --data \"$DATA_DIR\""
#  --base_model \"$BASE_MODEL\" \
# Execute the experiment command
echo "Running command: $CMD"
eval "$CMD"
exit 0
