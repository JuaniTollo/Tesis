#!/bin/bash

# Directorio donde están los archivos YAML
CONFIG_DIR="./yamls/hellaswag"
echo "Current directory: $(pwd)"
echo "YAML config directory: $CONFIG_DIR"

if [ ! -d "$CONFIG_DIR" ] || [ -z "$(ls $CONFIG_DIR/*.yaml 2> /dev/null)" ]; then
    echo "Error: No YAML files found in the directory $CONFIG_DIR."
    exit 1
fi

# Script run.sh path asumido en el mismo directorio de ejecución
RUN_SCRIPT_PATH="./run.sh"

# Verificar que el script run.sh existe y es ejecutable
if [ ! -x "$RUN_SCRIPT_PATH" ]; then
    echo "Error: run.sh not found or is not executable in the directory $(pwd)."
    exit 1
fi

# Loop through all the YAML configuration files
for config_file in "$CONFIG_DIR"/*.yaml; do
    echo "Running experiment with config: $config_file"
    bash "$RUN_SCRIPT_PATH" "$config_file"
    
    # Check the exit status of the script
    if [ $? -ne 0 ]; then
        echo "Experiment failed with config: $config_file. Continuing with next."
    fi
    
    # Optional: Clear RAM Cache - Uncomment the next line if you have the necessary permissions
    # sudo sync && sudo echo 3 > /proc/sys/vm/drop_caches
done
echo "All experiments completed."
