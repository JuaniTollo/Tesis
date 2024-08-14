#!/bin/bash

# Determina el directorio donde se encuentra run.sh
BASE_DIR=$(dirname "$(realpath "$0")")
echo $BASE_DIR 
# Activar el entorno virtual

# For Bash
# Correctly source the conda.sh script to initialize Conda
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the correct Conda environment
conda activate tesis_conda

# Ruta al archivo YAML proporcionada como primer argumento
YAML_PATH="$1"
echo $YAML_PATH
# Verifica la existencia del archivo YAML
if [ ! -f "$YAML_PATH" ]; then
    echo "Error: '$YAML_PATH' no such file or directory"
    exit 1
fi

# Crear directorios y preparar entorno del experimento
"$BASE_DIR/create_dirs.sh" "$YAML_PATH"

# Leer configuraciones del archivo YAML
MODEL=$(yq e '.model' "$YAML_PATH")
DATA=$(yq e '.data' "$YAML_PATH")
ITERS=$(yq e '.iters' "$YAML_PATH")
ADAPTER=$(yq e '.adapter-path' "$YAML_PATH")
SEED=$(yq e '.seed' "$YAML_PATH")
LORA_LAYERS=$(yq e '.lora-layers' "$YAML_PATH")
BATCH_SIZE=$(yq e '.batch-size' "$YAML_PATH")
LEARNING_RATE=$(yq e '.learning-rate' "$YAML_PATH")
BASE_MODEL=$(yq e '.base-model' "$YAML_PATH")

echo $DATA, $MODEL, $BASE_MODEL

# Cambiar al directorio del experimento que ahora incluye learning rate
EXPERIMENT_DIR="$BASE_DIR/output/${MODEL}/$(basename ${DATA})/${LORA_LAYERS}/${BATCH_SIZE}//${ITERS}/${LEARNING_RATE}"
mkdir -p "$EXPERIMENT_DIR"
cd "$EXPERIMENT_DIR" || exit 1

# Construir la ruta al directorio de datos que está al mismo nivel que el directorio del script
DATA_DIR=$(realpath "$BASE_DIR/../data/processed/$DATA")

CURVAS_PATH="$EXPERIMENT_DIR/Train_loss.csv"
if [ ! -f "$FILE_PATH" ]; then
  # Variables para intentos
  max_attempts=10
  attempt=1

  while [ $attempt -le $max_attempts ]; do
    CMD="python -m mlx_lm.lora \
        --model \"$MODEL\" \
        --train \
        --data \"$DATA_DIR\" \
        --iters $ITERS \
        --seed $SEED \
        --lora-layers $LORA_LAYERS \
        --adapter-path \"$ADAPTER\" \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE"
    echo "Running command in $EXPERIMENT_DIR: $CMD"
    eval $CMD

    # Chequear el código de salida del comando
    status=$?
    if [ $status -eq 0 ]; then
      echo "Command completed successfully."
      break
    else
      echo "Command failed with exit code $status. Attempt $attempt of $max_attempts."
      # Cambiar a un comando que intente usar el adaptador del último checkpoint si falla
      CMD="python -m mlx_lm.lora \
          --model \"$MODEL\" \
          --train \
          --data \"$DATA_DIR\" \
          --iters $ITERS \
          --seed $SEED \
          --lora-layers $LORA_LAYERS \
          --adapter-path \"$ADAPTER\" \
          --resume-adapter-file ./adapters_best_val/adapters.safetensors \
          --batch-size $BATCH_SIZE \
          --learning-rate $LEARNING_RATE"
      echo "Retrying with adapter recovery..."
    fi
    attempt=$((attempt + 1))
  done

  if [ $attempt -gt $max_attempts ]; then
    echo "Maximum attempts reached without success."
  fi
fi

# Definir la ubicación del archivo
TEST_TARGETS_PATH="$EXPERIMENT_DIR//train_all_targets.npy"

# Verificar si existe el archivo
if [ ! -f "$TEST_TARGETS_PATH" ] && [ -f "$CURVAS_PATH" ]; then
    ADAPTER_TEST_PATH="$EXPERIMENT_DIR/adapters_best_val"
    # Ejemplo adicional con adapter-path y test
    CMD="python -m mlx_lm.lora \
      --model \"$MODEL\" \
      --adapter-path \"$ADAPTER_TEST_PATH\" \
      --test \
      --data \"$DATA_DIR\""
    echo "Running command: $CMD"
    eval "$CMD"
else
    echo "El archivo 'test_all_targets.npy' existe en la ubicación $EXPERIMENT_DIR"
fi
