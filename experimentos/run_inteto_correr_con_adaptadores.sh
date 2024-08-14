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
# Construir la ruta al directorio de datos que está al mismo nivel que el directorio del script
DATA_DIR=$(realpath "$BASE_DIR/../data/processed/$DATA")

# Cambiar al directorio del experimento que ahora incluye learning rate
EXPERIMENT_DIR="$BASE_DIR/output/${MODEL}/$(basename ${DATA})/${LORA_LAYERS}/${BATCH_SIZE}//${ITERS}/${LEARNING_RATE}"

# Cambia al directorio del experimento
cd "$EXPERIMENT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: No se pudo cambiar al directorio del experimento $EXPERIMENT_DIR"
    exit 1
fi

CURVAS_PATH="$EXPERIMENT_DIR//Train_loss.csv"

 if [ ! -f "$CURVAS_PATH" ]; then
     echo "El archivo $CURVAS_PATH no existe."
    # Verificar si el directorio ya existe
    if [ -d "$EXPERIMENT_DIR" ]; then
        echo "El directorio del experimento ya existe: $EXPERIMENT_DIR"

        ADAPTER_BEST="$EXPERIMENT_DIR/adapters_best_val/adapters.safetensors"
        if [ ! -f "$ADAPTER_BEST" ]; then
            echo "Error: Falta archivo de adaptador en $ADAPTER_PATH. Considera usar la opción --train para generar los adaptadores."
            exit 1
        fi

            # Extraer los números de iteración de los nombres de los archivos de adaptadores y encontrar el más grande
            LAST_ITERATION=$(ls $ADAPTER | grep -o '[0-9]\+_adapters.safetensors' | cut -d'_' -f1 | sort -nr | head -n 1)

            if [ ! -z "$LAST_ITERATION" ]; then
                echo "La última iteración de adaptador guardada fue la número: $LAST_ITERATION"
            else
                echo "No se encontraron archivos de adaptadores válidos."
                LAST_ITERATION=""
            fi
        else
            echo "No existe el directorio de adaptadores o está vacío: $ADAPTER_DIR"
            LAST_ITERATION=""
        fi

        # Aquí puedes calcular cuántas iteraciones te faltan correr
        TOTAL_ITERS=$ITERS  # Asumiendo que este es tu total de iteraciones objetivo
        if [ ! -z "$LAST_ITERATION" ]; then
            ITERS_LEFT=$((TOTAL_ITERS - LAST_ITERATION))
            echo "Te faltan correr $ITERS_LEFT iteraciones."
        else
            echo "Deberás correr todas las $TOTAL_ITERS iteraciones desde el inicio."
        fi

        CMD="python -m mlx_lm.lora \
            --model \"$MODEL\" \
            --train \
            --data \"$DATA_DIR\" \  
            --iters $ITERS_LEFT \
            --seed $SEED \
            --lora-layers $LORA_LAYERS \
            --adapter-path \"$ADAPTER\" \
            --resume-adapter-file ./adapters_best_val/adapters.safetensors \
            --batch-size $BATCH_SIZE \
            --learning-rate $LEARNING_RATE"
        echo "Running command in $EXPERIMENT_DIR: $CMD"
        #eval $CMD
        if [ ! -z "$LAST_ITERATION" ] && [ "$LAST_ITERATION" -gt 0 ]; then
            ITERS_LEFT=$((TOTAL_ITERS - LAST_ITERATION))
            echo "Iterations left to run: $ITERS_LEFT"
            if [ "$ITERS_LEFT" -gt 0 ]; then
                CMD="python -m mlx_lm.lora --model \"$MODEL\" --train --data \"$DATA_DIR\" --iters $ITERS_LEFT --seed $SEED --lora-layers $LORA_LAYERS --adapter-path \"$ADAPTER\" --resume-adapter-file ./adapters_best_val/adapters.safetensors --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE"
                echo "Executing: $CMD"
                eval $CMD
            fi
        else
            echo "Invalid iteration count detected or no iterations are left to run."
        fi
    else
        echo "Creando el directorio del experimento: $EXPERIMENT_DIR"
        mkdir -p "$EXPERIMENT_DIR"
        # Definir el directorio del adaptador

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
