#!/bin/bash

# Leer el archivo YAML pasado como parámetro
YAML_PATH=$1
MODEL=$(yq e '.model' $YAML_PATH)
DATA=$(yq e '.data' $YAML_PATH)
DATA_NAME=$(basename $DATA)  # Extraer solo el nombre del dataset
LORA_LAYERS=$(yq e '.lora-layers' $YAML_PATH)
BATCH_SIZE=$(yq e '.batch-size' $YAML_PATH)
LEARNING_RATE=$(yq e '.learning-rate' "$YAML_PATH")
ITERS=$(yq e '.iters' "$YAML_PATH")


if [ -z "$BASE_MODEL" ]; then
    echo "BASE_MODEL is not set. Assuming False."
    BASE_MODEL="False"
fi

if [ "$BASE_MODEL" != "True" ]; then
EXPERIMENT_DIR="output/${MODEL}/${DATA_NAME}/${LORA_LAYERS}/${BATCH_SIZE}/${ITERS}/${LEARNING_RATE}"
else
EXPERIMENT_DIR="output/${MODEL}/${DATA_NAME}/${ITERS}"
fi

# Crear el árbol de directorios
mkdir -p $EXPERIMENT_DIR

# Copiar el archivo YAML al directorio del experimento
cp $YAML_PATH $EXPERIMENT_DIR/

# Cambiar al directorio del experimento

# Crear un enlace simbólico en la carpeta principal para fácil acceso
#ln -sfn $(realpath $EXPERIMENT_DIR) "$(dirname $YAML_PATH)/experiment_link"

echo "Directorio del experimento y enlace creado en: $EXPERIMENT_DIR"
