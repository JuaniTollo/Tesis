import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from utils_iteraciones import *
import os
from pathlib import Path
import pdb

def is_leaf_directory(path, ignore_dirs=None):
    """ Verifica si un directorio es un directorio hoja, ignorando ciertos subdirectorios. """
    if ignore_dirs is None:
        ignore_dirs = []
    if os.path.isdir(path):
        # Lista todos los elementos en el directorio que no están en 'ignore_dirs'
        entries = [entry for entry in os.listdir(path) if entry not in ignore_dirs]
        for entry in entries:
            entry_path = os.path.join(path, entry)
            # Si algún elemento es un directorio, entonces 'path' no es hoja
            if os.path.isdir(entry_path):
                return False
        return True
    return False

def list_leaf_directories(root_dir, ignore_dirs=None):
    """ Lista todos los directorios hoja en el directorio raíz dado, ignorando ciertos subdirectorios. """
    leaf_directories = []
    for root, dirs, files in os.walk(root_dir):
        # Modificar 'dirs' in-place para ignorar directorios específicos
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        # Comprobar si 'root' es un directorio hoja considerando directorios a ignorar
        if is_leaf_directory(root, ignore_dirs):
            leaf_directories.append(root)
    return leaf_directories

def parse_path(path):
    # Dividir la ruta en partes
    
    parts = path.split('/')
    
    # Crear un diccionario para almacenar los datos extraídos
    #pdb.set_trace()
    info = {
            "Empresa": "Microsoft",  # El tercer elemento es la empresa
            "Modelo": "phi 2",   # El cuarto elemento es el modelo
            "Dataset": "hellaswag",  # El quinto elemento es el dataset
            "Capas LoRa": 16,  # El sexto elemento son las capas LoRa
            "Tamaño batch": 5,  # El séptimo elemento es el tamaño del batch
            "Iters": 5,
            "Lora_Rank": parts[4], 
            "Learning rate": f"{float(1e-5):.1e}",  # El octavo elemento es la tasa de aprendizaje
            "Base/Fine-tuning": "Fine-tuning",
            "Adapter": "Best"
            }
    return info

def cargar_logits_y_targets(dir):
    pdb.set_trace()
    logits = np.load(dir + "/../test_all_logits.npy")
    targets = np.load(dir + "/../test_all_targets.npy")
    return logits, targets

def clean_and_concat(df_main, df_temp):
    # Limpiar columnas con todos valores NA
    df_temp_cleaned = df_temp.dropna(how='all', axis=1)
    # Concatenar solo si df_temp_cleaned no está vacío
    if not df_temp_cleaned.empty:
        return pd.concat([df_main, df_temp_cleaned], ignore_index=True)
    return df_main

# Initialize a DataFrame to hold all results
def iniciailizarDataSet():
    results_df = pd.DataFrame(columns=["Empresa","Modelo", "Dataset","Capas LoRa", "Tamaño batch", "Iters", "Learning rate","Adapter", "Accuracy", "Proportion Targets in Softmax", "Accuracy restringido", "Entropía cruzada","Entropía cruzada normalizada"])
    results_df['Learning rate'] = results_df['Learning rate'].astype(str)
    return results_df

results_df = iniciailizarDataSet()

root_directory = '../experimentos/output_loss_vs_accuracy'
ignore_folders = ['adapters', 'adapters_best_val', 'tokenizer', 'resultados con adapters',"socialiqa_debug", "1e-5", "socialiqa"]

leaf_dirs = list_leaf_directories(root_directory, ignore_folders)
#pdb.set_trace()
for dir in leaf_dirs[:]:
    try:
        prefixes = ["train", "val", "test"]
        print(dir)

        resultados = calcular_metricas_para_conjuntos(dir, prefixes)
        info_experimento = parse_path(dir)
    except:
        print(f"No se pudo parsear {dir}")

        # Preparar un DataFrame temporal con los resultados para este directorio
    # Crear el DataFrame incluyendo métricas para train, val y test
   
    temp_df = pd.DataFrame({
        "Empresa": info_experimento["Empresa"],
        "Dataset": info_experimento["Dataset"],
        "Modelo": info_experimento["Modelo"],
        "Base/Fine-tuning": info_experimento["Base/Fine-tuning"],
        "Capas LoRa": info_experimento["Capas LoRa"],
        "Tamaño batch": info_experimento["Tamaño batch"],
        "Learning rate": info_experimento['Learning rate'],
        "Iters": info_experimento["Iters"],
        "Adapter": info_experimento["Adapter"],
        "Lora_Rank": info_experimento["Lora_Rank"],
        "Accuracy Entrenamiento": [resultados.get("accuracy_restricted_train", None)],
        "Accuracy Validación": [resultados.get("accuracy_restricted_val", None)],
        "Accuracy Test": [resultados.get("accuracy_restricted_test", None)],
        "EC Entrenamiento": [resultados.get("cross_entropy_train", None)],
        "EC Validación": [resultados.get("cross_entropy_val", None)],
        "EC Testeo": [resultados.get("cross_entropy_test", None)],
        "ECN Entrenamiento": [resultados.get("cross_entropy_train", None) / resultados.get("cross_entropy_priors_train", 1) if resultados.get("cross_entropy_priors_train") else None],
        "ECN Validación": [resultados.get("cross_entropy_val", None) / resultados.get("cross_entropy_priors_val", 1) if resultados.get("cross_entropy_priors_val") else None],
        "ECN Testeo": [resultados.get("cross_entropy_test", None) / resultados.get("cross_entropy_priors_test", 1) if resultados.get("cross_entropy_priors_test") else None],
    })
    if not results_df.dropna(how='all', axis=1).empty:
        #results_df = pd.concat([results_df, temp_df], ignore_index=True)
        # Usar la función para limpiar y concatenar
        results_df = clean_and_concat(results_df, temp_df)
    
    else:
        results_df = temp_df

# Aplicar notación científica a la columna 'Learning rate'
results_df.to_csv("results_loss_vs_accuracy.csv", index=False)

        
    
