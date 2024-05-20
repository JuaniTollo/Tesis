import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from utils import *
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
    
    if parts[6] != "Base":
        info = {
            "Empresa": parts[3],  # El tercer elemento es la empresa
            "Modelo": parts[4],   # El cuarto elemento es el modelo
            "Dataset": parts[5],  # El quinto elemento es el dataset
            "Capas LoRa": parts[6],  # El sexto elemento son las capas LoRa
            "Tamaño batch": parts[7],  # El séptimo elemento es el tamaño del batch
            "Iters": parts[8], 
            "Learning rate": str(parts[9]),  # El octavo elemento es la tasa de aprendizaje
            "Base/Fine-tuning": "Fine-tuning",
            "Adapter": "Best"
            }
    else:
        
        info = {
            "Empresa": parts[3],  # El tercer elemento es la empresa
            "Modelo": parts[4],   # El cuarto elemento es el modelo
            "Dataset": parts[5],  # El quinto elemento es el dataset
            "Capas LoRa": "n/a",  # El sexto elemento son las capas LoRa
            "Tamaño batch": "n/a",  # El séptimo elemento es el tamaño del batch
            "Learning rate": "n/a",  # El octavo elemento es la tasa de aprendizaje
            "Iters":parts[7],
            "Base/Fine-tuning": "Base",
            "Adapter": "n/a"
}               
    #pdb.set_trace()
    return info
def cargar_logits_y_targets(dir):
    logits = np.load(dir + "/test_all_logits.npy")
    targets = np.load(dir + "/test_all_targets.npy")
    return logits, targets

def calcular_metricas(logits,targets):
    # Si ambos archivos existen, procede con el procesamiento
        np.bincount(targets, minlength=3)
        
        values, counts = np.unique(targets, return_counts=True)
        counts / sum(counts)

        predictions = np.argmax(logits, axis=1)
        
        accuracy = accuracy_score(targets, predictions)

        logits_sf = mx.softmax(mx.array(logits), axis=1)

        indices = np.unique(targets)
        selected_logits = np.take(logits_sf, indices, axis=1)
        priors_to_targets = np.sum(selected_logits, axis=1).mean()

        relevant_logits = logits[:, indices]
        predictions = np.argmax(relevant_logits, axis=1)

        index_to_position = {index: pos for pos, index in enumerate(indices)}
        mapped_targets = np.array([index_to_position[target] for target in targets if target in index_to_position])
        accuracy_restricted = accuracy_score(mapped_targets, predictions)

        cross_entropy = nn.losses.cross_entropy(mx.softmax(mx.array(logits)), mx.array(targets), reduction="mean").item()
        epsilon = 1e-10  # Añadir epsilon para evitar el log de cero
        priors = np.bincount(targets, minlength=logits.shape[1]) / targets.shape[0]
        priors *= epsilon
        priors = np.repeat(priors.reshape(1, -1), len(targets), axis=0)

        priors = np.clip(priors, a_min=1e-10, a_max=None)  # Ensure all values are >= 1e-10
        cross_entropy_priors = nn.losses.cross_entropy(mx.array(np.log(priors)), mx.array(targets), reduction="mean").item()
        
        return accuracy, accuracy_restricted, priors_to_targets,cross_entropy,cross_entropy_priors

# Initialize a DataFrame to hold all results
def iniciailizarDataSet():
    results_df = pd.DataFrame(columns=["Empresa","Modelo", "Dataset","Capas LoRa", "Tamaño batch", "Iters", "Learning rate","Adapter", "Accuracy", "Proportion Targets in Softmax", "Accuracy restringido", "Entropía cruzada","Entropía cruzada normalizada"])
    results_df['Learning rate'] = results_df['Learning rate'].astype(str)
    return results_df


results_df = iniciailizarDataSet()

root_directory = '../experimentos/output/'
ignore_folders = ['adapters', 'adapters_best_val', 'tokenizer', 'resultados con adapters',"socialiqa_debug"]

leaf_dirs = list_leaf_directories(root_directory, ignore_folders)


for dir in leaf_dirs[:]:
    try:
        #print("Procesando directorio:", dir)
        try:
            info_experimento = parse_path(dir)
        except:
            print(f"No se pudo parsear {dir}")

        try:
            logits, targets = cargar_logits_y_targets(dir)
        except:
            print(f"No se pudo encontrar 'test_all_logits.npy' o 'test_all_targets.npy' en {dir}, saltando este directorio.")
            continue
        accuracy, accuracy_restricted, priors_to_targets,cross_entropy,cross_entropy_priors = calcular_metricas(logits,targets)

        # Preparar un DataFrame temporal con los resultados para este directorio
        temp_df = pd.DataFrame({
            "Empresa": info_experimento["Empresa"],
            "Dataset": info_experimento["Dataset"],     
            "Modelo": info_experimento["Modelo"],
            "Base/Fine-tuning": info_experimento["Base/Fine-tuning"],
            "Capas LoRa": info_experimento["Capas LoRa"],
            "Tamaño batch": info_experimento["Tamaño batch"],
            "Learning rate": info_experimento["Learning rate"],
            "Iters": info_experimento["Iters"],
            "Adapter": info_experimento["Adapter"],
            "Accuracy": [accuracy],
            "Proportion Targets in Softmax": [priors_to_targets],
            "Accuracy restringido": [accuracy_restricted],
            "Entropía cruzada": [cross_entropy],
            "Entropía cruzada normalizada": [cross_entropy / cross_entropy_priors],
        })
        temp_df['Learning rate'].astype(str)

        # Concatenar el DataFrame temporal al principal
        #pdb.set_trace()
        if not results_df.dropna(how='all', axis=1).empty:
            results_df = pd.concat([results_df, temp_df], ignore_index=True)
        else:
            results_df = temp_df
        
        
    except Exception as e:
        print(f"Ocurrió un error inesperado procesando el directorio {dir}: {str(e)}")
    
    #pdb.set_trace()
    
results_df.to_csv("results_df.csv")

        
results_df.to_csv("results.csv")
pd.set_option('display.max_columns', 10)  # Máximo número de columnas a mostrar
pd.set_option('display.max_rows', 10)     # Máximo número de filas a mostrar
pd.set_option('display.width', 100)       # Ajustar la anchura para la visualización del DataFrame
pd.set_option('display.max_colwidth', 20) # Máximo ancho de las columnas
        
    
