from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import mlx.nn as nn
import mlx.core as mx
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

# Función para cargar logits y targets
def cargar_logits_y_targets(dir, prefix):
    logits_path = f"{dir}/{prefix}_all_logits.npy"
    targets_path = f"{dir}/{prefix}_all_targets.npy"
    try:
        logits = np.load(logits_path)
        targets = np.load(targets_path)
        return logits, targets
    except FileNotFoundError:
        #print(f"No se pudo encontrar '{logits_path}' o '{targets_path}', saltando este conjunto de datos.")
        return None, None

# Función para calcular métricas (Ejemplo)
def calcular_metricas(logits, targets):
    
    np.bincount(targets, minlength=3)
    values, counts = np.unique(targets, return_counts=True)
    counts / sum(counts)

    predictions = np.argmax(logits, axis=1)
    
    accuracy = accuracy_score(targets, predictions)

    logits_sf = mx.softmax(mx.array(logits), axis=1)

    indices = np.unique(targets)
    selected_logits = np.take(logits_sf, indices, axis=1)
    # Uso esta a continuacion para la CE normalizada
    priors_to_targets = np.sum(selected_logits, axis=1).mean()

    relevant_logits = logits[:, indices]
    predictions = np.argmax(relevant_logits, axis=1)

    index_to_position = {index: pos for pos, index in enumerate(indices)}
    mapped_targets = np.array([index_to_position[target] for target in targets if target in index_to_position])
    accuracy_restricted = accuracy_score(mapped_targets, predictions)
    cross_entropy = nn.losses.cross_entropy(mx.array(logits), mx.array(targets), reduction="mean").item()
    epsilon = 1e-10  # Añadir epsilon para evitar el log de cero
    priors = np.bincount(targets, minlength=logits.shape[1]) / targets.shape[0]
    
    # TODO: Volar epsilon y recalcular crosentropy
    priors *= epsilon
    priors = np.repeat(priors.reshape(1, -1), len(targets), axis=0)

    priors = np.clip(priors, a_min=1e-10, a_max=None)  # Ensure all values are >= 1e-10
    cross_entropy_priors = nn.losses.cross_entropy(mx.array(np.log(priors)), mx.array(targets), reduction="mean").item()
    accuracy = np.mean(np.argmax(logits, axis=1) == targets)
    accuracy_restricted = accuracy  # Supón que es una métrica calculada de manera similar
    
    return accuracy, accuracy_restricted, priors_to_targets, cross_entropy, cross_entropy_priors

# Función para calcular métricas para cada conjunto de datos
def calcular_metricas_para_conjuntos(dir, prefixes):
    resultados = {}
    for prefix in prefixes:
        try:
            logits, targets = cargar_logits_y_targets(dir, prefix)
        except:
            print(f"no se pudo cargar el dataset de {dir}")
        if logits is None or targets is None:
            print("logits or targets están vacíos")
            continue
        
        try:
            metrics = calcular_metricas(logits, targets)
            metric_names = ["accuracy", "accuracy_restricted", "priors_to_targets", "cross_entropy", "cross_entropy_priors"]
        except:
            print("no se pudieron computar las metricas")
        for name, value in zip(metric_names, metrics):
            resultados[f"{name}_{prefix}"] = value
    
    return resultados

