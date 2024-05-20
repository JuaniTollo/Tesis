import json
import random
from typing import List, Tuple

def read_jsonl(file_path: str) -> List[dict]:
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def shuffle_data(data: List[dict]) -> None:
    """Shuffle the data in place."""
    # Establecer la semilla del generador de números aleatorios
    random.seed(0)
    random.shuffle(data)


def split_data(data: List[dict], val_size: int) -> Tuple[List[dict], List[dict]]:
    """Split the data into training and test sets based on the test size percentage."""
    # import pdb 
    # pdb.set_trace()
    val_size = float(val_size)
    split_index = int(len(data) * val_size)
    return data[split_index:], data[:split_index]

def save_jsonl(data: List[dict], file_path: str) -> None:
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')
import os
print("Current working directory:", os.getcwd())


def sample_jsonl(input_file, output_file, sample_size, seed=42):
    # Establecer la semilla para la reproducibilidad
    random.seed(seed)

    # Leer todas las líneas del archivo original
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Comprobar si la muestra es menor que el número total de líneas
    if sample_size > len(lines):
        raise ValueError("Sample size is larger than the total number of lines in the file.")
    
    # Obtener una muestra aleatoria de líneas
    sampled_lines = random.sample(lines, sample_size)

    # Escribir las líneas muestreadas en un nuevo archivo JSONL
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in sampled_lines:
            file.write(line)

def eliminar_archivo(file_path):
    # Comprobar si el archivo existe antes de intentar eliminarlo
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"El archivo {file_path} ha sido eliminado correctamente.")
        except OSError as e:
            print(f"No se pudo eliminar el archivo: {e}")
    else:
        print("El archivo no existe.")

def split(input_dir: str, val_size: float, train_size=5000):
    
    #Elimino el archivo de test original
    original_test_file_path = os.path.join(input_dir, "test.jsonl")
    eliminar_archivo(original_test_file_path)
    
    # Path original de validacion
    original_val_file_path = os.path.join(input_dir, "valid.jsonl")
    
    # Cambio el nombre de validacion a testeo
    os.rename(original_val_file_path, original_test_file_path)
    
    input_file_path = os.path.join(input_dir, "train.jsonl")
    print("Reading data from:", input_file_path)
    data = read_jsonl(input_file_path)
    # Creo un set de validacion que es el 20% del de entrenamiento
    shuffle_data(data)
    train_data, val_data = split_data(data, .20)
    
    # # Ensure output directory exists
    # os.makedirs(input_dir, exist_ok=True)
    
    train_output_path = os.path.join(input_dir, "train_split.jsonl")
    val_output_path = os.path.join(input_dir, "valid.jsonl")
    save_jsonl(train_data, train_output_path)
    save_jsonl(val_data, val_output_path)
    
    sample_jsonl(train_output_path, train_output_path, sample_size=train_size, seed=42)
    
    # print(f"Data split into {len(train_data)} training and {len(val_data)} test records.")
