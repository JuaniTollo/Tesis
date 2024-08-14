import csv
import json
import os
import argparse
from split import split
import pdb
from split import read_jsonl, sample_jsonl
def build_parser():
    parser = argparse.ArgumentParser(description="Predicting logits for different fine tunning strategies")
    
    parser.add_argument(
    "--input_dir",
    default="socialiqa",
    help="The path to the local input directory",
    )
    
    parser.add_argument(
    "--val_size",
    default=.01,
    help="The path to the local input directory",
)
    
    parser.add_argument(
    "--instruction_prompt",
    default="",
    help="The path to the local input directory",
)
    
    return parser

#default="Read the scenario and question below, then choose the most appropriate answer from the options given (1, 2, or 3). Respond with the number of the correct answer based on the scenario's context and the question's implications."

def convert_csv_to_jsonl(input_csv_path, output_jsonl_path):

    def format_input(row, instruction_prompt, output_jsonl_path):
        
        # Extraemos las partes de la ruta
        path_parts = output_jsonl_path.split(os.sep)

        # Reconstruimos la ruta solo con las primeras dos carpetas
        first_two_folders_path = os.path.join(path_parts[0], path_parts[1])
        #pdb.set_trace()
        if first_two_folders_path == "processed/socialiqa":
            context = row["inputs"].split('<context>')[1].split('</context>')[0].strip()
            #context = row["inputs"].split('<context>')[1].split('</context>')[0].strip()
            question = row["inputs"].split('<question>')[1].split('</question>')[0].strip()
            answerA = row["inputs"].split('<answerA>')[1].split('</answerA>')[0].strip()
            answerB = row["inputs"].split('<answerB>')[1].split('</answerB>')[0].strip()
            answerC = row["inputs"].split('<answerC>')[1].split('</answerC>')[0].strip()

            input_text = f"{instruction_prompt}{context}\nQuestion: {question}"
            input_text += f" \nA. {answerA}"
            input_text += f" \nB. {answerB}"
            input_text += f" \nC. {answerC}"
            input_text += f" \nAnswer: {row['targets']}"
            
            input_text = input_text.replace("Answer: 1", "Answer: A")
            input_text = input_text.replace("Answer: 2", "Answer: B")
            input_text = input_text.replace("Answer: 3", "Answer: C")
            
        if first_two_folders_path == "processed/hellaswag":
            context = row["inputs"].split('<ctx>')[1].split('<ending_options')[0].strip()

            # Extracción de las respuestas
            answerA = row["inputs"].split('<ending_options 0>')[1].split('</ending_options 0>')[0].strip()
            answerB = row["inputs"].split('<ending_options 1>')[1].split('</ending_options 1>')[0].strip()
            answerC = row["inputs"].split('<ending_options 2>')[1].split('</ending_options 2>')[0].strip()
            answerD = row["inputs"].split('<ending_options 3>')[1].split('</ending_options 3>')[0].strip()

            input_text = f"{instruction_prompt}{context}"
            input_text += f" \nA. {answerA}"
            input_text += f" \nB. {answerB}"
            input_text += f" \nC. {answerC}"
            input_text += f" \nD. {answerD}"

            # Asumiendo que row['targets'] contiene una letra que indica la respuesta correcta, A, B, C, o D
            input_text += f" \nAnswer: {row['targets']}"
            
            # Reemplaza los números por letras en el texto de salida.
            
            input_text = input_text.replace("</ctx>", "")
            input_text = input_text.replace("Answer: 0", "Answer: A")
            input_text = input_text.replace("Answer: 1", "Answer: B")
            input_text = input_text.replace("Answer: 2", "Answer: C")
            input_text = input_text.replace("Answer: 3", "Answer: D")

        return input_text
        

# * {"text": "Skylar spent the beautiful day at the beach with all of their friends and family.\nQuestion: How would Skylar feel afterwards?\nA. sad to be sunburned\nB. grateful for a loving family\nC. angry to be excluded\nAnswer: B"}
        return input_text
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    # Read the CSV file and write to JSONL
    with open(input_csv_path, mode='r', encoding='utf-8') as csv_file, open(output_jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            formatted_text = format_input(row, instruction_prompt,output_jsonl_path)
            json.dump({"text": formatted_text}, jsonl_file)
            jsonl_file.write('\n')

def process_directory(input_dir, output_dir):
    # List all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_csv_path = os.path.join(input_dir, filename)
            output_jsonl_path = os.path.join(output_dir, filename.replace('.csv', '.jsonl'))
            convert_csv_to_jsonl(input_csv_path, output_jsonl_path)
            print(f'Processed {input_csv_path} and saved to {output_jsonl_path}')

def simplify_filename(filename):
    """
    Simplifies a filename by keeping only the first part before the first dot
    and the file extension. Intended for filenames like 'test.socialiqa.jsonl'
    to be simplified to 'test.jsonl'.
    """
    parts = filename.split('.')
    if len(parts) > 2:
        new_filename = f"{parts[0]}.jsonl"
    else:
        new_filename = filename

    return new_filename

import os

def rename_validation_to_valid(input_dir):
    """
    Renames 'validation.jsonl' to 'valid.jsonl' within the specified directory.
    """
    old_filename = 'validation.jsonl'
    new_filename = 'valid.jsonl'
    old_path = os.path.join(input_dir, old_filename)
    new_path = os.path.join(input_dir, new_filename)

    # Check if the original file exists
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed '{old_filename}' to '{new_filename}'.")
    else:
        print(f"'{old_filename}' does not exist in the directory.")

def overwrite_and_cleanup(input_dir: str):
    """
    Overwrites 'train.jsonl' with 'train_split.jsonl' and deletes 'train_split.jsonl'.
    
    Args:
    input_dir (str): Directory where the files are located.
    """
    train_split_path = os.path.join(input_dir, "train_split.jsonl")
    test_split_path = os.path.join(input_dir, "test_split.jsonl")

    train_path = os.path.join(input_dir, "train.jsonl")
    test_path = os.path.join(input_dir, "test.jsonl")

    # Check if 'train_split.jsonl' exists
    if os.path.exists(train_split_path):
        # Overwrite 'train.jsonl' with 'train_split.jsonl'
        os.replace(train_split_path, train_path)
        print(f"'{train_split_path}' has been renamed to '{train_path}', overwriting the original file.")
    else:
        print(f"The file '{train_split_path}' does not exist.")
    
    if os.path.exists(test_split_path):
        # Overwrite 'test.jsonl' with 'test_split.jsonl'
        os.replace(test_split_path, test_path)
        print(f"'{test_split_path}' has been renamed to '{test_path}', overwriting the original file.")
    else:
        print(f"The file '{test_split_path}' does not exist.")
# Example usage:
# overwrite_and_cleanup('/path/to/your/directory')

def rename_files_in_directory(input_dir):
    """
    Renames all files in the given directory according to the simplify_filename function.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            new_filename = simplify_filename(filename)
            original_path = os.path.join(input_dir, filename)
            new_path = os.path.join(input_dir, new_filename)
            # Example usage:
            os.rename(original_path, new_path)
            print(f'Renamed {filename} to {new_filename}')
    print(os.getcwd(), f"./{output_dir}/validation.jsonl")
    rename_validation_to_valid(f"{output_dir}")

import json

def filter_and_rewrite_jsonl(file_path, max_word_count):
    """ Abre un archivo JSONL, filtra las instancias con menos palabras que `max_word_count`, y reescribe el archivo con los datos filtrados. """
    filtered_data = []

    # Leer el archivo JSONL línea por línea
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)  # Carga cada línea como un objeto JSON
            if count_words_in_json(data) <= max_word_count:  # Cambiar a <= para incluir líneas con menos palabras
                filtered_data.append(data)
    
    # Reescribir el archivo original con los datos filtrados
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in filtered_data:
            json.dump(item, file)
            file.write('\n')  # Asegura que cada objeto JSON está en una nueva línea

# Función para contar palabras en JSON
def count_words_in_json(data):
    """Cuenta las palabras en una representación de string de un objeto JSON."""
    # Convertir el objeto JSON a un string
    json_string = json.dumps(data)
    # Contar las palabras en el string resultante
    word_count = len(json_string.split())
    return word_count

def list_jsonl_files(directory):
    """Lista todos los archivos .jsonl en el directorio especificado."""
    # Lista para almacenar los paths completos de los archivos .jsonl
    jsonl_files = []

    # Recorrer todos los archivos y directorios en el directorio especificado
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Verificar si el archivo tiene la extensión .jsonl
            if file.endswith('.jsonl'):
                # Añadir el path completo del archivo a la lista
                jsonl_files.append(os.path.join(root, file))

    return jsonl_files
import numpy as np

if __name__ == "__main__":
    np.random.seed(0)
    parser = build_parser()
    args = parser.parse_args()

    # Specify the input and output directories
    input_dir = os.path.join("raw", args.input_dir)
    output_dir = os.path.join("processed", args.input_dir)
    instruction_prompt = args.instruction_prompt
    
    process_directory(input_dir, output_dir)
    rename_files_in_directory(output_dir)
    
    for jsonl_set_path in list_jsonl_files(output_dir):
        filter_and_rewrite_jsonl(file_path=jsonl_set_path, max_word_count=100)
    
    split(output_dir,args.val_size, train_size=5000)
    overwrite_and_cleanup(output_dir)
    
    #Me aseguro que los samples tengan #5000 entrenamiento #300 validacion #2000 testeo
    d = {"train.jsonl":5000, "valid.jsonl":300, "test.jsonl":1000}
    for set in d.keys():
        file = os.path.join(output_dir, set)
        data = read_jsonl(file)
        sample_jsonl(file, file, sample_size = d[set], seed=42)