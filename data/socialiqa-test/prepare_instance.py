import json
import random

# Ruta del archivo de entrada y salida
input_file = 'socialiqa.jsonl'
output_file = 'test.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Cargamos la línea como un diccionario
        data = json.loads(line)
        
        # Crear el nuevo formato de texto
        new_text = f"{data['context']}\nQuestion: {data['question']}"
        new_text += f"\nA. {data['answerA']}\nB. {data['answerB']}\nC. {data['answerC']}"
        
        # Agregar una respuesta aleatoria
        new_text += f"\nAnswer: {random.choice(['A', 'B', 'C'])}"
        
        # Crear el nuevo diccionario con la clave "text"
        new_data = {'text': new_text}
        
        # Escribir la línea modificada en el archivo de salida
        json.dump(new_data, outfile)
        outfile.write('\n')