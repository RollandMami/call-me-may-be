import json
import argparse
import importlib
from pathlib import Path
import numpy as np
from llm_sdk import Small_LLM_Model
try:
    load_dotenv = importlib.import_module("dotenv").load_dotenv
except (ImportError, AttributeError):
    print("Importation error: python-dotenv n'est pas installé")
    raise

MAX_TOKENS = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Test de function calling")
    
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Chemin du fichier des définitions de fonctions",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Chemin du fichier JSON des tests",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Chemin du fichier JSON de sortie",
    )
    
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    model = Small_LLM_Model()
    eos_ids = set(model.encode("<|im_end|>")[0].tolist())
    
    args = parse_args()
    # 0. definition des variables:
    function_json_path = args.functions_definition
    prompt_path = args.input
    output_file_path = args.output
    output_dir = Path.__dir__(output_file_path)

    #1. recuperation du nom des fonctions:
        # 1.1 ouverture du ficher function:

    #2. recuperation des promptes


    prompt = "prompt: What is the sum of 2 and 3?"
    prompt += "\nAnswer:"
    # encodage: transforme le prompt en tensor
    input_ids = model.encode(prompt)[0].tolist()

    for _ in range(MAX_TOKENS):
        # list token proba:
        logits = model.get_logits_from_input_ids(input_ids)
        # obtention du max proba
        next_id = int(np.argmax(logits))
        if next_id in eos_ids:
            break
        text = model.decode([next_id])
        if "\n" in text:
            break

        input_ids.append(next_id)

        # vocab_path = model.get_path_to_vocab_file()
        
        # with open(vocab_path, "r") as f:
        #     vocab_data = {v: k for k, v in json.load(f).items()}

    print("="*10)
    print(f"Decoded: \n{model.decode([input_ids])[0]}")

    # GENERATION DU RESULTAT
    #-----------------------
    # ECRITURE DE L'OUTPUT
    if not Path.exists(output_dir):
        Path.mkdir(output_dir)

    with open(output_file_path, "a") as f:
        f.write(model.decode([input_ids])[0])

if __name__ == "__main__":
    main()
