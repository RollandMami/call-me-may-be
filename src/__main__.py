import json
import argparse
import importlib
import numpy as np
from math import inf
from pathlib import Path
from .function import FunctionModel
from llm_sdk import Small_LLM_Model
try:
    load_dotenv = importlib.import_module("dotenv").load_dotenv
except (ImportError, AttributeError):
    print("Importation error: python-dotenv n'est pas installé")
    raise

MAX_TOKENS = 100


def parse_args() -> argparse.Namespace:
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

def check_args(args: argparse.Namespace) -> None:
    if not Path(args.functions_definition).exists():
        raise FileNotFoundError("Function definitions file not found")
    if not Path(args.input).exists():
        raise FileNotFoundError("Input file not found")
    

def get_func_name(model, allowed_func, eos_ids, prompt) -> str:
    input_ids = model.encode(prompt.strip())[0].tolist()
    current_prefix = "fn"
    for _ in range(MAX_TOKENS):
        allowed_tokens = set()
        for func in allowed_func:
            if func.startswith(current_prefix):
                ids = model.encode(func)[0].tolist()
                allowed_tokens.update(ids)
        allowed_tokens.update(eos_ids)
        # list token proba:
        logits = model.get_logits_from_input_ids(input_ids)
        logits = [
            -float(inf) if index not in allowed_tokens else logit
            for index, logit in enumerate(logits)
        ]
        # obtention du max proba
        next_id = int(np.argmax(logits))
        if next_id in eos_ids:
            break
        text = model.decode([next_id])
        if "\n" in text:
            break
        if current_prefix in allowed_func:
            break
        if any(func.startswith(current_prefix+text) for func in allowed_func):
            current_prefix += text
            input_ids.append(next_id)
        else:
            break
    final_output = model.decode(input_ids)
    print("\nRésultat généré :\n", final_output)
    if current_prefix in allowed_func:
        return current_prefix
    else:
        return "Null"

def main() -> None:
    load_dotenv()
    model = Small_LLM_Model()
    eos_ids = set(model.encode("<|im_end|>")[0].tolist())
    
    args = parse_args()
    try:
        check_args(args)
        # 0. definition des variables:
        function_json_path = args.functions_definition
        prompt_path = args.input
        output_file_path = Path(args.output)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)


        #1. recuperation du nom des fonctions:
            # 1.1 ouverture du ficher function:
        with open(function_json_path, "r") as f:
            function_db = json.load(f)
        allowed_func= [item.get("name").lower() for item in function_db if "name" in item]

        #2. recuperation des promptes
            # 2.1 boucle pour chaque prompte
        with open(prompt_path, "r") as p:
            prompt_db = json.load(p)
        prompts = [item.get("prompt").lower() for item in prompt_db if "prompt" in item]

        for prompt in prompts:
            print(get_func_name(model, allowed_func, eos_ids, prompt))
        
        # GENERATION DU RESULTAT
        #-----------------------
        # ECRITURE DE L'OUTPUT


    except KeyboardInterrupt as k:
        print("interuption keyboard", k)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as k:
        print("interuption keyboard", k)
    except Exception as e:
        print(e)
