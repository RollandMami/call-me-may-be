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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Function calling")
    parser.add_argument(
        "--functions_definition",
        required=True,
        help="Chemin du fichier JSON des définitions de fonctions",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Chemin du fichier JSON des tests",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Chemin du fichier JSON de sortie",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    model = Small_LLM_Model()
    eos_ids = set(model.encode("<|im_end|>")[0].tolist())
    

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

if __name__ == "__main__":
    main()
