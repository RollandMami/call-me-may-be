import json
import argparse
import traceback
import numpy as np
from pathlib import Path
from .function import FunctionModel
from llm_sdk import Small_LLM_Model
try:
    from dotenv import load_dotenv  # type: ignore
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activer le mode debug",
    )
    
    return parser.parse_args()

def check_args(args: argparse.Namespace) -> None:
    if not Path(args.functions_definition).exists():
        raise FileNotFoundError("Function definitions file not found")
    if not Path(args.input).exists():
        raise FileNotFoundError("Input file not found")
    
def build_prompt(user_prompt: str, function_db: list) -> str:
    lines = []
    for f in function_db:
        params = ", ".join(
            f"{name}: {p.get('type', 'unknown')}"
            for name, p in f.get("parameters", {}).items()
        )
        lines.append(f"- {f['name']}({params}): {f.get('description', '')}")
    funcs = "\n".join(lines)
    prompt = (
        "<|im_start|>system\n"
        "Vous êtes un assistant qui doit identifier le nom d'une fonction "
        "à partir d'un prompt donné. Vous devez répondre uniquement par le nom "
        "de la fonction, sans explications ni commentaires. "
        "Si aucune fonction ne correspond, répondez par 'Null'.\n\n"
        f"Fonctions disponibles :\n{funcs}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

def build_candidate_tokens(model, allowed_fucntion) -> dict:
    with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    candidates = {}
    for txt, tid in vocab.items():
        if txt and any(txt in fn for fn in allowed_fucntion):
            candidates[tid] = txt
    return candidates

def get_func(model: Small_LLM_Model, allowed_func: list, eos_ids, prompt: str, args) -> str:
    input_ids = model.encode(prompt.strip())[0].tolist()
    current_prefix = ""
    for _ in range(MAX_TOKENS):
        candidates = build_candidate_tokens(model, ...)
        allowed = {
            tid for tid, txt in candidates.items()
            if any(f.startswith(current_prefix + txt) for f in allowed_func)
        }
        # list token proba:
        logits = np.array(model.get_logits_from_input_ids(input_ids), dtype=float)
        mask = np.full_like(logits, -np.inf)
        mask[list(allowed)] = 0
        # obtention du max proba
        next_id = int(np.argmax(logits + mask))
        if next_id in eos_ids:
            break
        text = model.decode([next_id])
        
        if args.debug:
            print('>>>',text)
        if "\n" in text:
            break

        if any(func.startswith(current_prefix+text) for func in allowed_func):
            current_prefix += text
            input_ids.append(next_id)
        else:
            break
    final_output = model.decode(input_ids)
    print("\nRésultat généré :\n", final_output)
    return current_prefix if current_prefix in allowed_func else "Null"

def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        check_args(args)
        model = Small_LLM_Model()
        eos_ids = set(model.encode("<|im_end|>")[0].tolist())
        # 0. definition des variables:
        function_json_path = args.functions_definition
        prompt_path = args.input
        output_file_path = Path(args.output)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)


        #1. recuperation du nom des fonctions:
            # 1.1 ouverture du ficher function:
        with open(function_json_path, "r") as f:
            function_db = json.load(f)
        allowed_func= [item.get("name") for item in function_db if "name" in item]

        #2. recuperation des promptes
            # 2.1 boucle pour chaque prompte
        with open(prompt_path, "r") as p:
            prompt_db = json.load(p)
        prompts = [item.get("prompt") for item in prompt_db if "prompt" in item]

        if not args.debug:
            for prompt in prompts:
                print(get_func(model, allowed_func, eos_ids, build_prompt(prompt, function_db), args))
        else:
            print(get_func(model, allowed_func, eos_ids, build_prompt(prompts[0], function_db), args))
        
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
        traceback.print_exc()
