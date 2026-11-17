import json
import re
import argparse
import traceback
import numpy as np
from pathlib import Path
from typing import Any, cast
from numpy.typing import NDArray
from llm_sdk import Small_LLM_Model

try:
    from dotenv import load_dotenv
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


def encode_ids(model: Small_LLM_Model, text: str) -> list[int]:
    return cast(list[int], model.encode(text)[0].tolist())


def check_args(args: argparse.Namespace) -> None:
    if not Path(args.functions_definition).exists():
        raise FileNotFoundError("Function definitions file not found")
    if not Path(args.input).exists():
        raise FileNotFoundError("Input file not found")


def build_prompt(user_prompt: str, function_db: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for f in function_db:
        params = ", ".join(
            f"{name}: {p.get('type', 'unknown')}"
            for name, p in f.get("parameters", {}).items()
        )
        lines.append(f"- {f['name']}({params}): {f.get('description', '')}")
    funcs = "\n".join(lines)
    prompt = (
        f"<|im_start|>system\nVous êtes un assistant qui doit identifier l"
        f"e nom d'une fonction à partir d'un prompt donné. Vous devez répo"
        f"ndre uniquement par le nom de la fonction, sans explications ni "
        f"commentaires. Si aucune fonction ne correspond, répondez par 'Nu"
        f"ll'.\n\nFonctions disponibles :\n"
        f"{funcs}"
        f"<|im_end|>\n<|im_start|>user\n"
        f"{user_prompt}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
    return prompt


def build_args_prompt(user_prompt: str, func: dict[str, Any]) -> str:
    params = ", ".join(
        f"{k}: {v.get('type', 'unknown')}"
        for k, v in func.get("parameters", {}).items()
    )
    return (
        "<|im_start|>system\n"
        "Tu extrais les valeurs des arguments d'une fonction à partir de la "
        "demande de l'utilisateur. Réponds uniquement en JSON.\n\n"
        f"Fonction : {func['name']}({params}) : {func.get('description', '')}"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f'{{"function": "{func["name"]}", "arguments": {{'
    )


def build_candidate_tokens(
    model: Small_LLM_Model, allowed_func: list[str]
) -> dict[int, str]:
    with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    candidates: dict[int, str] = {}
    for txt, tid in vocab.items():
        if txt and any(txt in fn for fn in allowed_func):
            candidates[tid] = txt
    return candidates


def build_value_tokens(
    model: Small_LLM_Model,
) -> tuple[set[int], set[int], dict[int, str]]:
    with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    number_ids = {
        tid for txt, tid in vocab.items() if re.fullmatch(r"[0-9.\-]+", txt)
    }
    stop_ids = {vocab[t] for t in (",", "}") if t in vocab}
    string_texts = {
        tid: model.decode([tid]) for tid in vocab.values()
    }
    return number_ids, stop_ids, string_texts


def get_func(
    model: Small_LLM_Model,
    allowed_func: list[str],
    eos_ids: set[int],
    prompt: str,
    candidates: dict[int, str],
    args: argparse.Namespace,
) -> str | None:
    input_ids = encode_ids(model, prompt.strip())
    current_prefix: str = ""
    for _ in range(MAX_TOKENS):
        allowed = {
            tid
            for tid, txt in candidates.items()
            if any(f.startswith(current_prefix + txt) for f in allowed_func)
        }
        if current_prefix in allowed_func:
            allowed |= eos_ids

        logits: NDArray[np.float64] = np.array(
            model.get_logits_from_input_ids(input_ids), dtype=np.float64
        )
        mask: NDArray[np.float64] = np.full_like(logits, -np.inf)
        mask[list(allowed)] = 0
        next_id: int = int(np.argmax(logits + mask))
        if next_id in eos_ids:
            break
        current_prefix += candidates[next_id]
        input_ids.append(next_id)

        if args.debug:
            print(">>>", repr(candidates[next_id]), "->", current_prefix)

        if current_prefix in allowed_func and not any(
            f != current_prefix and f.startswith(current_prefix)
            for f in allowed_func
        ):
            break

    return current_prefix if current_prefix in allowed_func else None


def generate_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    kind: str,
    eos_ids: set[int],
    number_ids: set[int],
    stop_ids: set[int],
    string_texts: dict[int, str],
) -> str:
    """Génère une valeur. Modifie input_ids. Renvoie le texte brut."""
    text = ""
    for _ in range(MAX_TOKENS):
        logits: NDArray[np.float64] = np.array(
            model.get_logits_from_input_ids(input_ids), dtype=float
        )
        mask: NDArray[np.float64] = np.full_like(logits, -np.inf)

        if kind in ("number", "integer"):
            allowed = set(number_ids)
            if text:
                allowed |= stop_ids | eos_ids
        else:
            allowed = {
                tid
                for tid, t in string_texts.items()
                if "\n" not in t and "\\" not in t
            }
            allowed |= eos_ids

        mask[list(allowed)] = 0
        next_id = int(np.argmax(logits + mask))

        if kind in ("number", "integer"):
            if next_id in stop_ids or next_id in eos_ids:
                break
            text += string_texts[next_id] if next_id in string_texts else ""
            input_ids.append(next_id)
        else:
            if next_id in eos_ids:
                break
            piece = string_texts[next_id]
            if '"' in piece:
                text += piece.split('"')[0]
                break
            text += piece
            input_ids.append(next_id)
    return text


def get_arguments(
    model: Small_LLM_Model,
    prompt: str,
    func: dict[str, Any],
    eos_ids: set[int],
    value_tokens: tuple[set[int], set[int], dict[int, str]],
) -> dict[str, int | float | str]:
    number_ids, stop_ids, string_texts = value_tokens
    input_ids: list[int] = encode_ids(model, build_args_prompt(prompt, func))
    args_out: dict[str, int | float | str] = {}

    for i, (key, spec) in enumerate(func.get("parameters", {}).items()):
        kind: str = spec.get("type", "string")
        forced = (
            ("" if i == 0 else ", ")
            + f'"{key}": '
            + ('"' if kind == "string" else "")
        )
        input_ids += encode_ids(model, forced)

        raw = generate_value(
            model, input_ids, kind, eos_ids, number_ids, stop_ids, string_texts
        )

        if kind == "string":
            args_out[key] = raw
            input_ids += encode_ids(model, '"')
        elif kind == "integer":
            args_out[key] = int(float(raw)) if raw else 0
        else:
            try:
                args_out[key] = (
                    int(raw) if raw.lstrip("-").isdigit() else float(raw)
                )
            except ValueError:
                args_out[key] = 0
    return args_out


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        check_args(args)
        model: Small_LLM_Model = Small_LLM_Model()
        eos_ids: set[int] = set(encode_ids(model, "<|im_end|>"))
        # 0. definition des variables:
        function_json_path = args.functions_definition
        prompt_path = args.input
        output_file_path = Path(args.output)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. recuperation du nom des fonctions:
        # 1.1 ouverture du ficher function:
        with open(function_json_path, "r") as f:
            function_db: list[dict[str, Any]] = json.load(f)
        allowed_func: list[str] = [
            item["name"] for item in function_db if "name" in item
        ]
        candidates: dict[int, str] = build_candidate_tokens(
            model, allowed_func
        )
        value_tokens: tuple[set[int], set[int], dict[int, str]] = (
            build_value_tokens(model)
        )
        func_by_name: dict[str, dict[str, Any]] = {
            f["name"]: f for f in function_db}
        results: list[dict[str, Any]] = []

        with open(prompt_path, "r") as p:
            prompt_db: list[dict[str, Any]] = json.load(p)
        prompts = [
            item["prompt"] for item in prompt_db if "prompt" in item
        ]

        to_run = prompts[:1] if args.debug else prompts
        for prompt in to_run:
            name: str | None = get_func(
                model,
                allowed_func,
                eos_ids,
                build_prompt(prompt, function_db),
                candidates,
                args,
            )
            if name is None:
                print("Aucune fonction trouvée pour :", prompt)
                continue
            arguments: dict[str, int | float | str] = get_arguments(
                model, prompt, func_by_name[name], eos_ids, value_tokens
            )
            result: dict[str, Any] = {
                "name": name,
                "parameters": arguments,
            }
            print(
                "Function Calling System:",
                json.dumps(result, ensure_ascii=False),
            )
            results.append({"prompt": prompt, **result})

        with open(output_file_path, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=2, ensure_ascii=False)

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
