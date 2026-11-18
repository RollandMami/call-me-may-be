import json
import traceback
from pathlib import Path
from typing import Any
from llm_sdk import Small_LLM_Model


from .prompts import build_prompt
from .validation import load_functions
from .parser import parse_args, check_args
from .answer import get_func, get_arguments
from .tokens import (build_candidate_tokens, build_value_tokens,
                     encode_ids)

try:
    from dotenv import load_dotenv
except (ImportError, AttributeError):
    print("Importation error: python-dotenv n'est pas installé")
    raise


def main() -> None:
    load_dotenv()
    args = parse_args()
    try:
        check_args(args)
        model: Small_LLM_Model = Small_LLM_Model()
        eos_ids: set[int] = set(encode_ids(model, "<|im_end|>"))
        functions = load_functions(args.functions_definition)
        prompt_path = args.input
        output_file_path = Path(args.output)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        function_db: list[dict[str, Any]] = [f.model_dump() for f in functions]
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
    except (FileNotFoundError, ValueError) as e:
        print(e)
    except KeyboardInterrupt as k:
        print("interuption keyboard", k)
    except Exception as e:
        print(e)
        traceback.print_exc()
