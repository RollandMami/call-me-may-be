import argparse
import numpy as np
from typing import Any
from .tokens import encode_ids
from numpy.typing import NDArray
from llm_sdk import Small_LLM_Model
from .prompts import build_args_prompt

MAX_TOKENS = 100


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
