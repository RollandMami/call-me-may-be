"""Constrained decoding routines for func selection and argument extraction.

This module implements logit-masking helpers to steer a language model toward
valid function names and typed JSON argument values during generation.
"""

import re
import json
import argparse
import numpy as np
from typing import Any
from .tokens import encode_ids
from numpy.typing import NDArray
from llm_sdk import Small_LLM_Model
from .prompts import build_args_prompt

MAX_TOKENS = 100
"""Maximum number of decoding steps allowed per function or value."""

NUM_PREFIX = {
    "number": re.compile(r"-?(\d+(\.\d*)?)?"),
    "integer": re.compile(r"-?\d*"),
}
"""Compiled regular expressions matching partial numeric argument prefixes."""

NUM_FULL = {
    "number": re.compile(r"-?\d+(\.\d+)?"),
    "integer": re.compile(r"-?\d+"),
}
"""Compiled regular expressions matching complete numeric argument values."""


def find_end_quote(s: str) -> int:
    """Find the index of the first unescaped double quote in a string.

    Args:
        s (str): String to scan for a closing double quote.

    Returns:
        int: Index of the first unescaped ``"`` character, or ``-1`` if none
            is found.
    """
    i = 0
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == '"':
            return i
        i += 1
    return -1


def decode_json_string(raw: str) -> str:
    """Unescape raw JSON string content when possible.

    Wraps ``raw`` in double quotes and parses it as JSON to decode escape
    sequences, falling back to the original string if decoding fails.

    Args:
        raw (str): Raw string fragment without surrounding quotes.

    Returns:
        str: Decoded string value, or ``raw`` if ``json.loads`` raises
            ``json.JSONDecodeError``.
    """
    try:
        return str(json.loads(f'"{raw}"'))
    except json.JSONDecodeError:
        return raw


def get_func(
    model: Small_LLM_Model,
    allowed_func: list[str],
    eos_ids: set[int],
    prompt: str,
    candidates: dict[int, str],
    args: argparse.Namespace,
) -> str | None:
    """Decode an allowed function name using prefix-constrained logit masking.

    Iteratively masks logits to tokens that extend ``current_prefix`` toward at
    least one name in ``allowed_func``, stopping early once a unique match is
    reached or when an end-of-sequence token is selected.

    Args:
        model (Small_LLM_Model): Language model used to compute next-token
            logits.
        allowed_func (list[str]): List of valid function names that may be
            selected.
        eos_ids (set[int]): Set of end-of-sequence token IDs.
        prompt (str): Formatted prompt instructing the model to output a
            function name.
        candidates (dict[int, str]): Mapping from candidate token IDs to their
            decoded string pieces.
        args (argparse.Namespace): Parsed CLI options; prints step-by-step
            decoding trace when ``args.debug`` is truthy.

    Returns:
        str | None: The matched function name if it belongs to
            ``allowed_func``, otherwise ``None``.
    """
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
    """Generate a single argument value while updating token IDs in place.

    Applies numeric prefix/completion constraints when ``kind`` is ``"number"``
    or ``"integer"``, and single-line string constraints terminated by an
    unescaped quote otherwise. Appends generated token IDs to ``input_ids``.

    Args:
        model (Small_LLM_Model): Language model used to predict next-token
            logits.
        input_ids (list[int]): Mutable sequence of token IDs updated as tokens
            are accepted.
        kind (str): Expected argument type name (such as ``"integer"``,
            ``"number"``, or ``"string"``).
        eos_ids (set[int]): Set of end-of-sequence token IDs.
        number_ids (set[int]): Set of token IDs allowed in numeric literals.
        stop_ids (set[int]): Set of JSON delimiter token IDs that terminate a
            numeric value.
        string_texts (dict[int, str]): Mapping from token IDs to their text
            representations.

    Returns:
        str: Raw generated text for the argument value before final type
            conversion or string unescaping.
    """
    text = ""
    for _ in range(MAX_TOKENS):
        logits: NDArray[np.float64] = np.array(
            model.get_logits_from_input_ids(input_ids), dtype=float
        )
        mask: NDArray[np.float64] = np.full_like(logits, -np.inf)

        if kind in ("number", "integer"):
            allowed = {
                tid for tid in number_ids
                if NUM_PREFIX[kind].fullmatch(text + string_texts[tid])
            }
            if NUM_FULL[kind].fullmatch(text):
                allowed |= stop_ids | eos_ids
        else:
            allowed = {
                tid for tid, t in string_texts.items()
                if "\n" not in t and "\r" not in t
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
            text += string_texts[next_id]
            end = find_end_quote(text)
            piece = string_texts[next_id]
            if end >= 0:
                kept = text[:end]
                tail = kept[
                    len(text) - len(piece):
                ] if len(kept) > len(text) - len(piece) else ""
                if tail:
                    input_ids += encode_ids(model, tail)
                text = kept
                break
            input_ids.append(next_id)
    return text


def get_arguments(
    model: Small_LLM_Model,
    prompt: str,
    func: dict[str, Any],
    eos_ids: set[int],
    value_tokens: tuple[set[int], set[int], dict[int, str]],
) -> dict[str, int | float | str]:
    """Extract and cast all arguments for a selected function schema.

    Builds the argument extraction prompt for ``func``, injects key and quote
    delimiters for each parameter in order, generates raw values via
    ``generate_value``, and converts each value to its declared Python type.

    Args:
        model (Small_LLM_Model): Language model used for constrained token
            generation.
        prompt (str): Original natural language user request.
        func (dict[str, Any]): Function schema dictionary containing ``name``
            and ``parameters`` specifications.
        eos_ids (set[int]): Set of end-of-sequence token IDs.
        value_tokens (tuple[set[int], set[int], dict[int, str]]): Tuple of
            ``(number_ids, stop_ids, string_texts)`` used during value
            generation.

    Returns:
        dict[str, int | float | str]: Mapping from parameter names to their
            parsed Python values.
    """
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
            args_out[key] = decode_json_string(raw)
            input_ids += encode_ids(model, '"')
        elif kind == "integer":
            args_out[key] = int(raw)
        else:
            args_out[key] = float(raw) if "." in raw else int(raw)
    return args_out
