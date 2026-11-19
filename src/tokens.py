"""Token encoding and vocabulary filtering helpers for constrained decoding.

This module provides utilities to encode text into token IDs and build token
sets or mappings used during constrained generation of function names and
argument values.
"""

import re
import json
from typing import cast
from llm_sdk import Small_LLM_Model


def encode_ids(model: Small_LLM_Model, text: str) -> list[int]:
    """Encode a text string into a flat list of token IDs.

    Args:
        model (Small_LLM_Model): Model instance used to
        tokenize the input text.
        text (str): Input string to encode.

    Returns:
        list[int]: Token IDs corresponding to the encoded text.
    """
    return cast(list[int], model.encode(text)[0].tolist())


def build_candidate_tokens(
    model: Small_LLM_Model, allowed_func: list[str]
) -> dict[int, str]:
    """Build a mapping candidate token IDs to token str for allowed functions.

    Reads the model vocabulary and keeps tokens that appear as substrings of at
    least one allowed function name.

    Args:
        model (Small_LLM_Model): Model instance providing access to the
            vocabulary file path.
        allowed_func (list[str]): List of allowed function names.

    Returns:
        dict[int, str]: Dictionary mapping token IDs to their vocabulary
            strings for matching candidate tokens.
    """
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
    """Build token sets and decoded mappings for argument value generation.

    Loads the model vocabulary to identify numeric token IDs, JSON delimiter
    stop token IDs, and decoded text strings for all vocabulary token IDs.

    Args:
        model (Small_LLM_Model): Model instance providing the vocabulary file
            path and token decoding method.

    Returns:
        tuple[set[int], set[int], dict[int, str]]: A tuple containing:
            - ``number_ids`` (set[int]): Token IDs matching numeric characters
              (digits, decimal points, and minus signs).
            - ``stop_ids`` (set[int]): Token IDs for JSON value terminators
              (``,`` and ``}``) when present in the vocabulary.
            - ``string_texts`` (dict[int, str]): Mapping from each token ID to
              its decoded string representation.
    """
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
