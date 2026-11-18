import re
import json
from llm_sdk import Small_LLM_Model
from typing import cast


def encode_ids(model: Small_LLM_Model, text: str) -> list[int]:
    return cast(list[int], model.encode(text)[0].tolist())


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
