"""Prompt builders for function selection and argument extraction.

This module provides helpers to format ChatML prompts for identifying a target
function from a catalog and extracting its arguments from a user request.
"""

from typing import Any


def build_prompt(user_prompt: str, function_db: list[dict[str, Any]]) -> str:
    """Build a ChatML prompt for selecting the matching function name.

    Formats each available function signature and description into the system
    message and instructs the model to reply strictly with the function name
    or ``'Null'`` when no function matches.

    Args:
        user_prompt (str): Natural language request from the user.
        function_db (list[dict[str, Any]]): List of function schema
            dictionaries containing ``name``, ``description``, and optional
            ``parameters``.

    Returns:
        str: Formatted ChatML prompt string ready for model encoding.
    """
    lines: list[str] = []
    for f in function_db:
        params = ", ".join(
            f"{name}: {p.get('type', 'unknown')}"
            for name, p in f.get("parameters", {}).items()
        )
        lines.append(f"- {f['name']}({params}): {f.get('description', '')}")
    funcs = "\n".join(lines)
    prompt = (
        f"<|im_start|>system\nYou are an assistant that must identify the "
        f"name of a function from a given prompt. You must reply only with "
        f"the function name, without explanations or comments. If no "
        f"function matches, reply with 'Null'.\n\nAvailable functions:\n"
        f"{funcs}"
        f"<|im_end|>\n<|im_start|>user\n"
        f"{user_prompt}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
    return prompt


def build_args_prompt(user_prompt: str, func: dict[str, Any]) -> str:
    """Build a ChatML prompt with few-shot examples for argument extraction.

    Constructs a prompt with English few-shot examples and a partial JSON
    assistant prefix so the model completes the args dictionary for ``func``.

    Args:
        user_prompt (str): Natural language request from the user.
        func (dict[str, Any]): Selected function schema dictionary containing
            ``name``, ``description``, and optional ``parameters``.

    Returns:
        str: Formatted ChatML prompt string ending at the opening of the
            ``arguments`` JSON object.
    """
    params = ", ".join(
        f"{k}: {v.get('type', 'unknown')}"
        for k, v in func.get("parameters", {}).items()
    )

    examples = (
        "Examples:\n"
        "Request: Add 2 and 3\n"
        '{"function": "fn_add_numbers", "arguments": {"a": 2, "b": 3}}\n'
        'Request: Replace digits with # in "abc123"\n'
        '{"function": "fn_substitute_string_with_regex", "arguments": '
        '{"source_string": "abc123", "regex": "\\\\d+", '
        '"replacement": "#"}}\n\n'
    )
    return (
        "<|im_start|>system\n"
        "You extract argument values for a function from the user's request. "
        "Reply only in JSON.\n\n"
        f"{examples}"
        f"Function: {func['name']}({params}): "
        f"{func.get('description', '')}"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f'{{"function": "{func["name"]}", "arguments": {{'
    )
