
from typing import Any


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

    examples = (
        "Exemples :\n"
        'Demande : Additionne 2 et 3\n'
        '{"function": "fn_add_numbers", "arguments": {"a": 2, "b": 3}}\n'
        'Demande : Remplace les chiffres par # dans "abc123"\n'
        '{"function": "fn_substitute_string_with_regex", "arguments": '
        '{"source_string": "abc123", "regex": "\\\\d+", '
        '"replacement": "#"}}\n\n'
        'Demande : Replace all spaces in "a b c" with underscores\n'
        '{"function": "fn_substitute_string_with_regex", "arguments": '
        '{"source_string": "a b c", "regex": " ", "replacement": "_"}}\n'
    )
    return (
        "<|im_start|>system\n"
        "Tu extrais les valeurs des arguments d'une fonction à partir de la "
        "demande de l'utilisateur. Réponds uniquement en JSON.\n\n"
        f"{examples}"
        f"Fonction : {func['name']}({params}) : "
        f"{func.get('description', '')}"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f'{{"function": "{func["name"]}", "arguments": {{'
    )
