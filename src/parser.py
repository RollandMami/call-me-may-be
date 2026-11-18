import argparse
from pathlib import Path


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
