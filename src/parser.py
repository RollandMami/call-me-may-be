"""Parse and validate command-line arguments for function calling tests.

This module defines CLI options for configuring input/output file paths and
enabling debug mode, then verifies that required input files exist before
execution.
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Configures the argument parser to retrieve paths for function definitions,
    test inputs, and test outputs, as well as the debug flag.

    Returns:
        argparse.Namespace: Namespace containing the parsed argument values:
            - ``functions_definition`` (str): Path to the function definitions
              JSON file.
            - ``input`` (str): Path to the input test cases JSON file.
            - ``output`` (str): Path to the output JSON file.
            - ``debug`` (bool): Whether debug mode is enabled.
    """
    parser = argparse.ArgumentParser(description="Test de function calling")

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions file",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON tests file",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    """Verify that required input files exist.

    Args:
        args (argparse.Namespace): Namespace returned by ``parse_args()``
            containing ``functions_definition`` and ``input`` attributes.

    Raises:
        FileNotFoundError: If the function definitions file or the input JSON
            file does not exist on disk.
    """
    if not Path(args.functions_definition).exists():
        raise FileNotFoundError("Function definitions file not found")
    if not Path(args.input).exists():
        raise FileNotFoundError("Input file not found")
