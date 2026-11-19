"""Schema models and loader for function calling definitions.

This module defines Pydantic models to validate function schemas, parameter
names, return types, and catalog uniqueness, along with a JSON file loader
that formats validation and I/O errors into clear messages.
"""

import re
import json
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

TypeName = Literal["number", "integer", "string", "boolean"]
IDENT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class TypeSpec(BaseModel):
    """Immutable specification of a parameter or return value type.

    Attributes:
        type (TypeName): Supported primitive type name for the value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    type: TypeName


class FunctionModel(BaseModel):
    """Immutable schema definition for a callable function.

    Attributes:
        name (str): Non-empty valid identifier naming the function.
        description (str): Non-empty description of what the function does.
        parameters (dict[str, TypeSpec]): Mapping of valid parameter names to
            their type specifications.
        returns (TypeSpec): Type specification of the function return value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, TypeSpec]
    returns: TypeSpec

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that the function name is a valid identifier.

        Args:
            v (str): Candidate function name.

        Returns:
            str: Validated function name.

        Raises:
            ValueError: If ``v`` does not match a valid identifier pattern.
        """

        if not IDENT.match(v):
            raise ValueError(f"Invalid function name: {v}")
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls, v: dict[str, TypeSpec]
    ) -> dict[str, TypeSpec]:
        """Validate that all parameter keys are valid identifiers.

        Args:
            v (dict[str, TypeSpec]): Mapping of parameter names to type specs.

        Returns:
            dict[str, TypeSpec]: Validated parameter dictionary.

        Raises:
            ValueError: If any parameter name is not a valid identifier.
        """

        for key in v:
            if not IDENT.match(key):
                raise ValueError(f"Invalid parameter name: {key!r}")
        return v


class FunctionCatalog(BaseModel):
    """Collection of validated function definitions with unique names.

    Attributes:
        functions (list[FunctionModel]): Non-empty list of function schemas.
    """

    functions: list[FunctionModel] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self) -> "FunctionCatalog":
        """Ensure no two functions in the catalog share the same name.

        Returns:
            FunctionCatalog: The validated catalog instance.

        Raises:
            ValueError: If duplicate function names are found.
        """
        names = [f.name for f in self.functions]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicated functions names: {sorted(dupes)}")
        return self


def load_functions(path: str) -> list[FunctionModel]:
    """Load and validate function definitions from a JSON file.

    Reads a JSON list of function schema objects from ``path`` and validates
    each entry and catalog-level uniqueness using ``FunctionCatalog``.

    Args:
        path (str): Path to the JSON file containing function definitions.

    Returns:
        list[FunctionModel]: List of validated function schema models.

    Raises:
        ValueError: If the file is missing, unreadable, contains invalid JSON,
            does not have a top-level list, or fails schema validation.
    """
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Bad Json format in {path} "
            f"(row {e.lineno}, colomns {e.colno}): {e.msg}"
        ) from None
    except OSError as e:
        raise ValueError(f"Read error {path}: {e}") from None

    if not isinstance(raw, list):
        raise ValueError(f"{path}: root must be a list")

    try:
        return FunctionCatalog(functions=raw).functions
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"][1:])
            lines.append(f"  [{loc}] {err['msg']}")
        raise ValueError(
            f"Invalid definitions in {path}:\n" + "\n".join(lines)
        ) from None
