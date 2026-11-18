import json
import re
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
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: TypeName


class FunctionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, TypeSpec]
    returns: TypeSpec

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not IDENT.match(v):
            raise ValueError(f"Invalid function name: {v}")
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls, v: dict[str, TypeSpec]
    ) -> dict[str, TypeSpec]:
        for key in v:
            if not IDENT.match(key):
                raise ValueError(f"Invalid parameter name: {key!r}")
        return v


class FunctionCatalog(BaseModel):
    functions: list[FunctionModel] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self) -> "FunctionCatalog":
        names = [f.name for f in self.functions]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"noms de fonctions dupliqués: {sorted(dupes)}")
        return self


def load_functions(path: str) -> list[FunctionModel]:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Fichier introuvable: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON invalide dans {path} "
            f"(ligne {e.lineno}, colonne {e.colno}): {e.msg}"
        ) from None
    except OSError as e:
        raise ValueError(f"Lecture impossible de {path}: {e}") from None

    if not isinstance(raw, list):
        raise ValueError(f"{path}: la racine doit être une liste")

    try:
        return FunctionCatalog(functions=raw).functions
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"][1:])
            lines.append(f"  [{loc}] {err['msg']}")
        raise ValueError(
            f"Définitions invalides dans {path}:\n" + "\n".join(lines)
        ) from None
