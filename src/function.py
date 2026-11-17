import json
import re
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    fieldValidator,
    model_validator,
)


class Parameters(BaseModel):
    kname: str
    types: str


class FunctionModel(BaseModel):
    name: str
    description: str
    parameters: list[Parameters]
    returns: str
