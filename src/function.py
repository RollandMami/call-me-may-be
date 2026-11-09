try:
    from pydantic import ( # type: ignore
        BaseModel,
        # Field,
        # model_validator,
        # field_validator,
        )
except (ImportError, ModuleNotFoundError) as e:
    raise ModuleNotFoundError(f"{type(e).__name__}: {e}")


class Parameters(BaseModel):
    kname: str
    types: str


class FunctionModel(BaseModel):
    name: str
    description: str
    parameters: list[Parameters]
    returns: str
