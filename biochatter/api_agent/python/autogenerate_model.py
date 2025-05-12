"""AutoGenerate Pydantic classes for each callable.

This module provides a function to generate Pydantic classes for each callable
(function/method) in a given module. It extracts parameters from docstrings
using docstring-parser and creates Pydantic models with fields corresponding to
the parameters. If a parameter name conflicts with BaseModel attributes, it is
aliased.

Examples
--------
>>> import scanpy as sc
>>> generated_classes = generate_pydantic_classes(sc.tl)
>>> for model in generated_classes:
...     print(model.schema())

"""

import inspect
from types import MappingProxyType, ModuleType
from typing import Any

from docstring_parser import parse
from langchain_core.pydantic_v1 import Field, create_model

from biochatter.api_agent.base.agent_abc import BaseAPIModel


def generate_pydantic_classes(module: ModuleType) -> list[type[BaseAPIModel]]:
    """Generate Pydantic classes for each callable.

    For each callable (function/method) in a given module. Extracts parameters
    from docstrings using docstring-parser. Each generated class has fields
    corresponding to the parameters of the function. If a parameter name
    conflicts with BaseModel attributes, it is aliased.

    Params:
    -------
    module : ModuleType
        The Python module from which to extract functions and generate models.

    Returns
    -------
    list[Type[BaseModel]]
        A list of Pydantic model classes corresponding to each function found in
            `module`.

    Notes
    -----
    - For now, all parameter types are set to `Any` to avoid complications with
      complex or external classes that are not easily JSON-serializable.
    - Optional parameters (those with a None default) are represented as
      `Optional[Any]`.
    - Required parameters (no default) use `...` to indicate that the field is
      required.

    """
    base_attributes = set(dir(BaseAPIModel))
    classes_list = []

    for name, func in inspect.getmembers(module, inspect.isfunction):
        # Skip private/internal functions (e.g., _something)
        if name.startswith("_"):
            continue

        # Parse docstring for parameter descriptions
        doc = inspect.getdoc(func) or ""
        parsed_doc = parse(doc)
        doc_params = {p.arg_name: p.description or "No description available." for p in parsed_doc.params}

        sig = inspect.signature(func)
        fields = {}

        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs for now
            if param_name in ("args", "kwargs"):
                continue

            # Fetch docstring description or fallback
            description = doc_params.get(param_name, "No description available.")

            # Determine default value
            # If no default, we use `...` indicating a required field
            if param.default is not inspect.Parameter.empty:
                default_value = param.default

                # Convert MappingProxyType to a dict for JSON compatibility
                if isinstance(default_value, MappingProxyType):
                    default_value = dict(default_value)

                # Handle non-JSON-compliant float values by converting to string
                if default_value in [float("inf"), float("-inf"), float("nan"), float("-nan")]:
                    default_value = str(default_value)
            else:
                default_value = ...  # No default means required

            # For now, all parameter types are Any
            annotation = Any

            # Append the original annotation as a note in the description if
            # available
            if param.annotation is not inspect.Parameter.empty:
                description += f"\nOriginal type annotation: {param.annotation}"

            # If default_value is None, parameter can be Optional
            # If not required, mark as Optional[Any]
            if default_value is None:
                annotation = Any | None

            # Prepare field kwargs
            field_kwargs = {"description": description, "default": default_value}

            # If field name conflicts with BaseModel attributes, alias it
            field_name = param_name
            if param_name in base_attributes:
                alias_name = param_name + "_param"
                field_kwargs["alias"] = param_name
                field_name = alias_name

            fields[field_name] = (annotation, Field(**field_kwargs))

        # Create the Pydantic model

        tl_parameters_model = create_model(
            name,
            **fields,
            __base__=BaseAPIModel,
        )
        classes_list.append(tl_parameters_model)
    return classes_list
