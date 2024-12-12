"""Autogenerate Pydantic classes from a module.
This module provides a function to generate Pydantic classes for each callable
(function/method) in a given module. It extracts parameters from docstrings
using docstring-parser and creates Pydantic models with fields corresponding to
the parameters. If a parameter name conflicts with BaseModel attributes, it is
aliased.
"""

import inspect
from typing import Any, Dict, Optional, Type, get_origin, get_args
from types import ModuleType, MappingProxyType
from docstring_parser import parse
from langchain_core.pydantic_v1 import BaseModel, Field, create_model

def _is_optional_type(t):
    """Check if a given type annotation is Optional[...] or Any."""
    if t is Any:
        return True
    origin = get_origin(t)
    if origin is getattr(__import__('typing'), 'Union', None):
        args = get_args(t)
        return type(None) in args
    return False

def generate_pydantic_classes(module: ModuleType) -> list[Type[BaseModel]]:
    """
    Generate Pydantic classes for each public callable (function/method) in a given module.
    
    Parameters
    ----------
    module : ModuleType
        The Python module from which to extract functions and generate models.
        
    Returns
    -------
    list[Type[BaseModel]]
        A list of Pydantic model classes corresponding to each function found in `module`.
        
    Notes
    -----
    - For now, all parameter types are set to `Any` to avoid complications with complex or
      external classes that are not easily JSON-serializable.
    - Optional parameters (those with a None default) are represented as `Optional[Any]`.
    - Required parameters (no default) use `...` to indicate that the field is required.
    """
    base_attributes = set(dir(BaseModel))
    classes_list = []

    for name, func in inspect.getmembers(module, inspect.isfunction):
        # Skip private/internal functions (e.g., _something)
        if name.startswith("_"):
            continue

        # Parse docstring for parameter descriptions
        doc = inspect.getdoc(func) or ""
        parsed_doc = parse(doc)
        doc_params = {p.arg_name: p.description or "No description available."
                      for p in parsed_doc.params}

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
                if default_value in [float('inf'), float('-inf'), float('nan'), float('-nan')]:
                    default_value = str(default_value)
            else:
                default_value = ...  # No default means required

            # For now, all parameter types are Any
            annotation = Any

            # Append the original annotation as a note in the description if available
            if param.annotation is not inspect.Parameter.empty:
                description += f"\nOriginal type annotation: {param.annotation}"

            # If default_value is None, parameter can be Optional
            # If not required, mark as Optional[Any]
            if default_value is None:
                annotation = Optional[Any]

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
        TLParametersModel = create_model(
            name,
            **fields)
        classes_list.append(TLParametersModel)

    return classes_list


# Example usage:
#import scanpy as sc
#generated_classes = generate_pydantic_classes(sc.tl)
#for func in generated_classes:  
#    print(func.schema())
