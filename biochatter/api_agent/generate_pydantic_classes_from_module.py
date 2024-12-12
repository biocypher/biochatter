"""Autogenerate Pydantic classes from a module.

This module provides a function to generate Pydantic classes for each callable
(function/method) in a given module. It extracts parameters from docstrings
using docstring-parser and creates Pydantic models with fields corresponding to
the parameters. If a parameter name conflicts with BaseModel attributes, it is
aliased.
"""

import inspect
from types import ModuleType

from docstring_parser import parse
from langchain_core.pydantic_v1 import BaseModel, Field, create_model

def generate_pydantic_classes(module: ModuleType) -> list[type[BaseModel]]:
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
    list[type[BaseModel]]
        A list of Pydantic model classes.

    """
    base_attributes = set(dir(BaseModel))
    classes_list = []

    # Iterate over all callables in the module
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # skip if method starts with _
        if name.startswith("_"):
            continue
        doc = inspect.getdoc(func)
        if not doc:
            # If no docstring, still create a model with no fields
            tl_parameters_model = create_model(f"{name}")
            classes_list.append(tl_parameters_model)
            continue

        parsed_doc = parse(doc)

        # Collect parameter descriptions
        param_info = {}
        for p in parsed_doc.params:
            if p.arg_name not in param_info:
                param_info[p.arg_name] = p.description or "No description available."

        # Prepare fields for create_model
        fields = {}
        alias_map = {}

        for param_name, param_desc in param_info.items():
            field_kwargs = {"default": None, "description": param_desc}
            field_name = param_name

            # Alias if conflicts with BaseModel attributes
            if param_name in base_attributes:
                aliased_name = param_name + "_param"
                field_kwargs["alias"] = param_name
                alias_map[aliased_name] = param_name
                field_name = aliased_name

            # Without type info, default to Optional[str]
            fields[field_name] = (str | None, Field(**field_kwargs))

        # Dynamically create the model for this function
        tl_parameters_model = create_model(name, **fields)
        classes_list.append(tl_parameters_model)

    return classes_list


# Example usage:
# import scanpy as sc
# generated_classes = generate_pydantic_classes(sc.tl)
# for func in generated_classes:
#     print(func.schema())
