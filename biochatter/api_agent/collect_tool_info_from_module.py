"""AutoGenerate Pydantic classes for each callable.

This module provides a function to generate Pydantic classes for each callable
(function/method) in a given module. It extracts parameters from docstrings
using docstring-parser and creates Pydantic models with fields corresponding to
the parameters. If a parameter name conflicts with BaseModel attributes, it is
aliased.

Examples
--------
>>> import scanpy as sc
>>> generated_classes = collect_tool_info(sc.tl)
>>> for model in generated_classes:
...     print(model.schema())

"""

import inspect
from types import MappingProxyType, ModuleType, UnionType
from typing import Any, Union, get_origin, get_args, ForwardRef
from typing import _eval_type  # For evaluating string type annotations

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model

from biochatter.api_agent.abc import BaseAPIModel

def collect_tool_info(
    module: ModuleType, 
    preserve_all_types: bool = False
) -> tuple[dict[str, dict], dict[str, str]]:
    """Generate parameter definitions and descriptions for each callable.

    For each callable (function/method) in a given module. Extracts parameters
    from docstrings using docstring-parser. Each function's parameters are stored
    with their types and Field definitions, matching the manual approach used in agents such as
    scanpy_pl.py.

    Parameters
    ----------
    module : ModuleType
        The Python module from which to extract functions and generate parameter definitions.
    preserve_all_types : bool, default=False
        If True, keeps all type annotations as-is without converting any to Any.
        If False, preserves standard types and typing constructs but converts external class types to Any.

    Returns
    -------
    tuple[dict, dict]
        A tuple containing:
        - tools_params: Dictionary mapping function names to their parameter definitions
        - tools_descriptions: Dictionary mapping function names to their descriptions

    Notes
    -----
    - Standard types (int, str, float, bool, etc.) are preserved
    - Typing constructs (Union, Optional, List, etc.) are preserved
    - By default, only external class types are converted to Any
    - Set preserve_all_types=True to keep all type annotations unchanged
    """
    def get_function_fields(func) -> tuple[dict, str]:
        """Extract fields and description from a function.
        
        Parameters
        ----------
        func : Callable
            The function to analyze
            
        Returns
        -------
        tuple[dict, str]
            A tuple containing:
            - Dictionary of fields with their types and Field definitions
            - Function description from docstring
        """
        # Parse docstring for parameter descriptions
        doc = inspect.getdoc(func) or ""
        parsed_doc = parse(doc)
        doc_params = {p.arg_name: p.description or "No description available." 
                     for p in parsed_doc.params}
        function_description = parsed_doc.description or "No description"
        sig = inspect.signature(func)
        fields = {}

        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs for now
            if param_name in ("args", "kwargs"):
                continue

            # Fetch docstring description or fallback
            description = doc_params.get(param_name, "No description available.")

            # Determine default value
            if param.default is not inspect.Parameter.empty:
                default_value = param.default

                # Convert MappingProxyType to a dict for JSON compatibility
                if isinstance(default_value, MappingProxyType):
                    default_value = dict(default_value)

                # Handle non-JSON-compliant float values by converting to string
                if default_value in [float("inf"), float("-inf"), float("nan"), float("-nan")]:
                    default_value = str(default_value)
            else:
                default_value = None  # No default means required

            # Get the original annotation
            original_annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any

            # Keep standard types and typing constructs, convert only external class types to Any
            type_info = is_standard_type(original_annotation)
            if type_info:
                if isinstance(type_info, tuple):
                    # Handle special cases like the array format
                    annotation, json_schema_extra = type_info
                    # Create Field with schema override
                    field = Field(
                        default=default_value,
                        description=description
                        #json_schema_extra=lambda _: json_schema_extra
                    )
                else:
                    annotation = original_annotation
                    field = Field(
                        default=default_value,
                        description=description
                    )
            else:
                # Convert only external class types to Any
                annotation = Any
                # Add original type to description
                if param.annotation is not inspect.Parameter.empty:
                    description += f"\nOriginal type annotation: {param.annotation}"
                field = Field(
                    default=default_value,
                    description=description
                )

            # If field name conflicts with BaseModel attributes, alias it
            field_name = param_name
            if param_name in base_attributes:
                alias_name = param_name + "_param"
                field = Field(
                    default=field.default,
                    description=field.description,
                    alias=param_name
                )
                field_name = alias_name

            fields[field_name] = (annotation, field)

        return fields, function_description

    def is_standard_type(typ) -> bool:
        """Check if a type is a standard type or typing construct."""
        if preserve_all_types:
            return True
            
        standard_types = (int, str, float, bool, list, dict, tuple, set)
        
        # Handle basic types first
        if isinstance(typ, type) and issubclass(typ, standard_types):
            return True
            
        # Handle None type
        if typ is type(None):
            return True

        # Special handling for tuple[int, int] (like dimensions)
        origin = get_origin(typ)
        if origin is tuple:
            args = get_args(typ)
            if len(args) == 2 and all(arg is int for arg in args):
                return (list, {
                    "type": "array", 
                    "minItems": 2, 
                    "maxItems": 2, 
                    "items": {"type": "integer"}
                })

        # Handle Sequence[tuple[int, int]] (like dimensions parameter)
        if origin in (list, tuple) or (isinstance(typ, type) and issubclass(typ, (list, tuple))):
            args = get_args(typ)
            if len(args) == 1 and get_origin(args[0]) is tuple:
                tuple_args = get_args(args[0])
                if len(tuple_args) == 2 and all(arg is int for arg in tuple_args):
                    return (list, {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "integer"}
                        }
                    })

        # Handle Union types
        if hasattr(typ, '__args__'):
            args = typ.__args__
            # Handle Optional[type] specially
            if len(args) == 2 and args[1] is type(None):
                base_type = is_standard_type(args[0])
                if isinstance(base_type, tuple):
                    # For special types like the array format above
                    base_type_schema = base_type[1]
                    return (base_type[0], {
                        "anyOf": [
                            base_type_schema,
                            {"type": "null"}
                        ]
                    })
                return base_type
            # For other unions, collect all valid schemas
            valid_types = []
            for arg in args:
                arg_type = is_standard_type(arg)
                if isinstance(arg_type, tuple):
                    valid_types.append(arg_type[1])
                elif arg_type:
                    if arg is type(None):
                        valid_types.append({"type": "null"})
                    else:
                        valid_types.append({"type": str(arg.__name__).lower()})
            if valid_types:
                return (Union[args], {"anyOf": valid_types})

        # Handle string type annotations
        if isinstance(typ, str):
            try:
                fr = ForwardRef(typ)
                try:
                    typ = fr._evaluate(globals(), module.__dict__, recursive_guard=set())
                except Exception:
                    # If evaluation fails, try parsing the string directly
                    if ' | ' in typ:
                        types = [t.strip() for t in typ.split('|')]
                        return all(is_standard_type(t) for t in types)
                    return typ in ('str', 'int', 'float', 'bool', 'None')
            except Exception:
                return False

        # Check for Union types by looking for __args__
        if hasattr(typ, '__args__'):
            args = typ.__args__
            # Handle Optional[type] specially
            if len(args) == 2 and args[1] is type(None):
                base_type = is_standard_type(args[0])
                if isinstance(base_type, tuple):
                    # For special types like the array format above
                    return base_type
                return base_type
            # For other unions, all types must be standard
            return all(is_standard_type(arg) for arg in args)
            
        # Get the base type for other typing constructs
        if origin is None:
            return False
            
        # Handle other typing constructs (List, etc.)
        if origin in (list, dict, set):
            args = get_args(typ)
            return all(is_standard_type(arg) for arg in args)
            
        return False
    base_attributes = set(dir(BaseAPIModel))
    tools_params = {}
    tools_descriptions = {}
    # Check if module is actually a function
    if inspect.isfunction(module):
        fields, desc = get_function_fields(module)
        name = module.__name__
        tools_params[name] = fields
        tools_descriptions[name] = desc
        return tools_params, tools_descriptions

    # Process all functions in module
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # Skip private/internal functions
        if name.startswith("_"):
            continue

        fields, desc = get_function_fields(func)
        tools_params[name] = fields
        tools_descriptions[name] = desc

    return tools_params, tools_descriptions
