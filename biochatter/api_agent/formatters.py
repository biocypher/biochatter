"""Formatters for API calls (Pydantic models to strings)."""

from urllib.parse import urlencode

from .abc import BaseAPIModel, BaseModel


def format_as_rest_call(model: BaseModel) -> str:
    """Convert a parameter model (BaseModel) into a REST API call string.

    Args:
        model: Pydantic model containing API call parameters

    Returns:
        String representation of the REST API call

    """
    params = model.dict(exclude_none=True)
    endpoint = params.pop("endpoint")
    base_url = params.pop("base_url")
    params.pop("question_uuid", None)

    full_url = f"{base_url.rstrip('/')}/{endpoint.strip('/')}"
    return f"{full_url}?{urlencode(params)}"


def format_as_python_call(model: BaseAPIModel) -> str:
    """Convert a parameter model into a Python method call string.

    Args:
        model: Pydantic model containing method parameters

    Returns:
        String representation of the Python method call

    """
    model_class = model.model_json_schema()
    params = model_class["properties"]
    method_name = model_class["title"]
    params.pop("question_uuid", None)
    # Before it was specifically for map anndata function
        # Generate parameter string
    param_str_list = []
    for param_name, param_details in params.items():
        if "anyOf" in param_details:
            # Handle `anyOf` by using a placeholder or default
            param_str_list.append(f"{param_name}=<choose_from_anyOf>")
        else:
            # Use the title as the value placeholder
            param_str_list.append(f"{param_name}=<value_for_{param_details.get('title', param_name)}>")

    param_str = ", ".join(param_str_list)
    #param_str = params.pop("dics", {}) if "dics" in params else ", ".join(f"{k}={v!r}" for k, v in params.items())

    return f"{method_name}({param_str})"
