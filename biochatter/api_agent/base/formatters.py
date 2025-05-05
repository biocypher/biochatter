"""Formatters for API calls (Pydantic models to strings)."""

from urllib.parse import urlencode

from biochatter.api_agent.base.agent_abc import BaseAPIModel, BaseModel
from biochatter.api_agent.python.anndata_agent import MapAnnData


def format_as_rest_call(model: BaseModel) -> str:
    """Convert a parameter model (BaseModel) into a REST API call string.

    Args:
    ----
        model: Pydantic model containing API call parameters

    Returns:
    -------
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
    ----
        model: Pydantic model containing method parameters

    Returns:
    -------
        String representation of the Python method call

    """
    params = model.dict(exclude_none=True)
    method_name = params.pop("method_name", None)
    params.pop("question_uuid", None)
    if isinstance(model, MapAnnData):
        param_str = params.pop("dics", {})
    else:
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())

    return f"{method_name}({param_str})"
