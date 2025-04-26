from typing import Any

from fastmcp import FastMCP
from tooluniverse import ToolUniverse


def extract_optional_params(func_defs: list[dict[str, Any]]) -> list[str]:
    """Scan each function definition in `func_defs` and return a list of
    all parameter names that are marked as optional (required=False).
    """
    optional = []
    for meta in func_defs:
        props = meta.get("parameter", {}).get("properties", {})
        for pname, pinfo in props.items():
            # If 'required' is not True, treat as optional
            if not pinfo.get("required", False):
                optional.append(pname)
    return optional


def execute_api_call(func_name: str, params: dict[str, Any]) -> Any:
    """Dispatch to the real implementation by func_name.
    """
    try:

        #extract optional params
        optional_params = set(extract_optional_params(tooluni.get_tool_by_name([func_name])))
        #remove optional params from param if not specified
        # Remove optional parameters from params if their value is None
        for opt in optional_params:
            if opt in params and params[opt] is None:
                del params[opt]
        query = {"name": func_name, "arguments": params}
        return tooluni.run(query)
    except Exception:
        raise NotImplementedError(f"execute_api_call not implemented for {func_name!r}")

def generate_api_functions(func_defs: list[dict[str, Any]], namespace: dict[str, Any]) -> None:
    """Dynamically defines stub functions based on metadata dictionaries.
    Return types are emitted as quoted forward references to avoid NameError.
    """
    for meta in func_defs:
        name        = meta["name"]
        desc        = meta.get("description", "").strip()
        props       = meta["parameter"]["properties"]
        #return_type = str#meta.get('type', 'Any')

        # Build signature parts, doc-params and body-params
        sig_parts, doc_params, body_params = [], [], []
        for pname, pinfo in props.items():
            ptype    = pinfo.get("type", "Any")
            required = pinfo.get("required", False)
            py_type = {
                "string":  "str",
                "integer": "int",
                "number":  "float",
                "boolean": "bool",
                "object":  "dict",
                "array":   "list"
            }.get(ptype, "Any")

            if required:
                sig_parts.append(f"{pname}: {py_type}")
            else:
                sig_parts.append(f"{pname}: Optional[{py_type}] = None")

            doc_params.append(f"{pname} ({py_type}{'' if required else ', optional'}): {pinfo.get('description','')}")
            body_params.append(f'"{pname}": {pname}')

        signature = ", ".join(sig_parts)
        # Quote the return type
        #return_annotation = f"\"{return_type}\""

        # Build an indented docstring block
        doc_lines = ['    """']
        if desc:
            doc_lines += [f"    {line}" for line in desc.splitlines()]
            doc_lines.append("    ")
        doc_lines.append("    Parameters:")
        for dp in doc_params:
            doc_lines.append(f"    - {dp}")
        doc_lines.append("    ")
        doc_lines.append("    Returns:")
        #doc_lines.append(f"    - {return_type}")
        doc_lines.append('    """')
        docstring = "\n".join(doc_lines)

        # Build the function body (indented)
        body_lines = [
            "    return execute_api_call(",
            f"        {name!r},",
            "        {"
        ]
        body_lines += [f"            {bp}," for bp in body_params]
        body_lines += [
            "        }",
            "    )"
        ]
        body = "\n".join(body_lines)

        # Final source text
        #func_src = f"def {name}({signature}) -> {return_annotation}:\n{docstring}\n{body}\n"
        func_src = f"@mcp.tool()\ndef {name}({signature}):\n{docstring}\n{body}\n"
        #func_src = f"def {name}({signature}):\n{docstring}\n{body}\n"
        exec(func_src, namespace)
        fn = namespace[name]
        fn.name = name
        fn.description = desc
        fn.tool_call_schema = tooluni.get_tool_by_name([name])[0]["parameter"]

#tool definition
tooluni = ToolUniverse()
tooluni.load_tools()
mcp = FastMCP("Tool Universe")

for tool in ["get_target_id_description_by_name","get_drug_names_by_indication","get_risk_info_by_drug_name"]:
    generate_api_functions([tooluni.get_one_tool_by_one_name(tool)], globals())


if __name__ == "__main__":
    mcp.run(transport="stdio")
