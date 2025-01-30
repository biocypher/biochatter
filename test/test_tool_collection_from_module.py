import scanpy as sc

from biochatter.api_agent.collect_tool_info_from_module import collect_tool_info

def test_collect_tool_info():
    # Generate the tool information
    tools_params, tools_descriptions = collect_tool_info(sc.tl)

    # Verify that both are dictionaries
    assert isinstance(tools_params, dict)
    assert isinstance(tools_descriptions, dict)

    # Verify that both dictionaries have the same keys
    assert tools_params.keys() == tools_descriptions.keys()

    # Verify that each entry in tools_params is a dictionary of parameters
    for params in tools_params.values():
        assert isinstance(params, dict)

    # Verify that each entry in tools_descriptions is a string description
    for description in tools_descriptions.values():
        assert isinstance(description, str)


def test_collect_tool_info_umap():
    # Test a specific function we know should be in sc.tl
    tools_params, tools_descriptions = collect_tool_info(sc.tl)

    # Verify that 'umap' is one of the keys
    assert 'umap' in tools_params
    assert 'umap' in tools_descriptions

    # Check parameters from umap function
    umap_params = tools_params['umap']
    assert isinstance(umap_params, dict)
    assert len(umap_params) > 0  # Ensure there are parameters

    # Check description from umap function
    umap_description = tools_descriptions['umap']
    assert isinstance(umap_description, str)
    assert len(umap_description) > 0  # Ensure there is a description
