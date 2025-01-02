import scanpy as sc

from biochatter.api_agent.generate_pydantic_classes_from_module import generate_pydantic_classes

def test_generate_pydantic_classes():
    # Generate the Pydantic classes
    generated_classes = generate_pydantic_classes(sc.tl)

    # just verify that pydantic classes were generated
    assert len(generated_classes) > 0

    # Test that each generated class has the expected Pydantic model properties
    for cls in generated_classes:
        # Check if it has schema method (indicating it's a Pydantic model)
        assert hasattr(cls, "schema")
        schema = cls.schema()

        # Basic schema validation (must contain properties, title, type)
        assert isinstance(schema, dict)
        assert isinstance(schema["properties"], dict)


def test_generate_pydantic_classes_umap():
    # Test a specific function we know should be in sc.tl
    generated_classes = generate_pydantic_classes(sc.tl)

    # Find the umap function
    umap_function = next(
        (cls for cls in generated_classes if cls.schema()["title"] == "umap"),
        None,
    )
    # Verify we found the class
    assert umap_function is not None

    # Check parameters from umap function
    properties = umap_function.schema()["properties"]
    assert len(properties) == 16  # noqa: PLR2004
    assert set(properties.keys()) == {
        "question_uuid",
        "gamma",
        "method",
        "alpha",
        "n_components",
        "a",
        "adata",
        "init_pos",
        "neighbors_key",
        "copy_param",
        "negative_sample_rate",
        "min_dist",
        "maxiter",
        "spread",
        "b",
        "random_state",
    }
