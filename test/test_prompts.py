from biochatter.prompts import BioCypherPrompts


def test_from_biocypher():
    ps = BioCypherPrompts(schema_config_path="test/test_schema_config.yaml")
    assert list(ps.entities.keys()) == [
        "Protein",
        "Pathway",
        "Gene",
        "Disease",
    ]
    assert list(ps.relationships.keys()) == [
        "PostTranslationalInteraction",
        "Phosphorylation",
        "GeneToDiseaseAssociation",
    ]

    assert "name" in ps.entities.get("Protein").get("properties")
    assert (
        ps.relationships.get("Phosphorylation").get("represented_as") == "edge"
    )
