from unittest import mock

from biochatter.kg_grounding import ground_entities, KGAdapter


class FakeAdapter(KGAdapter):
    """Stands in for Neo4jAdapter — no real database needed."""

    def __init__(self, values_by_type: dict[str, list[str]]):
        self.values_by_type = values_by_type

    def get_all_values(self, entity_type, search_property):
        return self.values_by_type.get(entity_type, [])

    def get_node_value_by_id_substring(self, entity_type, search_property, ontology_id):
        return None  # not exercised by this test


def fake_conversation_factory():
    """Stands in for a real LLM — returns a canned expansion."""
    conv = mock.Mock()
    conv.append_system_message = mock.Mock()
    conv.query = mock.Mock(return_value=("atrial fibrillation", None, None))
    return conv


def test_ground_entities_resolves_abbreviation_via_tier1_expansion():
    schema = {
        "Disease": {
            "properties": {"name": "str", "ICD10": "str", "DSM5": "str"},
            "preferred_id": "doid",
        },
    }

    fake_adapter = FakeAdapter(
        values_by_type={"Disease": ["atrial fibrillation", "heart failure"]},
    )

    with mock.patch("biochatter.kg_grounding.make_adapter", return_value=fake_adapter):
        with mock.patch("biochatter.kg_grounding._make_ontomcp_manager", return_value=(None, None)):
            grounded_question, grounded = ground_entities(
                question="which genes are upregulated in AF?",
                selected_entity_terms=[("AF", "Disease")],
                connection_args={"host": "localhost", "port": "7687"},
                schema=schema,
                conversation_factory=fake_conversation_factory,
            )

    assert grounded["AF"] == "atrial fibrillation"
    assert "atrial fibrillation" in grounded_question
    assert "AF" not in grounded_question