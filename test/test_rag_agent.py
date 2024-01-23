import pytest
from unittest.mock import MagicMock, patch
from biochatter.rag_agent import RagAgent


def test_rag_agent_kg_mode():
    with patch("biochatter.database.DatabaseAgent") as MockDatabaseAgent:
        mock_agent = MockDatabaseAgent.return_value
        mock_agent.get_query_results.return_value = [
            "result1",
            "result2",
            "result3",
        ]
        agent = RagAgent(
            mode="kg",
            model_name="test_model",
            connection_args={},
            schema_config_or_info_dict={
                "schema_config_or_info_dict": "test_schema_config_or_info_dict"
            },
        )
        assert agent.mode == "kg"
        assert agent.model_name == "test_model"
        assert agent.generate_responses("test question", 3) == [
            "result1",
            "result2",
            "result3",
        ]
        MockDatabaseAgent.assert_called_once()
        mock_agent.get_query_results.assert_called_once_with("test question")


def test_rag_agent_vectorstore_mode():
    with patch(
        "biochatter.vectorstore_host.VectorDatabaseHostMilvus"
    ) as MockVectorDatabaseHostMilvus:
        mock_agent = MockVectorDatabaseHostMilvus.return_value
        mock_agent.similarity_search.return_value = [
            "result1",
            "result2",
            "result3",
        ]
        agent = RagAgent(
            mode="vectorstore",
            model_name="test_model",
            connection_args={},
            embedding_func=MagicMock(),
            embedding_collection_name="test_embedding_collection",
            metadata_collection_name="test_metadata_collection",
        )
        assert agent.mode == "vectorstore"
        assert agent.model_name == "test_model"
        assert agent.generate_responses("test question", 3) == [
            "result1",
            "result2",
            "result3",
        ]
        MockVectorDatabaseHostMilvus.assert_called_once()
        mock_agent.similarity_search.assert_called_once_with("test question")


def test_rag_agent_invalid_mode():
    with pytest.raises(ValueError) as excinfo:
        RagAgent(
            mode="invalid_mode", model_name="test_model", connection_args={}
        )
    assert (
        str(excinfo.value)
        == "Invalid mode. Choose either 'kg' or 'vectorstore'."
    )
