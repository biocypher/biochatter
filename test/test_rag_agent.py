import json
import pytest
from unittest.mock import MagicMock, patch
from biochatter.rag_agent import RagAgent
from biochatter.vectorstore_agent import Document


def test_rag_agent_kg_mode():
    with patch("biochatter.database_agent.DatabaseAgent") as MockDatabaseAgent:
        mock_agent = MockDatabaseAgent.return_value
        mock_agent.get_query_results.return_value = [
            Document(
                page_content=json.dumps(
                    {
                        "node1": {
                            "name": "result1",
                            "description": "result1 description",
                            "type": "result1 type",
                        }
                    },
                ),
                metadata={
                    "cypher_query": "test_query",
                },
            ),
            Document(
                page_content=json.dumps(
                    {
                        "node2": {
                            "name": "result2",
                            "description": "result2 description",
                            "type": "result2 type",
                        }
                    },
                ),
                metadata={
                    "cypher_query": "test_query",
                },
            ),
            Document(
                page_content=json.dumps(
                    {
                        "node3": {
                            "name": "result3",
                            "description": "result3 description",
                            "type": "result3 type",
                        }
                    },
                ),
                metadata={
                    "cypher_query": "test_query",
                },
            ),
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
        result = agent.generate_responses("test question", 3)
        assert len(result) == 3
        assert type(result) == list
        assert type(result[0]) == tuple
        MockDatabaseAgent.assert_called_once()
        mock_agent.get_query_results.assert_called_once_with("test question", 3)


def test_rag_agent_vectorstore_mode():
    with patch(
        "biochatter.vectorstore_agent.VectorDatabaseAgentMilvus"
    ) as MockVectorDatabaseAgentMilvus:
        mock_agent = MockVectorDatabaseAgentMilvus.return_value
        mock_agent.similarity_search.return_value = [
            Document(
                page_content="result1",
                metadata={
                    "name": "result1",
                    "description": "result1 description",
                },
            ),
            Document(
                page_content="result2",
                metadata={
                    "name": "result2",
                    "description": "result2 description",
                },
            ),
            Document(
                page_content="result3",
                metadata={
                    "name": "result3",
                    "description": "result3 description",
                },
            ),
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
        result = agent.generate_responses("test question", 3)
        assert len(result) == 3
        assert type(result) == list
        assert type(result[0]) == tuple
        MockVectorDatabaseAgentMilvus.assert_called_once()
        mock_agent.similarity_search.assert_called_once_with("test question", 3)


def test_rag_agent_invalid_mode():
    with pytest.raises(ValueError) as excinfo:
        RagAgent(
            mode="invalid_mode", model_name="test_model", connection_args={}
        )
    assert (
        str(excinfo.value)
        == "Invalid mode. Choose either 'kg' or 'vectorstore'."
    )
