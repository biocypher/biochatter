from unittest.mock import MagicMock, patch
import os
import json

import pytest

from biochatter.rag_agent import RagAgent, RagAgentModeEnum
from biochatter.llm_connect import GptConversation
from biochatter.vectorstore_agent import Document


def conversation_factory():
    conversation = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
    )
    conversation.set_api_key(os.getenv("OPENAI_API_KEY"), user="test")

    return conversation


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
            use_prompt=True,
            mode=RagAgentModeEnum.KG,
            model_name="test_model",
            connection_args={"host": "xxx", "port": "xxx"},
            schema_config_or_info_dict={
                "schema_config_or_info_dict": "test_schema_config_or_info_dict"
            },
        )
        assert agent.mode == RagAgentModeEnum.KG
        assert agent.model_name == "test_model"
        result = agent.generate_responses("test question")
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
            use_prompt=True,
            mode=RagAgentModeEnum.VectorStore,
            model_name="test_model",
            connection_args={"host": "xxx", "port": "xxx"},
            embedding_func=MagicMock(),
        )
        assert agent.mode == RagAgentModeEnum.VectorStore
        assert agent.model_name == "test_model"
        result = agent.generate_responses("test question")
        assert len(result) == 3
        assert type(result) == list
        assert type(result[0]) == tuple
        MockVectorDatabaseAgentMilvus.assert_called_once()
        mock_agent.similarity_search.assert_called_once_with(
            "test question", 3, doc_ids=None
        )


def test_rag_agent_invalid_mode():
    with pytest.raises(ValueError) as excinfo:
        RagAgent(
            mode="invalid_mode", model_name="test_model", connection_args={}
        )
    assert "Invalid mode. Choose either" in str(excinfo.value)


def conversation_factory():
    # Mock conversation factory
    return MagicMock()


@patch("biochatter.api_agent.oncokb.OncoKBQueryBuilder")
@patch("biochatter.api_agent.oncokb.OncoKBFetcher")
@patch("biochatter.api_agent.oncokb.OncoKBInterpreter")
def test_rag_agent_api_oncokb_mode(
    mock_query_builder,
    mock_fetcher,
    mock_interpreter,
):
    """
    Test the API agent in 'api_oncokb' mode.
    """
    # Create an instance of RagAgent in 'api_oncokb' mode
    rag_agent = RagAgent(
        mode=RagAgentModeEnum.API_ONCOKB,
        model_name="gpt-4o",
        use_prompt=True,  # Ensure prompts are used to get responses
        conversation_factory=conversation_factory,
    )
    assert (
        rag_agent.mode == RagAgentModeEnum.API_ONCOKB
    ), "Agent mode should be 'api_oncokb'"

    # Define the test question
    question = "What is the oncogenic potential of BRAF mutation?"

    # Generate responses using the test question
    responses = rag_agent.generate_responses(question)
    assert responses, "No responses generated"
    assert isinstance(responses, list), "Responses should be a list"
    assert all(
        isinstance(response, tuple) for response in responses
    ), "Each response should be a tuple"

    if responses:
        print("Test response:", responses[0][1])

    mock_query_builder.assert_called_once()
    mock_fetcher.assert_called_once()
    mock_interpreter.assert_called_once()


@pytest.mark.skip(reason="Live test for development purposes")
def test_rag_agent_api_mode_no_mock():
    """
    Test the API agent with a specific DNA sequence question.
    """
    # Define the test question
    question = "Which organism does the DNA sequence come from: TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"

    # Create an instance of RagAgent in 'api' mode
    api_agent = RagAgent(
        mode="api_blast",
        model_name="gpt-4o",
        use_prompt=True,  # Ensure prompts are used to get responses
        conversation_factory=conversation_factory,
    )
    assert api_agent.mode == "api_blast", "Agent mode should be 'api_blast'"

    # Generate responses using the test question
    responses = api_agent.generate_responses(question)
    assert responses, "No responses generated"
    assert isinstance(responses, list), "Responses should be a list"
    assert all(
        isinstance(response, tuple) for response in responses
    ), "Each response should be a tuple"

    if responses:
        print("Test response:", responses[0][1])
