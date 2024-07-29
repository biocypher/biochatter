import unittest.mock as mock

from biochatter.database_agent import Document, DatabaseAgent
from biochatter.langgraph_agent_base import ReflexionAgentResult


def test_get_query_results():
    db_agent = DatabaseAgent(
        "model_name",
        {
            "db_name": "test_db",
            "host": "localhost",
            "port": 7687,
            "user": "neo4j",
            "password": "password",
        },
        {"schema_config": "test_schema"},
        None,
    )
    db_agent.connect()  # Call the connect method to initialize the driver

    with mock.patch(
        "biochatter.database_agent.KGQueryReflexionAgent"
    ) as mock_KGQueryReflexionAgent:
        mock_KGQueryReflexionAgent.return_value.execute.return_value = ReflexionAgentResult(
            answer="test_query", tool_result=[
                {"key": "value"},
                {"key": "value"},
                {"key": "value"},
                {"key": "value"},
            ]
        )
        
        # Mock the prompt_engine.generate_query method
        with mock.patch.object(
            db_agent.prompt_engine,
            "generate_query_prompt",
            return_value="prompts for user's question",
        ):
            result = db_agent.get_query_results("test_query", 3)

    # Check if the result is as expected
    expected_result = [
        Document(
            page_content='{"key": "value"}',
            metadata={"cypher_query": "test_query"},
        ),
        Document(
            page_content='{"key": "value"}',
            metadata={"cypher_query": "test_query"},
        ),
        Document(
            page_content='{"key": "value"}',
            metadata={"cypher_query": "test_query"},
        ),
    ]
    assert result == expected_result
