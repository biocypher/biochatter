import unittest.mock as mock

from biochatter.database_agent import Document, DatabaseAgent
from biochatter.langgraph_agent_base import ReflexionAgentResult


def test_get_query_results_with_reflexion():
    db_agent = DatabaseAgent(
        model_name="model_name",
        connection_args={
            "db_name": "test_db",
            "host": "localhost",
            "port": 7687,
            "user": "neo4j",
            "password": "password",
        },
        schema_config_or_info_dict={"schema_config": "test_schema"},
        conversation_factory=None,
        use_reflexion=True,
    )
    db_agent.connect()  # Call the connect method to initialize the driver

    with mock.patch(
        "biochatter.database_agent.KGQueryReflexionAgent"
    ) as mock_KGQueryReflexionAgent:
        mock_KGQueryReflexionAgent.return_value.execute.return_value = (
            ReflexionAgentResult(
                answer="test_query",
                tool_result=[
                    {"key": "value"},
                    {"key": "value"},
                    {"key": "value"},
                    {"key": "value"},
                ],
            )
        )

        # Mock the prompt_engine.generate_query method
        with mock.patch.object(
            db_agent.prompt_engine,
            "generate_query_prompt",
            return_value="prompts for user's question",
        ):
            result = db_agent.get_query_results("test_query", 3)

    # Check if the result is as expected
    expected_result = db_agent._build_response([
        {"key": "value"},{"key": "value"},{"key": "value"},{"key": "value"}
    ], "test_query")
    assert result == expected_result


def test_get_query_results_without_reflexion():
    db_agent = DatabaseAgent(
        model_name="model_name",
        connection_args={
            "db_name": "test_db",
            "host": "localhost",
            "port": 7687,
            "user": "neo4j",
            "password": "password",
        },
        schema_config_or_info_dict={"schema_config": "test_schema"},
        conversation_factory=None,
        use_reflexion=False,
    )
    db_agent.connect()  # Call the connect method to initialize the driver

    # Mock the prompt_engine.generate_query method
    with mock.patch.object(
        db_agent.prompt_engine, "generate_query", return_value="test_query"
    ):
        # Mock the driver.query method
        with mock.patch.object(
            db_agent.driver,
            "query",
            return_value=[
                [
                    {"key": "value"},
                    {"key": "value"},
                    {"key": "value"},
                    {"key": "value"},
                ],
                {},
            ],
        ):
            result = db_agent.get_query_results("test_query", 3)

    # Check if the result is as expected
    expected_result = db_agent._build_response([
        {"key": "value"},{"key": "value"},{"key": "value"},{"key": "value"}
    ], "test_query")
    assert result == expected_result
