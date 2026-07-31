from unittest import mock

from biochatter.database_agent import DatabaseAgent
from biochatter.langgraph_agent_base import ReflexionAgentResult


def test_get_query_results_with_reflexion():
    db_agent = DatabaseAgent(
        model_provider="openai",
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
        "biochatter.database_agent.KGQueryReflexionAgent",
    ) as mock_KGQueryReflexionAgent:
        mock_KGQueryReflexionAgent.return_value.execute.return_value = ReflexionAgentResult(
            answer="test_query",
            tool_result=[
                {"key": "value"},
                {"key": "value"},
                {"key": "value"},
                {"key": "value"},
            ],
        )

        # Mock the prompt_engine.generate_query method
        with mock.patch.object(
            db_agent.prompt_engine,
            "generate_query_prompt",
            return_value="prompts for user's question",
        ):
            result = db_agent.get_query_results("test_query", 3)

    # Check if the result is as expected
    expected_result = db_agent._build_response(
        [
            {"key": "value"},
            {"key": "value"},
            {"key": "value"},
        ],
        "test_query",
    )
    assert result == expected_result


def test_get_query_results_without_reflexion():
    db_agent = DatabaseAgent(
        model_provider="openai",
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
        db_agent.prompt_engine,
        "generate_query",
        return_value="test_query",
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
    expected_result = db_agent._build_response(
        [
            {"key": "value"},
            {"key": "value"},
            {"key": "value"},
        ],
        "test_query",
    )
    assert result == expected_result


def test_database_agent_passes_model_provider_to_prompt_engine():
    """Test that DatabaseAgent properly passes model_provider to BioCypherPromptEngine."""
    with mock.patch("biochatter.database_agent.BioCypherPromptEngine") as MockPromptEngine:
        mock_prompt_engine = MockPromptEngine.return_value

        db_agent = DatabaseAgent(
            model_provider="google_genai",
            model_name="gemini-3.5-flash",
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

        # Verify that BioCypherPromptEngine was called with the correct model_provider
        MockPromptEngine.assert_called_once_with(
            model_provider="google_genai",
            model_name="gemini-3.5-flash",
            schema_config_or_info_dict={"schema_config": "test_schema"},
            conversation_factory=None,
            connection_args={
                "db_name": "test_db",
                "host": "localhost",
                "port": 7687,
                "user": "neo4j",
                "password": "password",
                },
                use_grounding=False,
)

        assert db_agent.prompt_engine == mock_prompt_engine


def test_database_agent_defaults_to_gemini():
    """Test that DatabaseAgent works with the new default Gemini model."""
    with mock.patch("biochatter.database_agent.BioCypherPromptEngine") as MockPromptEngine:
        mock_prompt_engine = MockPromptEngine.return_value

        db_agent = DatabaseAgent(
            model_provider="google_genai",
            model_name="gemini-3.5-flash",
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

        # Verify that BioCypherPromptEngine was called with Gemini defaults
        MockPromptEngine.assert_called_once_with(
            model_provider="google_genai",
            model_name="gemini-3.5-flash",
            schema_config_or_info_dict={"schema_config": "test_schema"},
            conversation_factory=None,
            connection_args={
                "db_name": "test_db",
                "host": "localhost",
                "port": 7687,
                "user": "neo4j",
                "password": "password",
            },
            use_grounding=False,
        )

def test_query_sanitisation_strips_markdown_fences():
    """Test that markdown code fences are stripped from generated queries."""
    db_agent = DatabaseAgent(
        model_provider="openai",
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
    db_agent.connect()

    with mock.patch.object(
        db_agent.prompt_engine,
        "generate_query",
        return_value="```cypher\nMATCH (n) RETURN n\n```",
    ):
        with mock.patch.object(
            db_agent.driver,
            "query",
            return_value=[[{"key": "value"}], {}],
        ) as mock_query:
            db_agent.get_query_results("test question", 3)
            mock_query.assert_called_once_with(query="MATCH (n) RETURN n")


def test_query_sanitisation_passes_clean_query_unchanged():
    """Test that clean queries without markdown pass through unchanged."""
    db_agent = DatabaseAgent(
        model_provider="openai",
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
    db_agent.connect()

    with mock.patch.object(
        db_agent.prompt_engine,
        "generate_query",
        return_value="MATCH (n) RETURN n",
    ):
        with mock.patch.object(
            db_agent.driver,
            "query",
            return_value=[[{"key": "value"}], {}],
        ) as mock_query:
            db_agent.get_query_results("test question", 3)
            mock_query.assert_called_once_with(query="MATCH (n) RETURN n")


def test_query_sanitisation_strips_sql_markdown_fences():
    """Test that markdown fences are stripped for non-Cypher query languages."""
    db_agent = DatabaseAgent(
        model_provider="openai",
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
    db_agent.connect()

    with mock.patch.object(
        db_agent.prompt_engine,
        "generate_query",
        return_value="```sql\nSELECT * FROM table\n```",
    ):
        with mock.patch.object(
            db_agent.driver,
            "query",
            return_value=[[{"key": "value"}], {}],
        ) as mock_query:
            db_agent.get_query_results("test question", 3)
            mock_query.assert_called_once_with(query="SELECT * FROM table")

def test_query_sanitisation_strips_fences_without_language_tag():
    """Test that markdown fences without a language tag are also stripped."""
    db_agent = DatabaseAgent(
        model_provider="openai",
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
    db_agent.connect()

    with mock.patch.object(
        db_agent.prompt_engine,
        "generate_query",
        return_value="```\nSELECT * FROM table\n```",
    ):
        with mock.patch.object(
            db_agent.driver,
            "query",
            return_value=[[{"key": "value"}], {}],
        ) as mock_query:
            db_agent.get_query_results("test question", 3)
            mock_query.assert_called_once_with(query="SELECT * FROM table")
