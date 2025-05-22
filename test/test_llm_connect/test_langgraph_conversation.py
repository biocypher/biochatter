"""Tests for the LangGraph conversation module."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from biochatter.llm_connect.langgraph_conversation import LangGraphConversation, ConversationState


# Mock tools for testing
@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers together."""
    return first_int + second_int


@tool
def echo(text: str) -> str:
    """Echo the input text back."""
    return f"Echo: {text}"


class TestConversationState:
    """Test the ConversationState Pydantic model."""

    def test_conversation_state_initialization(self):
        """Test that ConversationState initializes correctly."""
        state: ConversationState = {"messages": [], "plan": None, "intermediate_steps": []}
        assert state["messages"] == []
        assert state["plan"] is None
        assert state["intermediate_steps"] == []

    def test_conversation_state_with_data(self):
        """Test ConversationState with initial data."""
        message = HumanMessage(content="Hello")
        state: ConversationState = {
            "messages": [message],
            "plan": "1. Step one",
            "intermediate_steps": [("tool1", "result1")],
        }
        assert len(state["messages"]) == 1
        assert state["plan"] == "1. Step one"
        assert len(state["intermediate_steps"]) == 1


class TestLangGraphConversation:
    """Test the LangGraphConversation class."""

    def test_initialization(self):
        """Test basic initialization of LangGraphConversation."""
        with patch("biochatter.llm_connect.langgraph_conversation.init_chat_model") as mock_init:
            mock_llm = MagicMock()
            mock_init.return_value = mock_llm

            convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai")

            assert convo.model_name == "gpt-4o"
            assert convo.model_provider == "openai"
            assert convo.tools == []
            assert convo.config == {}
            assert convo.checkpointer is None
            assert convo.llm == mock_llm

            mock_init.assert_called_once_with(model="gpt-4o", model_provider="openai")

    def test_initialization_with_tools(self):
        """Test initialization with tools."""
        with patch("biochatter.llm_connect.langgraph_conversation.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()

            tools = [add, echo]
            convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai", tools=tools)

            assert len(convo.tools) == 2
            assert convo.tools[0] == add
            assert convo.tools[1] == echo

    def test_initialization_with_config_and_kwargs(self):
        """Test initialization with config and LLM kwargs."""
        with patch("biochatter.llm_connect.langgraph_conversation.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()

            config = {"user_id": "test_user"}
            llm_kwargs = {"temperature": 0.5}

            convo = LangGraphConversation(
                model_name="gpt-4o", model_provider="openai", config=config, llm_kwargs=llm_kwargs
            )

            assert convo.config == config
            mock_init.assert_called_once_with(model="gpt-4o", model_provider="openai", temperature=0.5)

    def test_bind_tools(self):
        """Test binding additional tools."""
        with patch("biochatter.llm_connect.langgraph_conversation.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()

            convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai", tools=[add])

            assert len(convo.tools) == 1

            # Bind additional tools
            convo.bind_tools([echo])

            assert len(convo.tools) == 2
            assert add in convo.tools
            assert echo in convo.tools

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_direct_response_path(self, mock_init):
        """Test the direct response path (no tools)."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock router response (DIRECT)
        router_response = MagicMock()
        router_response.content = "DIRECT"

        # Mock final response
        final_response = MagicMock()
        final_response.content = "The capital of France is Paris."

        # Set up call sequence
        mock_llm.invoke.side_effect = [router_response, final_response]

        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai", tools=[add, echo])

        result = convo.invoke("What is the capital of France?")

        assert result == "The capital of France is Paris."
        assert mock_llm.invoke.call_count == 2  # Router + final response
        # Verify bind_tools was not called for direct response
        mock_llm.bind_tools.assert_not_called()

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_tool_response_path(self, mock_init):
        """Test the tool response path (single tool usage)."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_init.return_value = mock_llm

        # Mock router response (TOOL)
        router_response = MagicMock()
        router_response.content = "TOOL"

        # Mock final response
        final_response = MagicMock()
        final_response.content = "The sum is 8"

        # Set up call sequence
        mock_llm.invoke.side_effect = [router_response]
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_with_tools.invoke.return_value = final_response

        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai", tools=[add, echo])

        result = convo.invoke("Add 5 and 3")

        assert result == "The sum is 8"
        assert mock_llm.invoke.call_count == 1  # Only router call
        # Verify bind_tools was called for tool response
        mock_llm.bind_tools.assert_called_once_with([add, echo])
        mock_llm_with_tools.invoke.assert_called_once()

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_tool_response_path_no_tools(self, mock_init):
        """Test the tool response path fallback when no tools available."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock router response (TOOL)
        router_response = MagicMock()
        router_response.content = "TOOL"

        # Mock final response
        final_response = MagicMock()
        final_response.content = "I cannot perform calculations without tools."

        # Set up call sequence
        mock_llm.invoke.side_effect = [router_response, final_response]

        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai")  # No tools

        result = convo.invoke("Add 5 and 3")

        assert result == "I cannot perform calculations without tools."
        assert mock_llm.invoke.call_count == 2  # Router + fallback response
        # Verify bind_tools was not called since no tools available
        mock_llm.bind_tools.assert_not_called()

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_planned_execution_path(self, mock_init):
        """Test the planned execution path with tool calls."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock router response (plan)
        router_response = MagicMock()
        router_response.content = "1. Add 5 and 3\n2. Echo the result"

        # Mock final synthesis response
        final_response = MagicMock()
        final_response.content = "Based on the calculations, 5 + 3 = 8, and echoing gives: Echo: 8"

        mock_llm.invoke.side_effect = [router_response, final_response]

        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai", tools=[add, echo])

        result = convo.invoke("Add 5 and 3, then echo the result")

        assert "Echo: 8" in result or "8" in result
        assert mock_llm.invoke.call_count == 2  # Router + synthesis

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_invoke_with_adhoc_tools(self, mock_init):
        """Test invoke with ad-hoc tools."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock direct response
        router_response = MagicMock()
        router_response.content = "DIRECT"

        final_response = MagicMock()
        final_response.content = "Echo: Hello World"

        mock_llm.invoke.side_effect = [router_response, final_response]

        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai")

        # Initially no tools
        assert len(convo.tools) == 0

        # Invoke with ad-hoc tools
        result = convo.invoke("Echo hello world", tools=[echo])

        # Tools should be restored after invocation
        assert len(convo.tools) == 0
        assert result == "Echo: Hello World"
        # Verify bind_tools was not called for direct response even with ad-hoc tools
        mock_llm.bind_tools.assert_not_called()

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_tool_response_with_adhoc_tools(self, mock_init):
        """Test tool response with ad-hoc tools."""
        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_init.return_value = mock_llm

        # Mock tool response
        router_response = MagicMock()
        router_response.content = "TOOL"

        final_response = MagicMock()
        final_response.content = "Echo: Hello World"

        mock_llm.invoke.side_effect = [router_response]
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        mock_llm_with_tools.invoke.return_value = final_response

        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai")

        # Initially no tools
        assert len(convo.tools) == 0

        # Invoke with ad-hoc tools
        result = convo.invoke("Echo hello world", tools=[echo])

        # Tools should be restored after invocation
        assert len(convo.tools) == 0
        assert result == "Echo: Hello World"
        # Verify bind_tools was called with ad-hoc tools
        mock_llm.bind_tools.assert_called_once_with([echo])
        mock_llm_with_tools.invoke.assert_called_once()

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_parse_plan(self, mock_init):
        """Test plan parsing functionality."""
        mock_init.return_value = MagicMock()

        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai")

        # Test numbered plan
        plan = "1. First step\n2. Second step\n3. Third step"
        steps = convo._parse_plan(plan)

        assert len(steps) == 3
        assert "First step" in steps[0]
        assert "Second step" in steps[1]
        assert "Third step" in steps[2]

        # Test empty plan
        assert convo._parse_plan("") == []
        assert convo._parse_plan(None) == []

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_resolve_tool_for_step(self, mock_init):
        """Test tool resolution for plan steps."""
        mock_init.return_value = MagicMock()

        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai")

        tools = [add, echo]

        # Test successful tool resolution
        tool_name, tool_args = convo._resolve_tool_for_step("Add two numbers", tools)
        assert tool_name == "add"
        assert tool_args == {"input": "Add two numbers"}

        # Test echo tool resolution
        tool_name, tool_args = convo._resolve_tool_for_step("Echo something", tools)
        assert tool_name == "echo"
        assert tool_args == {"input": "Echo something"}

        # Test no matching tool
        tool_name, tool_args = convo._resolve_tool_for_step("Unknown operation", tools)
        assert tool_name is None
        assert tool_args is None

        # Test with no tools available
        tool_name, tool_args = convo._resolve_tool_for_step("Any step", [])
        assert tool_name is None
        assert tool_args is None

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_create_synthesis_prompt(self, mock_init):
        """Test synthesis prompt creation."""
        mock_init.return_value = MagicMock()

        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai")

        original_query = "Calculate 5 + 3 and echo the result"
        intermediate_steps = [("add", 8), ("echo", "Echo: 8")]

        prompt = convo._create_synthesis_prompt(original_query, intermediate_steps)

        assert "Calculate 5 + 3 and echo the result" in prompt
        assert "- add: 8" in prompt
        assert "- echo: Echo: 8" in prompt
        assert "synthesize" in prompt.lower()

    @patch("biochatter.llm_connect.langgraph_conversation.init_chat_model")
    def test_no_response_generated(self, mock_init):
        """Test handling when no response is generated."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        # Mock empty response
        router_response = MagicMock()
        router_response.content = "DIRECT"

        mock_llm.invoke.side_effect = [router_response]
        mock_llm.bind_tools.return_value = mock_llm

        # Mock graph returning empty messages
        convo = LangGraphConversation(model_name="gpt-4o", model_provider="openai")

        with patch.object(convo.graph, "invoke") as mock_graph:
            mock_graph.return_value = {"messages": []}

            result = convo.invoke("Test query")
            assert result == "No response generated."
