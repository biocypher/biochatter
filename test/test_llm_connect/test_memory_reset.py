#!/usr/bin/env python3
"""Test script to verify that reset method properly clears conversation memory."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add the biochatter module to the path (adjust for test directory location)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from biochatter.llm_connect.langgraph_conversation import LangGraphConversation
from langchain_core.messages import AIMessage


def create_mock_llm():
    """Create a mock LLM that returns predictable responses for testing."""
    mock_llm = MagicMock()

    def mock_invoke(messages):
        # Check if this is a routing message
        if len(messages) == 1 and "Decide how to handle" in messages[0].content:
            # Return DIRECT to force direct response path
            return AIMessage(content="DIRECT")
        # For all other messages, return the standard mock response
        return AIMessage(content="Mock response")

    def mock_bind_tools(tools):
        return mock_llm

    mock_llm.invoke.side_effect = mock_invoke
    mock_llm.bind_tools.side_effect = mock_bind_tools
    return mock_llm


def test_memory_reset_between_invokes():
    """Test that reset properly clears memory between invoke calls."""
    # Create conversation with memory enabled
    conv = LangGraphConversation(
        model_name="gpt-3.5-turbo",
        model_provider="openai",
        save_history=True,
        thread_id=42,
    )

    # Replace the real LLM with a mock
    conv.llm = create_mock_llm()

    # Store original memory reference to verify it gets replaced
    original_memory = conv.memory
    original_graph = conv.graph

    # First invoke - establish some context
    response1 = conv.invoke("Please remember that my favorite color is blue and my name is Alice.")
    assert response1 == "Mock response", "First invoke should return mock response"

    # Reset conversation memory
    conv.reset()

    # Verify new instances were created
    assert conv.memory is not original_memory, "Memory instance should be replaced"
    assert conv.graph is not original_graph, "Graph instance should be replaced"
    assert conv.config["configurable"]["thread_id"] == 42, "Thread ID should be preserved"

    # Verify memory is empty after reset
    memory_empty_after_reset = not bool(conv.memory.storage) if hasattr(conv.memory, "storage") else True
    assert memory_empty_after_reset, "Memory should be empty after reset"

    # Second invoke - should have no memory of first
    response2 = conv.invoke("What is my name and favorite color?")
    assert response2 == "Mock response", "Second invoke should return mock response"

    # Verify conversation still functional
    response3 = conv.invoke("Hello, can you help me?")
    assert response3 == "Mock response", "Third invoke should return mock response"


def test_memory_reset_sqlite():
    """Test that reset properly clears SQLite memory between invoke calls."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_conversation.db"

        # Create conversation with SQLite memory enabled
        conv = LangGraphConversation(
            model_name="gpt-3.5-turbo",
            model_provider="openai",
            save_history=True,
            thread_id=42,
            checkpointer_type="sqlite",
            sqlite_db_path=str(db_path),
        )

        # Replace the real LLM with a mock
        conv.llm = create_mock_llm()

        # Store original memory reference to verify it gets replaced
        original_memory = conv.memory
        original_graph = conv.graph

        # Verify initial setup
        assert conv.checkpointer_type == "sqlite", "Should use SQLite checkpointer"
        assert conv.sqlite_db_path == str(db_path), "Should store correct DB path"

        # First invoke - establish some context
        response1 = conv.invoke("Please remember that my favorite color is blue.")
        assert response1 == "Mock response", "First invoke should return mock response"

        # Reset conversation memory
        conv.reset()

        # Verify new instances were created
        assert conv.memory is not original_memory, "Memory instance should be replaced"
        assert conv.graph is not original_graph, "Graph instance should be replaced"
        assert conv.config["configurable"]["thread_id"] == 42, "Thread ID should be preserved"
        assert conv.checkpointer_type == "sqlite", "Checkpointer type should be preserved"
        assert conv.sqlite_db_path == str(db_path), "DB path should be preserved"

        # Second invoke - should have no memory of first
        response2 = conv.invoke("What is my favorite color?")
        assert response2 == "Mock response", "Second invoke should return mock response"


def test_memory_management_setup():
    """Test the _setup_memory_management auxiliary method."""
    # Test MemorySaver setup
    conv_memory = LangGraphConversation(
        model_name="gpt-3.5-turbo",
        model_provider="openai",
        checkpointer_type="memory",
    )
    conv_memory.llm = create_mock_llm()

    assert conv_memory.checkpointer_type == "memory", "Should use memory checkpointer"
    assert conv_memory.memory is not None, "Memory should be initialized"
    assert "thread_id" in conv_memory.config["configurable"], "Config should have thread_id"

    # Test SqliteSaver setup with default path
    conv_sqlite = LangGraphConversation(
        model_name="gpt-3.5-turbo",
        model_provider="openai",
        checkpointer_type="sqlite",
    )
    conv_sqlite.llm = create_mock_llm()

    assert conv_sqlite.checkpointer_type == "sqlite", "Should use SQLite checkpointer"
    assert conv_sqlite.memory is not None, "Memory should be initialized"
    assert conv_sqlite.sqlite_db_path == "conversation_checkpoints.db", "Should use default DB path"

    # Test SqliteSaver setup with custom path
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_db_path = str(Path(temp_dir) / "custom.db")
        conv_sqlite_custom = LangGraphConversation(
            model_name="gpt-3.5-turbo",
            model_provider="openai",
            checkpointer_type="sqlite",
            sqlite_db_path=custom_db_path,
        )
        conv_sqlite_custom.llm = create_mock_llm()

        assert conv_sqlite_custom.sqlite_db_path == custom_db_path, "Should use custom DB path"

    # Test no history setup
    conv_no_hist = LangGraphConversation(
        model_name="gpt-3.5-turbo",
        model_provider="openai",
        save_history=False,
    )
    conv_no_hist.llm = create_mock_llm()

    assert conv_no_hist.memory is None, "Memory should be None"
    assert conv_no_hist.config == {}, "Config should be empty"


def test_multiple_resets():
    """Test that multiple resets work correctly."""
    conv = LangGraphConversation(model_name="gpt-3.5-turbo", model_provider="openai", save_history=True, thread_id=123)

    # Replace the real LLM with a mock
    conv.llm = create_mock_llm()

    memory_objects = [conv.memory]
    graph_objects = [conv.graph]

    # Perform multiple resets
    for i in range(3):
        conv.reset()
        memory_objects.append(conv.memory)
        graph_objects.append(conv.graph)

    # Verify all objects are different (using 'is' comparison)
    for i in range(len(memory_objects)):
        for j in range(i + 1, len(memory_objects)):
            assert memory_objects[i] is not memory_objects[j], f"Memory objects {i} and {j} should be different"

    for i in range(len(graph_objects)):
        for j in range(i + 1, len(graph_objects)):
            assert graph_objects[i] is not graph_objects[j], f"Graph objects {i} and {j} should be different"


def test_multiple_resets_sqlite():
    """Test that multiple resets work correctly with SQLite."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_resets.db"

        conv = LangGraphConversation(
            model_name="gpt-3.5-turbo",
            model_provider="openai",
            save_history=True,
            thread_id=123,
            checkpointer_type="sqlite",
            sqlite_db_path=str(db_path),
        )

        # Replace the real LLM with a mock
        conv.llm = create_mock_llm()

        memory_objects = [conv.memory]
        graph_objects = [conv.graph]

        # Perform multiple resets
        for i in range(3):
            conv.reset()
            memory_objects.append(conv.memory)
            graph_objects.append(conv.graph)
            # Verify properties are preserved
            assert conv.checkpointer_type == "sqlite", f"Checkpointer type should be preserved after reset {i}"
            assert conv.sqlite_db_path == str(db_path), f"DB path should be preserved after reset {i}"

        # Verify all objects are different (using 'is' comparison)
        for i in range(len(memory_objects)):
            for j in range(i + 1, len(memory_objects)):
                assert memory_objects[i] is not memory_objects[j], f"Memory objects {i} and {j} should be different"


def test_reset_without_memory():
    """Test that reset fails appropriately when memory is disabled."""
    conv = LangGraphConversation(model_name="gpt-3.5-turbo", model_provider="openai", save_history=False)

    # Replace the real LLM with a mock
    conv.llm = create_mock_llm()

    # Should raise ValueError when trying to reset without memory
    try:
        conv.reset()
        assert False, "Reset should fail when memory is disabled"
    except ValueError as e:
        assert "No checkpointer configured" in str(e), "Should raise appropriate error message"


def test_invalid_checkpointer_type():
    """Test that invalid checkpointer types raise appropriate errors."""
    try:
        conv = LangGraphConversation(
            model_name="gpt-3.5-turbo",
            model_provider="openai",
            checkpointer_type="invalid_type",  # type: ignore
        )
        assert False, "Should raise ValueError for invalid checkpointer type"
    except ValueError as e:
        assert "Unsupported checkpointer_type" in str(e), "Should raise appropriate error message"


if __name__ == "__main__":
    test_memory_reset_between_invokes()
    test_memory_reset_sqlite()
    test_memory_management_setup()
    test_multiple_resets()
    test_multiple_resets_sqlite()
    test_reset_without_memory()
    test_invalid_checkpointer_type()
    print("All tests passed!")
