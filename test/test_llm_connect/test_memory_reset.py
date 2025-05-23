#!/usr/bin/env python3
"""Test script to verify that reset method properly clears conversation memory."""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add the biochatter module to the path (adjust for test directory location)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biochatter.llm_connect.langgraph_conversation import LangGraphConversation
from langchain_core.messages import AIMessage


def create_mock_llm():
    """Create a mock LLM that simulates chat model behavior."""
    mock_llm = Mock()

    # Mock the invoke method to return a simple AI message
    def mock_invoke(messages):
        return AIMessage(content="Mock response")

    # Mock the bind_tools method to return self (chainable)
    def mock_bind_tools(tools):
        return mock_llm

    mock_llm.invoke = mock_invoke
    mock_llm.bind_tools = mock_bind_tools

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


if __name__ == "__main__":
    test_memory_reset_between_invokes()
    test_multiple_resets()
    test_reset_without_memory()
    print("All tests passed!")
