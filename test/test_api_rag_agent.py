import os
import pytest

from biochatter.rag_agent import RagAgent

def test_api_agent():
    """
    Test the API agent with a specific DNA sequence question.
    """
    # Define the test question
    question = "Which organism does the DNA sequence come from: TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"

    # Create an instance of RagAgent in 'API' mode
    api_agent = RagAgent(
        mode="API",
        model_name="gpt-4",
        connection_args={},  # Add necessary connection arguments if needed
        use_prompt=True  # Ensure prompts are used to get responses
    )
    assert api_agent.mode == "API", "Agent mode should be 'API'"

    # Generate responses using the test question
    responses = api_agent.generate_responses(question)
    assert responses, "No responses generated"
    assert isinstance(responses, list), "Responses should be a list"
    assert all(isinstance(response, tuple) for response in responses), "Each response should be a tuple"

    if responses:
        print("Test response:", responses[0][1])


test_api_agent()