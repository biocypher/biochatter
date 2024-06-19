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

    # Generate responses using the test question
    responses = api_agent.execute(question)
    print(responses)
    # Print the responses
    # for response in responses:
    #     print("Response Text:", response[0])
    #     print("Response Metadata:", response[1])

# Run the test function
output = test_api_agent()
print(output)