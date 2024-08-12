from biochatter.correction_langgraph_agent import CorrectionReflexionAgent
from biochatter.kg_langgraph_agent import KGQueryReflexionAgent
import pytest
import os
from biochatter.llm_connect import GptConversation


def conversation_factory():
    conversation = GptConversation(
        model_name="gpt-4o-mini", prompts={}, correct=False
    )
    conversation.set_api_key(os.getenv("OPENAI_API_KEY"), user="biochatter")
    return conversation


# @pytest.skip("Live test for development")
def test_correction_langgraph_agent():
    agent = CorrectionReflexionAgent(
        conversation_factory=conversation_factory,
    )
    query = "What is the function of the protein TP53?"
    result = agent.execute(query)
    print(result)
