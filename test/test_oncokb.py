import os
import unittest

import pytest

from biochatter.llm_connect import GptConversation
from biochatter.api_agent.oncokb import (
    OncoKBQueryParameters,
    OncoKBQueryBuilder,
    OncoKBFetcher,
    OncoKBInterpreter,
)
from biochatter.api_agent.api_agent import APIAgent


@pytest.fixture
def oncokb_api_agent():
    def conversation_factory():
        conversation = GptConversation(
            model_name="gpt-4o",
            prompts={},
            correct=False,
        )
        conversation.set_api_key(os.getenv("OPENAI_API_KEY"), user="test")
        return conversation

    return APIAgent(
        conversation_factory=conversation_factory,
        query_builder=OncoKBQueryBuilder(),
        result_fetcher=OncoKBFetcher(),
        result_interpreter=OncoKBInterpreter(),
    )


@pytest.mark.skip(reason="Live test for development purposes")
def test_fetch_oncokb_results(oncokb_api_agent):
    question = "What can you tell me about BRAF in cancer?"
    # Run the method to test
    answer = oncokb_api_agent.execute(question)
    print(answer)
    # assert "rattus norwegicus" in answer.lower()


@pytest.mark.skip(reason="Live test for development purposes")
class TestOncokbQueryBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = OncoKBQueryBuilder()
        self.fetcher = OncoKBFetcher()
        self.interpreter = OncoKBInterpreter()
