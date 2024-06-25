import os
import unittest

from langchain.chat_models import ChatOpenAI
import pytest

from biochatter.api_agent.api_agent import (  # Adjust the import as necessary
    APIAgent,
    BlastFetcher,
    BlastQueryBuilder,
    BlastInterpreter,
)
from biochatter.llm_connect import GptConversation

###
### TO DO
###
### Continue to refactor the code in biochatter/api_agent.py, and test it here.


@pytest.mark.skip(reason="Live test for development purposes")
class TestBlastQueryBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = BlastQueryBuilder()
        self.fetcher = BlastFetcher()
        self.interpreter = BlastInterpreter()

    # def test_blast_structured_output_prompt(self):
    #   builder = BlastQueryBuilder()
    #   prompt_template = builder.BLAST_structured_output_prompt
    #   self.assertIsNotNone(prompt_template)

    # def test_read_blast_prompt(self):
    #   builder = BlastQueryBuilder()
    #   blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
    #   blast_prompt = builder.read_blast_prompt(blast_prompt_path)
    #   self.assertIsNotNone(blast_prompt)

    # def test_create_runnable(self):
    #   builder = BlastQueryBuilder()
    #   blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
    #   blast_prompt = builder.read_blast_prompt(blast_prompt_path)
    #   self.assertIsNotNone(blast_prompt)
    #   runnable = builder.create_runnable(llm, BlastQuery)
    #   self.assertIsNotNone(runnable)

    # def test_generate_blast_query(self):
    #   builder = BlastQueryBuilder()
    #   blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
    #   question = "Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"
    #   blast_query = builder.generate_blast_query(question, blast_prompt_path, llm)
    #   self.assertIsNotNone(blast_query)
    #   # query = builder.submit_blast_query("Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT")
    #   # self.assertIsNotNone(query)
    # def test_execute_blast_query(self):
    #   builder = BlastQueryBuilder()
    #   blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
    #   question = "Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"
    #   blast_query = builder.generate_blast_query(question, blast_prompt_path, llm)
    #   self.assertIsNotNone(blast_query)
    #   print(blast_query)
    #   rid = builder.submit_blast_query(blast_query)    # self.assertIsNotNone(query)


# def test_fetch_blast_results(self):
#   builder = BlastQueryBuilder()
#   fetcher = BlastFetcher()
#   BLAST_result_path = "docs/api_agent/BLAST_tool/BLAST_response/results"
#   blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
#   question = "Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"
#   blast_query = builder.generate_blast_query(question, blast_prompt_path, llm)
#   self.assertIsNotNone(blast_query)
#   print(blast_query)
#   rid = builder.submit_blast_query(blast_query)
#   print(rid)
#   blast_file_name = fetcher.fetch_and_save_blast_results(blast_query.question_uuid, '62YGMDCX013', BLAST_result_path, 100)
#   final_answer = fetcher.answer_extraction(question, os.path.join(BLAST_result_path, blast_file_name), 100)
#   print(final_answer)

import os

import pytest


@pytest.fixture
def api_agent():
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
        query_builder=BlastQueryBuilder(),
        result_fetcher=BlastFetcher(),
        result_interpreter=BlastInterpreter(),
    )


@pytest.mark.skip(reason="Live test for development purposes")
def test_fetch_blast_results(api_agent):
    question = "Which organism does the DNA sequence come from: TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"

    # Run the method to test
    api_agent.execute(question)

    # Check for the final answer or errors
    assert hasattr(
        api_agent, "final_answer"
    ), "The API agent does not have a final_answer attribute."
    assert hasattr(
        api_agent, "error"
    ), "The API agent does not have an error attribute."

    if api_agent.final_answer:
        print("Test passed with response:", api_agent.final_answer)
    else:
        assert api_agent.error, "The API agent failed without setting an error."
        print("Test failed with error:", api_agent.error)


if __name__ == "__main__":
    pytest.main()
