from unittest.mock import Mock, MagicMock, patch
import os
import uuid
import unittest

import pytest
import requests

from biochatter.llm_connect import GptConversation
from biochatter.api_agent.blast import (
    BLAST_QUERY_PROMPT,
    BLAST_SUMMARY_PROMPT,
    BlastFetcher,
    BlastInterpreter,
    BlastQueryBuilder,
    BlastQueryParameters,
)
from biochatter.api_agent.api_agent import APIAgent

from biochatter.llm_connect import GptConversation
from biochatter.api_agent.oncokb import (
    OncoKBFetcher,
    OncoKBQueryBuilder,
    OncoKBQueryParameters,
)


def conversation_factory():
    conversation = GptConversation(
        model_name="gpt-4o",
        correct=False,
        prompts={},
    )
    conversation.set_api_key(os.getenv("OPENAI_API_KEY"), user="test")
    return conversation


class TestBlastQueryBuilder:
    @patch("biochatter.api_agent.blast.BlastQueryBuilder.create_runnable")
    def test_create_runnable(self, mock_create_runnable):
        # Arrange
        mock_runnable = MagicMock()
        mock_create_runnable.return_value = mock_runnable

        query_parameters = BlastQueryParameters()
        builder = BlastQueryBuilder()

        # Act
        result = builder.create_runnable(
            query_parameters=query_parameters, llm=None
        )

        # Assert
        mock_create_runnable.assert_called_once_with(
            query_parameters=query_parameters,
            llm=None,
        )
        assert result == mock_runnable

    @patch("biochatter.api_agent.blast.BlastQueryBuilder.create_runnable")
    @patch("biochatter.llm_connect.GptConversation")
    def test_parameterise_query(self, mock_conversation, mock_create_runnable):
        # Arrange
        mock_runnable = MagicMock()
        mock_create_runnable.return_value = mock_runnable

        mock_blast_query_parameters = MagicMock()
        mock_runnable.invoke.return_value = mock_blast_query_parameters

        question = "What is the sequence of the gene?"
        mock_conversation_instance = mock_conversation.return_value

        builder = BlastQueryBuilder()

        # Act
        result = builder.parameterise_query(
            question, mock_conversation_instance
        )

        # Assert
        mock_create_runnable.assert_called_once_with(
            query_parameters=BlastQueryParameters,
            conversation=mock_conversation_instance,
        )
        mock_runnable.invoke.assert_called_once_with(
            {"input": f"Answer:\n{question} based on:\n {BLAST_QUERY_PROMPT}"}
        )
        assert result == mock_blast_query_parameters
        assert hasattr(result, "question_uuid")


class TestBlastFetcher:
    @patch("biochatter.api_agent.blast.BlastFetcher.submit_query")
    def test_submit_query(self, mock_submit_query):
        # Arrange
        mock_response = MagicMock()
        mock_submit_query.return_value = mock_response

        query_parameters = BlastQueryParameters()
        fetcher = BlastFetcher()

        # Act
        result = fetcher.submit_query(query_parameters)

        # Assert
        mock_submit_query.assert_called_once_with(query_parameters)
        assert result == mock_response

    @patch("biochatter.api_agent.blast.BlastFetcher.fetch_and_save_results")
    def test_fetch_and_save_results(self, mock_fetch_and_save_results):
        # Arrange
        mock_response = MagicMock()
        mock_fetch_and_save_results.return_value = mock_response

        query_id = "test_query_id"
        fetcher = BlastFetcher()

        # Act
        result = fetcher.fetch_and_save_results(query_id)

        # Assert
        mock_fetch_and_save_results.assert_called_once_with(query_id)
        assert result == mock_response

    @patch("requests.post")
    def test_submit_query_value_error(self, mock_post):
        # Arrange
        mock_response = MagicMock()
        mock_response.text = "No RID found in this response"
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        query_parameters = BlastQueryParameters(
            cmd="some_cmd",
            program="some_program",
            database="some_database",
            query="some_query",
            format_type="some_format",
            megablast="some_megablast",
            max_hits=10,
            url="http://example.com/blast",
        )
        fetcher = BlastFetcher()

        # Act & Assert
        with pytest.raises(
            ValueError, match="RID not found in BLAST submission response."
        ):
            fetcher.submit_query(query_parameters)


@pytest.fixture
def mock_conversation():
    with patch("biochatter.llm_connect.GptConversation") as mock:
        yield mock


@pytest.fixture
def mock_output_parser():
    with patch("biochatter.api_agent.blast.StrOutputParser") as mock:
        mock_parser = MagicMock()
        mock.return_value = mock_parser
        yield mock_parser


@pytest.fixture
def mock_chain(mock_conversation, mock_output_parser):
    with patch(
        "biochatter.api_agent.blast.ChatPromptTemplate.from_messages"
    ) as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.return_value = mock_prompt_instance
        mock_chain = (
            mock_prompt_instance | mock_conversation.chat | mock_output_parser
        )
        yield mock_chain


class TestBlastInterpreter:
    @patch("builtins.open", new_callable=MagicMock)
    def test_read_first_n_lines_file_not_found(self, mock_open):
        # Arrange
        mock_open.side_effect = FileNotFoundError
        interpreter = BlastInterpreter()

        # Act
        result = interpreter.read_first_n_lines("non_existent_file.txt", 5)

        # Assert
        assert result == "The file was not found."

    @patch("builtins.open", new_callable=MagicMock)
    def test_read_first_n_lines_general_exception(self, mock_open):
        # Arrange
        mock_open.side_effect = Exception("Some error")
        interpreter = BlastInterpreter()

        # Act
        result = interpreter.read_first_n_lines("some_file.txt", 5)

        # Assert
        assert result == "An error occurred: Some error"

    @patch("builtins.open", new_callable=MagicMock)
    def test_read_first_n_lines_success(self, mock_open):
        # Arrange
        mock_file = MagicMock()
        mock_file.readline.side_effect = [
            "line1\n",
            "line2\n",
            "line3\n",
            "",
            "",
        ]
        mock_open.return_value.__enter__.return_value = mock_file
        interpreter = BlastInterpreter()

        # Act
        result = interpreter.read_first_n_lines("some_file.txt", 5)

        # Assert
        assert result == "line1\nline2\nline3"

    def test_summarise_results(mock_prompt, mock_conversation, mock_chain):
        # Arrange
        interpreter = BlastInterpreter()
        question = "What is the best hit?"
        file_path = "test_blast_results.txt"
        n_lines = 10
        expected_context = "Mocked context from file"
        expected_summary_prompt = BLAST_SUMMARY_PROMPT.format(
            question=question, context=expected_context
        )
        expected_answer = "Mocked answer"

        # Mock the methods and functions
        interpreter.read_first_n_lines = MagicMock(
            return_value=expected_context
        )
        mock_chain.invoke = MagicMock(return_value=expected_answer)

        # Act
        result = interpreter.summarise_results(
            question, mock_conversation, file_path, n_lines
        )

        # Assert
        assert result == expected_answer
        interpreter.read_first_n_lines.assert_called_once_with(
            file_path, n_lines
        )
        mock_chain.invoke.assert_called_once_with(
            {"input": {expected_summary_prompt}}
        )


@pytest.mark.skip(reason="Non-mocked development test")
class TestBlastQueryBuilderNoMock(unittest.TestCase):
    def setUp(self):
        self.builder = BlastQueryBuilder()
        conversation = conversation_factory()
        self.llm = conversation.chat

    def test_create_runnable(self):
        builder = BlastQueryBuilder()
        blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
        blast_prompt = builder.read_blast_prompt(blast_prompt_path)
        self.assertIsNotNone(blast_prompt)
        runnable = builder.create_runnable(self.llm, BlastQueryParameters)
        self.assertIsNotNone(runnable)

    @patch("biochatter.api_agent.create_structured_output_runnable")
    def test_generate_blast_query(self, mock_create_runnable):
        blast_prompt_path = "docs/api_agent/BLAST_tool/persistent_files/api_documentation/BLAST.txt"
        question = "Which organism does the DNA sequence come from:TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"
        mock_runnable = Mock()
        mock_create_runnable.return_value = mock_runnable
        mock_blast_call_obj = Mock()
        mock_runnable.invoke.return_value = mock_blast_call_obj
        blast_query = self.builder.generate_blast_query(
            question, blast_prompt_path, self.llm
        )
        print(blast_query.question_uuid)

    @patch("requests.post")
    def test_submit_blast_query(self, mock_post):
        blast_query = BlastQueryParameters(
            cmd="blastn",
            program="blastn",
            database="nt",
            query="AGCTG",
            format_type="XML",
            megablast=True,
            max_hits=10,
            url="https://blast.ncbi.nlm.nih.gov/Blast.cgi",
        )
        mock_response = Mock()
        mock_response.text = "RID = 1234"
        mock_post.return_value = mock_response

        rid = self.builder.submit_blast_query(blast_query)

        self.assertEqual(rid, "1234")
        print(rid)


@pytest.mark.skip(reason="Non-mocked development test")
class TestBlastFetcherNoMock(unittest.TestCase):
    @patch("requests.get")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_fetch_and_save_blast_results(self, mock_open, mock_get):
        fetcher = BlastFetcher()
        question_uuid = str(uuid.uuid4())
        BLAST_result_path = "docs/api_agent/BLAST_tool/BLAST_response/results"
        blast_query_return = "62YGMDCX013"

        # Mock the GET request response for status checking COULD add not ready response, but it would delay the test
        mock_status_response_ready = Mock()
        mock_status_response_ready.text = "Status=READY\nThereAreHits=yes"

        mock_results_response = Mock()
        mock_results_response.text = "Mock BLAST results"

        # Setup the side effect for status and results checking
        mock_get.side_effect = [
            mock_status_response_ready,
            mock_results_response,
        ]

        file_name = fetcher.fetch_and_save_results(
            question_uuid, blast_query_return, BLAST_result_path, 100
        )
        # assert file name is correct
        self.assertEqual(file_name, f"BLAST_results_{question_uuid}.txt")

        # Verify the file write operations
        expected_path = f"{BLAST_result_path}/{file_name}"
        # assert the paths
        mock_open.assert_called_once_with(expected_path, "w")
        # assert the correct content is in the file
        mock_open().write.assert_called_once_with("Mock BLAST results")

        @patch(
            "builtins.open",
            new_callable=mock_open,
            read_data="line1\nline2\nline3\nline4\nline5\n",
        )
        @patch.object(BlastFetcher, "read_first_n_lines")
        def test_answer_extraction(
            self,
            mock_parser,
            mock_llm,
            mock_prompt,
            mock_read_first_n_lines,
            mock_file,
        ):
            fetcher = BlastFetcher()
            question = "What organism does this sequence belong to?"
            file_path = "fake_path.txt"
            n = 3
            mock_read_first_n_lines.return_value = "line1\nline2\nline3"
            mock_prompt.from_messages.return_value = mock_prompt
            mock_parser_instance = mock_parser.return_value
            mock_parser_instance.invoke.return_value = "Mocked Answer"

            result = fetcher.answer_extraction(question, file_path, n)

            self.assertEqual(result, "Mocked Answer")
            mock_read_first_n_lines.assert_called_once_with(file_path, n)
            mock_parser_instance.invoke.assert_called_once()


@pytest.mark.skip(reason="Non-mocked development test")
class TestAPIAgent(unittest.TestCase):
    """TO DO: add test for errors in the APIAgent class."""

    @patch("biochatter.api_agent.BlastQueryBuilder")
    @patch("biochatter.api_agent.BlastFetcher")
    def test_execute_blast_query(
        self,
        mock_blast_fetcher,
        mock_blast_query_builder,
    ):
        api_agent = APIAgent(conversation_factory=conversation_factory)

        # Prepare mocks
        question = "Which organism does the DNA sequence come from: TTCATCGGTCTGAGCAGAGGATGAAGTTGCAAATGATGCAAGCAAAACAGCTCAAAGATGAAGAGGAAAAGGCTATACACAACAGGAGCAATGTAGATACAGAAGGT"
        mock_blast_query = Mock(spec=BlastQueryParameters)
        mock_blast_query_builder_instance = (
            mock_blast_query_builder.return_value
        )
        mock_blast_query_builder_instance.generate_blast_query.return_value = (
            mock_blast_query
        )
        mock_blast_query.question_uuid = str(uuid.uuid4())
        mock_blast_query_builder_instance.submit_blast_query.return_value = (
            "MOCK_RID"
        )
        mock_blast_fetcher_instance = mock_blast_fetcher.return_value
        mock_blast_fetcher_instance.fetch_and_save_blast_results.return_value = (
            "mock_file.txt"
        )
        mock_blast_fetcher_instance.answer_extraction.return_value = (
            "Mocked Answer"
        )

        # Mock the LLM-related functions
        mock_runnable = Mock()
        mock_blast_query_builder_instance.create_runnable.return_value = (
            mock_runnable
        )
        mock_runnable.invoke.return_value = mock_blast_query

        # Run the method to test
        api_agent.execute(question)

        # Assertions
        self.assertEqual(api_agent.final_answer, "Mocked Answer")
        self.assertIsNone(api_agent.error)
        # Verify the method calls within APIAgent
        conversation = conversation_factory()
        mock_blast_query_builder_instance.generate_blast_query.assert_called_once_with(
            question, api_agent.blast_prompt_path, conversation.chat
        )
        mock_blast_query_builder_instance.submit_blast_query.assert_called_once_with(
            mock_blast_query
        )
        mock_blast_fetcher_instance.fetch_and_save_blast_results.assert_called_once_with(
            mock_blast_query.question_uuid,
            "MOCK_RID",
            api_agent.result_path,
            100,
        )
        mock_blast_fetcher_instance.answer_extraction.assert_called_once_with(
            question, ".blast/mock_file.txt", 100
        )


@pytest.mark.skip(reason="Non-mocked development test")
class TestOncoKBQueryBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = OncoKBQueryBuilder()
        conversation = conversation_factory()
        self.llm = conversation.chat

    def test_create_runnable(self):
        builder = OncoKBQueryBuilder()
        self.assertIsNotNone(builder)
        runnable = builder.create_runnable(self.llm, OncoKBQueryParameters)
        self.assertIsNotNone(runnable)

    @patch("biochatter.api_agent.create_structured_output_runnable")
    def test_generate_oncokb_query(self, mock_create_runnable):
        question = (
            "What is the annotation for the mutation BRAF V600E in Melanoma?"
        )
        mock_runnable = Mock()
        mock_create_runnable.return_value = mock_runnable
        mock_oncokb_call_obj = Mock()
        mock_runnable.invoke.return_value = mock_oncokb_call_obj
        oncokb_query = self.builder.parameterise_query(question, self.llm)
        print(oncokb_query.question_uuid)

    @patch("requests.get")
    def test_submit_oncokb_query(self, mock_get):
        oncokb_query = OncoKBQueryParameters(
            endpoint="annotate/mutations/byProteinChange",
            hugoSymbol="BRAF",
            alteration="V600E",
            tumorType="Melanoma",
        )
        mock_response = Mock()
        mock_response.url = "https://demo.oncokb.org/api/v1/annotate/mutations/byProteinChange?hugoSymbol=BRAF&alteration=V600E&tumorType=Melanoma"
        mock_get.return_value = mock_response

        fetcher = OncoKBFetcher()
        url = fetcher.submit_query(oncokb_query)

        self.assertEqual(url, mock_response.url)
        print(url)


@pytest.mark.skip(reason="Non-mocked development test")
class TestOncoKBFetcher(unittest.TestCase):
    @patch("requests.get")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_fetch_and_save_oncokb_results(self, mock_open, mock_get):
        fetcher = OncoKBFetcher()
        question_uuid = str(uuid.uuid4())
        oncokb_result_path = (
            "docs/api_agent/oncokb_tool/oncokb_response/results"
        )
        oncokb_query_return = "https://demo.oncokb.org/api/v1/annotate/mutations/byProteinChange?hugoSymbol=BRAF&alteration=V600E&tumorType=Melanoma"

        mock_response = Mock()
        mock_response.text = "Mock OncoKB results"

        mock_get.return_value = mock_response

        file_name = fetcher.fetch_and_save_results(
            question_uuid, oncokb_query_return, oncokb_result_path, 100
        )
        self.assertEqual(file_name, f"OncoKB_results_{question_uuid}.oncokb")

        expected_path = f"{oncokb_result_path}/{file_name}"
        mock_open.assert_called_once_with(expected_path, "w")
        mock_open().write.assert_called_once_with("Mock OncoKB results")

        @patch(
            "builtins.open",
            new_callable=mock_open,
            read_data="line1\nline2\nline3\nline4\nline5\n",
        )
        @patch.object(OncoKBFetcher, "read_first_n_lines")
        def test_answer_extraction(
            self,
            mock_parser,
            mock_llm,
            mock_prompt,
            mock_read_first_n_lines,
            mock_file,
        ):
            fetcher = OncoKBFetcher()
            question = "What is the annotation for the mutation BRAF V600E in Melanoma?"
            file_path = "fake_path.txt"
            n = 3
            mock_read_first_n_lines.return_value = "line1\nline2\nline3"
            mock_prompt.from_messages.return_value = mock_prompt
            mock_parser_instance = mock_parser.return_value
            mock_parser_instance.invoke.return_value = "Mocked Answer"

            result = fetcher.answer_extraction(question, file_path, n)

            self.assertEqual(result, "Mocked Answer")
            mock_read_first_n_lines.assert_called_once_with(file_path, n)
            mock_parser_instance.invoke.assert_called_once()


@pytest.mark.skip(reason="Non-mocked development test")
class TestAPIAgent(unittest.TestCase):
    @patch("biochatter.api_agent.OncoKBQueryBuilder")
    @patch("biochatter.api_agent.OncoKBFetcher")
    def test_execute_oncokb_query(
        self,
        mock_oncokb_fetcher,
        mock_oncokb_query_builder,
    ):
        api_agent = APIAgent(conversation_factory=conversation_factory)

        question = (
            "What is the annotation for the mutation BRAF V600E in Melanoma?"
        )
        mock_oncokb_query = Mock(spec=OncoKBQueryParameters)
        mock_oncokb_query_builder_instance = (
            mock_oncokb_query_builder.return_value
        )
        mock_oncokb_query_builder_instance.parameterise_query.return_value = (
            mock_oncokb_query
        )
        mock_oncokb_query.question_uuid = str(uuid.uuid4())
        mock_oncokb_query_builder_instance.submit_query.return_value = (
            "MOCK_URL"
        )
        mock_oncokb_fetcher_instance = mock_oncokb_fetcher.return_value
        mock_oncokb_fetcher_instance.fetch_and_save_results.return_value = (
            "mock_file.oncokb"
        )
        mock_oncokb_fetcher_instance.answer_extraction.return_value = (
            "Mocked Answer"
        )

        mock_runnable = Mock()
        mock_oncokb_query_builder_instance.create_runnable.return_value = (
            mock_runnable
        )
        mock_runnable.invoke.return_value = mock_oncokb_query

        api_agent.execute(question)

        self.assertEqual(api_agent.final_answer, "Mocked Answer")
        self.assertIsNone(api_agent.error)

        conversation = conversation_factory()
        mock_oncokb_query_builder_instance.parameterise_query.assert_called_once_with(
            question, conversation.chat
        )
        mock_oncokb_query_builder_instance.submit_query.assert_called_once_with(
            mock_oncokb_query
        )
        mock_oncokb_fetcher_instance.fetch_and_save_results.assert_called_once_with(
            mock_oncokb_query.question_uuid,
            "MOCK_URL",
            api_agent.result_path,
            100,
        )
        mock_oncokb_fetcher_instance.answer_extraction.assert_called_once_with(
            question, ".oncokb/mock_file.oncokb", 100
        )


if __name__ == "__main__":
    unittest.main()
    pytest.main()
