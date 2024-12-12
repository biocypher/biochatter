import os
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from biochatter.api_agent.abc import (
    BaseFetcher,
    BaseInterpreter,
    BaseQueryBuilder,
)
from biochatter.api_agent.anndata import AnnDataIOQueryBuilder
from biochatter.api_agent.scanpy_pp_reduced import ScanpyPpQueryBuilder
from biochatter.api_agent.api_agent import APIAgent
from biochatter.api_agent.blast import (
    BLAST_QUERY_PROMPT,
    BLAST_SUMMARY_PROMPT,
    BlastFetcher,
    BlastInterpreter,
    BlastQueryBuilder,
    BlastQueryParameters,
)
from biochatter.api_agent.oncokb import (
    ONCOKB_QUERY_PROMPT,
    ONCOKB_SUMMARY_PROMPT,
    OncoKBFetcher,
    OncoKBInterpreter,
    OncoKBQueryBuilder,
    OncoKBQueryParameters,
)
from biochatter.api_agent.scanpy_pl import (
    ScanpyPlQueryBuilder,
)

from biochatter.api_agent.scanpy_tl import ScanpyTlQueryBuilder
from biochatter.llm_connect import Conversation, GptConversation

from langchain_core.output_parsers import PydanticToolsParser


def conversation_factory():
    conversation = GptConversation(
        model_name="gpt-4o",
        correct=False,
        prompts={},
    )
    conversation.set_api_key(os.getenv("OPENAI_API_KEY"), user="test")
    return conversation


class TestQueryBuilder(BaseQueryBuilder):
    def create_runnable(
        self,
        query_parameters: BaseModel,
        conversation: Conversation,
    ) -> Callable[..., Any]:
        return "mock_runnable"

    def parameterise_query(
        self,
        question: str,
        conversation: Conversation,
    ) -> list[BaseModel]:
        return ["mock_result"]


class TestFetcher(BaseFetcher):
    def submit_query(self, request_data):
        return "mock_url"

    def fetch_results(
        self,
        request_data: list[BaseModel],
        retries: int | None = 3,
    ) -> str:
        return "mock_results"


class TestInterpreter(BaseInterpreter):
    def summarise_results(
        self,
        question: str,
        conversation_factory: Callable[..., Any],
        response_text: str,
    ) -> str:
        return "mock_summary"


class MockModel(BaseModel):
    field: str


@pytest.fixture
def query_builder():
    return TestQueryBuilder()


@pytest.fixture
def fetcher():
    return TestFetcher()


@pytest.fixture
def interpreter():
    return TestInterpreter()


@pytest.fixture
def test_agent(query_builder, fetcher, interpreter):
    return APIAgent(
        conversation_factory=MagicMock(),
        query_builder=query_builder,
        fetcher=fetcher,
        interpreter=interpreter,
    )


class TestAPIAgent:
    def test_parameterise_query(self, test_agent):
        result = test_agent.parameterise_query("Mock question")
        assert result == ["mock_result"]

    def test_fetch_results(self, test_agent):
        result = test_agent.fetch_results("mock_query_model")
        assert result == "mock_results"

    def test_summarise_results(self, test_agent):
        result = test_agent.summarise_results("mock_question", "mock_results")
        assert result == "mock_summary"

    def test_execute(self, test_agent):
        result = test_agent.execute("mock_question")
        assert result == "mock_summary"


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
            query_parameters=query_parameters,
            llm=None,
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
            question,
            mock_conversation_instance,
        )

        # Assert
        mock_create_runnable.assert_called_once_with(
            query_parameters=BlastQueryParameters,
            conversation=mock_conversation_instance,
        )
        mock_runnable.invoke.assert_called_once_with(
            {"input": f"Answer:\n{question} based on:\n {BLAST_QUERY_PROMPT}"},
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == mock_blast_query_parameters
        assert hasattr(result[0], "question_uuid")


class TestBlastFetcher:
    @patch("biochatter.api_agent.blast.BlastFetcher._submit_query")
    def test_submit_query(self, mock_submit_query):
        # Arrange
        mock_response = MagicMock()
        mock_submit_query.return_value = mock_response

        query_parameters = BlastQueryParameters()
        fetcher = BlastFetcher()

        # Act
        result = fetcher._submit_query(query_parameters)

        # Assert
        mock_submit_query.assert_called_once_with(query_parameters)
        assert result == mock_response

    @patch("biochatter.api_agent.blast.BlastFetcher.fetch_results")
    def test_fetch_results(self, mock_fetch_results):
        # Arrange
        mock_response = MagicMock()
        mock_fetch_results.return_value = mock_response

        query_parameters = [BlastQueryParameters()]
        fetcher = BlastFetcher()

        # Act
        result = fetcher.fetch_results(query_parameters)

        # Assert
        mock_fetch_results.assert_called_once_with([query_parameters[0]])
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
            ValueError,
            match="RID not found in BLAST submission response.",
        ):
            fetcher._submit_query(query_parameters)


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
        "biochatter.api_agent.blast.ChatPromptTemplate.from_messages",
    ) as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.return_value = mock_prompt_instance
        mock_chain = mock_prompt_instance | mock_conversation.chat | mock_output_parser
        yield mock_chain


class TestBlastInterpreter:
    # FIX THIS TEST
    def test_summarise_results(mock_prompt, mock_conversation, mock_chain):
        # Arrange
        interpreter = BlastInterpreter()
        question = "What is the best hit?"
        expected_context = "Mocked context from file"
        expected_summary_prompt = BLAST_SUMMARY_PROMPT.format(
            question=question,
            context=expected_context,
        )
        expected_answer = "Mocked answer"
        # Mock the methods and functions
        mock_chain.invoke = MagicMock(return_value=expected_answer)

        # Act
        result = interpreter.summarise_results(
            question,
            mock_conversation,
            expected_context,
        )

        # Assert
        assert result == expected_answer
        mock_chain.invoke.assert_called_once_with(
            {"input": {expected_summary_prompt}},
        )


class TestOncoKBQueryBuilder:
    @patch("biochatter.api_agent.oncokb.OncoKBQueryBuilder.create_runnable")
    def test_create_runnable(self, mock_create_runnable):
        # Arrange
        mock_runnable = MagicMock()
        mock_create_runnable.return_value = mock_runnable

        query_parameters = OncoKBQueryParameters(
            endpoint="specific/endPoint/toHit",
            base_url="https://demo.oncokb.org/api/v1",
        )
        builder = OncoKBQueryBuilder()

        # Act
        result = builder.create_runnable(
            query_parameters=query_parameters,
            llm=None,
        )

        # Assert
        mock_create_runnable.assert_called_once_with(
            query_parameters=query_parameters,
            llm=None,
        )
        assert result == mock_runnable

    @patch("biochatter.api_agent.oncokb.OncoKBQueryBuilder.create_runnable")
    @patch("biochatter.llm_connect.GptConversation")
    def test_parameterise_query(self, mock_conversation, mock_create_runnable):
        # Arrange
        mock_runnable = MagicMock()
        mock_create_runnable.return_value = mock_runnable

        mock_oncokb_query_parameters = MagicMock()
        mock_runnable.invoke.return_value = mock_oncokb_query_parameters

        question = "What is the sequence of the gene?"
        mock_conversation_instance = mock_conversation.return_value

        builder = OncoKBQueryBuilder()

        # Act
        result = builder.parameterise_query(
            question,
            mock_conversation_instance,
        )

        # Assert
        mock_create_runnable.assert_called_once_with(
            query_parameters=OncoKBQueryParameters,
            conversation=mock_conversation_instance,
        )
        mock_runnable.invoke.assert_called_once_with(
            {"input": f"Answer:\n{question} based on:\n {ONCOKB_QUERY_PROMPT}"},
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == mock_oncokb_query_parameters
        assert hasattr(result[0], "question_uuid")


class TestOncoKBFetcher:
    @patch("biochatter.api_agent.oncokb.OncoKBFetcher.fetch_results")
    def test_fetch_results(self, mock_fetch_results):
        # Arrange
        mock_response = MagicMock()
        mock_fetch_results.return_value = mock_response

        query_parameters = [OncoKBQueryParameters(endpoint="test/endpoint")]
        fetcher = OncoKBFetcher()

        # Act
        result = fetcher.fetch_results(query_parameters)

        # Assert
        mock_fetch_results.assert_called_once_with(query_parameters)
        assert result == mock_response


@pytest.fixture
def mock_conversation():
    with patch("biochatter.llm_connect.GptConversation") as mock:
        yield mock


@pytest.fixture
def mock_output_parser():
    with patch("biochatter.api_agent.oncokb.StrOutputParser") as mock:
        mock_parser = MagicMock()
        mock.return_value = mock_parser
        yield mock_parser


@pytest.fixture
def mock_chain(mock_conversation, mock_output_parser):
    with patch(
        "biochatter.api_agent.oncokb.ChatPromptTemplate.from_messages",
    ) as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.return_value = mock_prompt_instance
        mock_chain = mock_prompt_instance | mock_conversation.chat | mock_output_parser
        yield mock_chain


class TestOncoKBInterpreter:
    def test_summarise_results(mock_prompt, mock_conversation, mock_chain):
        # Arrange
        interpreter = OncoKBInterpreter()
        question = "What is the best hit?"
        expected_context = "Mocked context from file"
        expected_summary_prompt = ONCOKB_SUMMARY_PROMPT.format(
            question=question,
            context=expected_context,
        )
        expected_answer = "Mocked answer"

        # Mock the methods and functions
        mock_chain.invoke = MagicMock(return_value=expected_answer)

        # Act
        result = interpreter.summarise_results(
            question,
            mock_conversation,
            expected_context,
        )

        # Assert
        assert result == expected_answer
        mock_chain.invoke.assert_called_once_with(
            {"input": {expected_summary_prompt}},
        )


class TestScanpyPlQueryBuilder:
    @pytest.fixture
    def mock_create_runnable(self):
        with patch(
            "biochatter.api_agent.scanpy_pl.ScanpyPlQueryBuilder.create_runnable",
        ) as mock:
            mock_runnable = MagicMock()
            mock.return_value = mock_runnable
            yield mock_runnable

    def test_create_runnable(self):
        pass

    def test_parameterise_query(self, mock_create_runnable):
        # Arrange
        query_builder = ScanpyPlQueryBuilder()
        mock_conversation = MagicMock()
        question = "Create a scatter plot of n_genes_by_counts vs total_counts."
        expected_input = f"{question}"

        mock_query_obj = MagicMock()
        mock_create_runnable.invoke.return_value = mock_query_obj

        # Act
        result = query_builder.parameterise_query(question, mock_conversation)

        # Assert
        mock_create_runnable.invoke.assert_called_once_with(expected_input)
        assert result == mock_query_obj


class TestScanpyPlFetcher:
    pass


class TestScanpyPlInterpreter:
    pass


class TestAnndataIOQueryBuilder:
    @pytest.fixture
    def mock_create_runnable(self):
        with patch(
            "biochatter.api_agent.anndata.AnnDataIOQueryBuilder.create_runnable",
        ) as mock:
            mock_runnable = MagicMock()
            mock.return_value = mock_runnable
            yield mock_runnable

    def test_create_runnable(self):
        pass

    def test_parameterise_query(self, mock_create_runnable):
        # Arrange
        query_builder = AnnDataIOQueryBuilder()
        mock_conversation = MagicMock()
        question = "read a .h5ad file into an anndata object."
        expected_input = f"{question}"
        mock_query_obj = MagicMock()
        mock_create_runnable.invoke.return_value = mock_query_obj

        # Act
        result = query_builder.parameterise_query(question, mock_conversation)

        # Assert
        mock_create_runnable.invoke.assert_called_once_with(expected_input)
        assert result == mock_query_obj

class TestScanpyPpQueryBuilder:
    @pytest.fixture
    def mock_create_runnable(self):
        with patch(
            "biochatter.api_agent.scanpy_pp_reduced.ScanpyPpQueryBuilder.create_runnable",
        ) as mock:
            mock_runnable = MagicMock()
            mock.return_value = mock_runnable
            yield mock_runnable

    def test_create_runnable(self):
        pass

    def test_parameterise_query(self, mock_create_runnable):
        # Arrange
        query_builder = ScanpyPpQueryBuilder()
        mock_conversation = MagicMock()
        question = "I want to use scanpy pp to filter cells with at least 200 genes"
        expected_input = f"{question}"
        mock_query_obj = MagicMock()
        mock_create_runnable.invoke.return_value = mock_query_obj

        # Act
        result = query_builder.parameterise_query(question, mock_conversation)

        # Assert
        mock_create_runnable.invoke.assert_called_once_with(expected_input)
        assert result == mock_query_obj
"""
class TestScanpyTLQueryBuilder:
    @pytest.fixture
    def mock_create_runnable(self):
        with patch(
            "biochatter.api_agent.anndata.AnnDataIOQueryBuilder.create_runnable",
        ) as mock:
            mock_runnable = MagicMock()
            mock.return_value = mock_runnable
            yield mock_runnable

    @patch("biochatter.llm_connect.GptConversation")
    def test_create_runnable(self, mock_conversation):
        # Mock the list of Pydantic classes as a list of Mock objects
        class MockTool1(BaseModel):
            param1: str

        class MockTool2(BaseModel):
            param2: int

        mock_generated_classes = [MockTool1, MockTool2]

        # Mock the conversation object and LLM
        mock_conversation_instance = mock_conversation.return_value
        mock_llm = MagicMock()
        mock_conversation_instance.chat = mock_llm

        # Mock the LLM with tools
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        # Mock the chain
        mock_chain = MagicMock()
        mock_llm_with_tools.__or__.return_value = mock_chain

        # Act
        builder = ScanpyTlQueryBuilder()
        result = builder.create_runnable(
            query_parameters=mock_generated_classes,
            conversation=mock_conversation_instance,
        )

        # Assert
        mock_llm.bind_tools.assert_called_once_with(mock_generated_classes, tool_choice = "required")
        mock_llm_with_tools.__or__.assert_called_once_with(
            PydanticToolsParser(tools=mock_generated_classes)
        )
        # Verify the returned chain
        assert result == mock_chain

    def test_parameterise_query(self, mock_create_runnable):
        # Arrange
        query_builder = AnnDataIOQueryBuilder()
        mock_conversation = MagicMock()
        question = "i want to run PCA on my data"
        expected_input = f"{question}"
        mock_query_obj = MagicMock()
        mock_create_runnable.invoke.return_value = mock_query_obj

        # Act
        result = query_builder.parameterise_query(question, mock_conversation)

        # Assert
        mock_create_runnable.invoke.assert_called_once_with(expected_input)
        assert result == mock_query_obj
"""