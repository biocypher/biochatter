from biochatter.vectorstore import (
    DocumentEmbedder,
    DocumentReader,
    Document,
    XinferenceDocumentEmbedder,
)
import os
from xinference.client import Client
from unittest.mock import patch, MagicMock

print(os.getcwd())

# setup milvus connection
if os.getenv("DEVCONTAINER"):
    _HOST = "milvus-standalone"
else:
    _HOST = "127.0.0.1"
_PORT = "19530"

splitted_docs = [
    Document(
        page_content="Democratising",
        metadata={
            "format": "PDF 1.4",
            "title": "Accepted_Version_Editor_edits_May_2023",
            "producer": "Skia/PDF m115 Google Docs Renderer",
            "source": "pdf",
        },
    ),
    Document(
        page_content="",
        metadata={
            "format": "PDF 1.4",
            "title": "Accepted_Version_Editor_edits_May_2023",
            "producer": "Skia/PDF m115 Google Docs Renderer",
            "source": "pdf",
        },
    ),
    Document(
        page_content="",
        metadata={
            "format": "PDF 1.4",
            "title": "Accepted_Version_Editor_edits_May_2023",
            "producer": "Skia/PDF m115 Google Docs Renderer",
            "source": "pdf",
        },
    ),
]

@patch("biochatter.vectorstore.OpenAIEmbeddings")
@patch("biochatter.vectorstore.VectorDatabaseAgentMilvus")
@patch("biochatter.vectorstore.RecursiveCharacterTextSplitter")
def test_retrieval_augmented_generation(
    mock_textsplitter, mock_host, mock_openaiembeddings
):
    # mocking
    mock_textsplitter.from_huggingface_tokenizer.return_value = (
        mock_textsplitter()
    )
    mock_textsplitter.from_tiktoken_encoder.return_value = mock_textsplitter()
    mock_textsplitter.return_value.split_documents.return_value = splitted_docs
    mock_host.return_value.store_embeddings.return_value = "1"

    pdf_path = "test/bc_summary.pdf"
    with open(pdf_path, "rb") as f:
        doc_bytes = f.read()
    assert isinstance(doc_bytes, bytes)

    reader = DocumentReader()
    doc = reader.document_from_pdf(doc_bytes)
    rag_agent = DocumentEmbedder(
        embedding_collection_name="openai_embedding_test",
        metadata_collection_name="openai_metadata_test",
        connection_args={"host": _HOST, "port": _PORT},
    )
    rag_agent.connect()
    doc_id = rag_agent.save_document(doc)
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

    mock_host.return_value.get_all_documents.return_value = [
        {"id": "1"},
        {"id": "2"},
    ]
    docs = rag_agent.get_all_documents()
    cnt = len(docs)
    assert cnt > 0
    rag_agent.remove_document(doc_id)
    mock_host.return_value.get_all_documents.return_value = [{"id": "2"}]
    docs = rag_agent.get_all_documents()
    assert (cnt - 1) == len(docs)
    mock_host.return_value.remove_document.assert_called_once()


@patch("xinference.client.Client")
@patch("biochatter.vectorstore.XinferenceEmbeddings")
@patch("biochatter.vectorstore.VectorDatabaseAgentMilvus")
@patch("biochatter.vectorstore.RecursiveCharacterTextSplitter")
def test_retrieval_augmented_generation_generic_api(
    mock_textsplitter, mock_host, mock_embeddings, mock_client
):
    # mocking
    mock_textsplitter.from_huggingface_tokenizer.return_value = (
        mock_textsplitter()
    )
    mock_textsplitter.from_tiktoken_encoder.return_value = mock_textsplitter()
    mock_textsplitter.return_value.split_documents.return_value = splitted_docs
    mock_host.return_value.store_embeddings.return_value = "1"
    mock_client.return_value.list_models.return_value = {
        "48c76b62-904c-11ee-a3d2-0242acac0302": {
            "model_type": "embedding",
            "address": "",
            "accelerators": ["0"],
            "model_name": "gte-large",
            "dimensions": 1024,
            "max_tokens": 512,
            "language": ["en"],
            "model_revision": "",
        }
    }

    pdf_path = "test/bc_summary.pdf"
    with open(pdf_path, "rb") as f:
        doc_bytes = f.read()
    assert isinstance(doc_bytes, bytes)

    reader = DocumentReader()
    doc = reader.document_from_pdf(doc_bytes)

    rag_agent = XinferenceDocumentEmbedder(
        base_url=os.getenv(
            "GENERIC_TEST_OPENAI_BASE_URL", "http://llm.biocypher.org/"
        ),
        embedding_collection_name="xinference_embedding_test",
        metadata_collection_name="xinference_metadata_test",
        connection_args={"host": _HOST, "port": _PORT},
    )
    rag_agent.connect()

    doc_id = rag_agent.save_document(doc)
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

    mock_host.return_value.get_all_documents.return_value = [
        {"id": "1"},
        {"id": "2"},
    ]
    docs = rag_agent.get_all_documents()
    cnt = len(docs)
    assert cnt > 0
    rag_agent.remove_document(doc_id)
    mock_host.return_value.get_all_documents.return_value = [{"id": "2"}]
    docs = rag_agent.get_all_documents()
    assert (cnt - 1) == len(docs)
    mock_host.return_value.remove_document.assert_called_once()


def test_load_txt():
    reader = DocumentReader()
    text_path = "test/bc_summary.txt"
    document = reader.load_document(text_path)
    assert isinstance(document, list)
    assert isinstance(document[0], Document)


def test_load_pdf():
    reader = DocumentReader()
    pdf_path = "test/bc_summary.pdf"
    document = reader.load_document(pdf_path)
    assert isinstance(document, list)
    assert isinstance(document[0], Document)


def test_byte_txt():
    text_path = "test/bc_summary.txt"
    with open(text_path, "rb") as f:
        document = f.read()
    assert isinstance(document, bytes)

    reader = DocumentReader()
    doc = reader.document_from_txt(document)
    assert isinstance(doc, list)
    assert isinstance(doc[0], Document)
    # do we want byte string or regular string?


def test_byte_pdf():
    pdf_path = "test/bc_summary.pdf"
    # open as type "application/pdf"
    with open(pdf_path, "rb") as f:
        document = f.read()
    assert isinstance(document, bytes)

    reader = DocumentReader()
    doc = reader.document_from_pdf(document)
    assert isinstance(doc, list)
    assert isinstance(doc[0], Document)
    assert "numerous attempts at standardising KGs" in doc[0].page_content


CHUNK_SIZE = 100
CHUNK_OVERLAP = 10


def check_document_splitter(
    rag_agent: DocumentEmbedder,
    fn: str,
    expected_length: int,
):
    reader = DocumentReader()
    doc = reader.load_document(fn)
    splitted = rag_agent._split_document(doc)
    assert expected_length == len(splitted)


@patch("biochatter.vectorstore.OpenAIEmbeddings")
@patch("biochatter.vectorstore.VectorDatabaseAgentMilvus")
@patch("biochatter.vectorstore.RecursiveCharacterTextSplitter")
def test_split_by_characters(mock_textsplitter, mock_host, mock_embeddings):
    # character splitter
    mock_textsplitter.from_huggingface_tokenizer.return_value = (
        mock_textsplitter()
    )
    mock_textsplitter.from_tiktoken_encoder.return_value = mock_textsplitter()
    mock_textsplitter.return_value.split_documents.return_value = splitted_docs
    rag_agent = DocumentEmbedder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    check_document_splitter(
        rag_agent, "test/bc_summary.pdf", len(splitted_docs)
    )
    check_document_splitter(rag_agent, "test/dcn.pdf", len(splitted_docs))
    check_document_splitter(
        rag_agent, "test/bc_summary.txt", len(splitted_docs)
    )

    # tiktoken
    rag_agent = DocumentEmbedder(
        split_by_characters=False,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    check_document_splitter(
        rag_agent, "test/bc_summary.pdf", len(splitted_docs)
    )
    mock_textsplitter.from_tiktoken_encoder.assert_called_once()
    check_document_splitter(rag_agent, "test/dcn.pdf", len(splitted_docs))
    assert mock_textsplitter.from_tiktoken_encoder.call_count == 2
    check_document_splitter(
        rag_agent, "test/bc_summary.txt", len(splitted_docs)
    )
    assert mock_textsplitter.from_tiktoken_encoder.call_count == 3

    # huggingface tokenizer
    rag_agent = DocumentEmbedder(
        split_by_characters=False,
        model="bigscience/bloom",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    check_document_splitter(
        rag_agent, "test/bc_summary.pdf", len(splitted_docs)
    )
    mock_textsplitter.from_huggingface_tokenizer.assert_called_once()
    check_document_splitter(rag_agent, "test/dcn.pdf", len(splitted_docs))
    assert mock_textsplitter.from_huggingface_tokenizer.call_count == 2
    check_document_splitter(
        rag_agent, "test/bc_summary.txt", len(splitted_docs)
    )
    assert mock_textsplitter.from_huggingface_tokenizer.call_count == 3
