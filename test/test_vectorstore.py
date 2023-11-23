from biochatter.vectorstore import (
    DocumentEmbedder,
    DocumentReader,
    Document,
    XinferenceDocumentEmbedder,
)
import os

print(os.getcwd())

# setup milvus connection
if os.getenv("DEVCONTAINER"):
    _HOST = "milvus-standalone"
else:
    _HOST = "127.0.0.1"
_PORT = "19530"


def test_retrieval_augmented_generation():
    # runs long, requires OpenAI API key and local milvus server
    # uses ada-002 for embeddings
    pdf_path = "test/bc_summary.pdf"
    with open(pdf_path, "rb") as f:
        doc_bytes = f.read()
    assert isinstance(doc_bytes, bytes)

    reader = DocumentReader()
    doc = reader.document_from_pdf(doc_bytes)
    rag_agent = DocumentEmbedder(
        embedding_collection_name="openai_embedding_test",
        metadata_collection_name="openai_metadata_test",
    )
    rag_agent.connect(_HOST, _PORT)
    doc_id = rag_agent.save_document(doc)
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

    query = "What is BioCypher?"
    results = rag_agent.similarity_search(query)
    assert len(results) == 3
    assert all(["BioCypher" in result.page_content for result in results])

    docs = rag_agent.get_all_documents()
    cnt = len(docs)
    assert cnt > 0
    rag_agent.remove_document(doc_id)
    docs = rag_agent.get_all_documents()
    assert (cnt - 1) == len(docs)


def test_retrieval_augmented_generation_generic_api():
    # runs long, requires OpenAI API key and local milvus server
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
    )
    rag_agent.connect(_HOST, _PORT)

    doc_id = rag_agent.save_document(doc)
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

    query = "What is BioCypher?"
    results = rag_agent.similarity_search(query)
    assert len(results) == 3
    assert all(["BioCypher" in result.page_content for result in results])

    docs = rag_agent.get_all_documents()
    cnt = len(docs)
    assert cnt > 0
    rag_agent.remove_document(doc_id)
    docs = rag_agent.get_all_documents()
    assert (cnt - 1) == len(docs)


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


def test_split_by_characters():
    # requires OpenAI API key
    rag_agent = DocumentEmbedder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    check_document_splitter(rag_agent, "test/bc_summary.pdf", 195)
    check_document_splitter(rag_agent, "test/dcn.pdf", 246)
    check_document_splitter(rag_agent, "test/bc_summary.txt", 103)


def test_split_by_tokens_tiktoken():
    # requires OpenAI API key
    rag_agent = DocumentEmbedder(
        split_by_characters=False,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    check_document_splitter(rag_agent, "test/bc_summary.pdf", 46)
    check_document_splitter(rag_agent, "test/dcn.pdf", 69)
    check_document_splitter(rag_agent, "test/bc_summary.txt", 20)


def test_split_by_tokens_tokenizers():
    # requires OpenAI API key
    rag_agent = DocumentEmbedder(
        split_by_characters=False,
        model="bigscience/bloom",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    check_document_splitter(rag_agent, "test/bc_summary.pdf", 48)
    check_document_splitter(rag_agent, "test/dcn.pdf", 72)
    check_document_splitter(rag_agent, "test/bc_summary.txt", 21)
