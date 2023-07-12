from biochatter.vectorstore import (
    DocumentEmbedder,
    DocumentReader,
    Document,
)

import os
print(os.getcwd())

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def test_document_summariser():
    # runs long, requires OpenAI API key and local milvus server
    # uses ada-002 for embeddings
    pdf_path = "test/bc_summary.pdf"
    with open(pdf_path, "rb") as f:
        doc_bytes = f.read()
    assert isinstance(doc_bytes, bytes)

    reader = DocumentReader()
    doc = reader.document_from_pdf(doc_bytes)
    docsum = DocumentEmbedder()
    docsum.set_document(doc)
    docsum.split_document()
    assert isinstance(docsum.split, list)
    assert isinstance(docsum.split[0], Document)

    docsum.store_embeddings()
    assert docsum.vector_db is not None

    query = "What is BioCypher?"
    results = docsum.similarity_search(query)
    assert len(results) == 3
    assert all(["BioCypher" in result.page_content for result in results])


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

def check_document_splitter(docsum: DocumentEmbedder, fn: str, expected_length: int):
    docsum._load_document(fn)
    docsum.split_document()
    assert expected_length == len(docsum.split)


def test_split_by_characters():
    docsum = DocumentEmbedder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    check_document_splitter(docsum, "test/bc_summary.pdf", 197)
    check_document_splitter(docsum, "test/dcn.pdf", 245)
    check_document_splitter(docsum, "test/bc_summary.txt", 104)

def test_split_by_tokens_tiktoken():
    docsum = DocumentEmbedder(
        split_by_characters=False,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    check_document_splitter(docsum, "test/bc_summary.pdf", 46)
    check_document_splitter(docsum, "test/dcn.pdf", 71)
    check_document_splitter(docsum, "test/bc_summary.txt", 40)

def test_split_by_tokens_tokenizers():
    docsum = DocumentEmbedder(
        split_by_characters=False, 
        model="bigscience/bloom",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    check_document_splitter(docsum, "test/bc_summary.pdf", 49)
    check_document_splitter(docsum, "test/dcn.pdf", 71)
    check_document_splitter(docsum, "test/bc_summary.txt", 44)

