from biochatter.vectorstore import (
    DocumentEmbedder,
    DocumentReader,
    Document,
)

import os

print(os.getcwd())


def test_document_summariser():
    # runs long, requires API key and local milvus server
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
