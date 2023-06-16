from biochatter.vectorstore import (
    DocumentEmbedder,
    Document,
    document_from_pdf,
    document_from_txt,
)

import os

print(os.getcwd())


def test_document_summariser():
    # runs long, requires API key and local milvus server
    # uses ada-002 for embeddings
    pdf_path = "test/bc_summary.pdf"
    with open(pdf_path, "rb") as f:
        document = f.read()
    assert isinstance(document, bytes)

    doc = document_from_pdf(document)
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
    docsum = DocumentEmbedder()
    text_path = "test/bc_summary.txt"
    document = docsum._load_document(text_path)
    assert isinstance(document, list)
    assert isinstance(document[0], Document)


def test_load_pdf():
    docsum = DocumentEmbedder()
    pdf_path = "test/bc_summary.pdf"
    document = docsum._load_document(pdf_path)
    assert isinstance(document, list)
    assert isinstance(document[0], Document)


def test_byte_txt():
    text_path = "test/bc_summary.txt"
    with open(text_path, "rb") as f:
        document = f.read()
    assert isinstance(document, bytes)

    doc = document_from_txt(document)
    assert isinstance(doc, list)
    assert isinstance(doc[0], Document)
    # do we want byte string or regular string?


def test_byte_pdf():
    pdf_path = "test/bc_summary.pdf"
    # open as type "application/pdf"
    with open(pdf_path, "rb") as f:
        document = f.read()
    assert isinstance(document, bytes)

    doc = document_from_pdf(document)
    assert isinstance(doc, list)
    assert isinstance(doc[0], Document)
    assert "numerous attempts at standardising KGs" in doc[0].page_content
