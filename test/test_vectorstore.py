from biochatter.vectorstore import (
    DocumentEmbedder,
    DocumentReader,
    Document,
)

import os
from unittest.mock import patch

print(os.getcwd())

# setup milvus connection
if os.getenv("DEVCONTAINER"):
    _HOST = "milvus-standalone"
else:
    _HOST = "127.0.0.1"
_PORT = "19530"

splitted_docs = [
    Document(
        page_content='Democratising',
        metadata={'format': 'PDF 1.4', 'title': 'Accepted_Version_Editor_edits_May_2023', 'producer': 'Skia/PDF m115 Google Docs Renderer', 'source': 'pdf'}
    ),
    Document(
        page_content='',
        metadata={'format': 'PDF 1.4', 'title': 'Accepted_Version_Editor_edits_May_2023', 'producer': 'Skia/PDF m115 Google Docs Renderer', 'source': 'pdf'}
    ),
    Document(
        page_content='',
        metadata={'format': 'PDF 1.4', 'title': 'Accepted_Version_Editor_edits_May_2023', 'producer': 'Skia/PDF m115 Google Docs Renderer', 'source': 'pdf'}
    )
]
    

@patch("biochatter.vectorstore.OpenAIEmbeddings")
@patch("biochatter.vectorstore.VectorDatabaseHostMilvus")
@patch("biochatter.vectorstore.RecursiveCharacterTextSplitter")
def test_document_summariser(mock_textsplitter, mock_host, mock_openaiembeddings):
    # runs long, requires OpenAI API key and local milvus server
    # uses ada-002 for embeddings
    search_docs = [
        Document(
            page_content='Democratising knowledge representation with BioCypher\nSebastian Lobentanzer1,*, Patrick Aloy2,3, Jan Baumbach4, Balazs Bohar5,6, Pornpimol\nCharoentong8,9, Katharina Danhauser10, Tunca Doğan11,12, Johann Dreo13,14, Ian Dunham15,16,\nAdrià Fernandez-Torras2, Benjamin M. Gyori17, Michael',
            metadata={id: '1'}
        ),
        Document(
            page_content='BioCypher has been built with continuous consideration of the FAIR and TRUST',
            metadata={id: '1'}
        ),
        Document(
            page_content='adopting their own, arbitrary formats of representation. To our knowledge, no\nframework provides easy access to state-of-the-art KGs to the average biomedical researcher,\na gap that BioCypher aims to fill. We demonstrate some key advantages of BioCypher by\ncase studies in Supplementary Note 5.\n5\nFigure ',
            metadata={id: '1'}
        )
    ]
    mock_textsplitter.from_huggingface_tokenizer.return_value = mock_textsplitter()
    mock_textsplitter.from_tiktoken_encoder.return_value = mock_textsplitter()
    mock_textsplitter.return_value.split_documents.return_value = splitted_docs
    mock_host.return_value.store_embeddings.return_value = '1'
    mock_host.return_value.similarity_search.return_value = search_docs


    pdf_path = "test/bc_summary.pdf"
    with open(pdf_path, "rb") as f:
        doc_bytes = f.read()
    assert isinstance(doc_bytes, bytes)

    reader = DocumentReader()
    doc = reader.document_from_pdf(doc_bytes)
    docsum = DocumentEmbedder()
    docsum.connect(_HOST, _PORT)
    doc_id = docsum.save_document(doc)
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

    query = "What is BioCypher?"
    results = docsum.similarity_search(query)
    assert len(results) == 3
    assert all(["BioCypher" in result.page_content for result in results])

    mock_host.return_value.get_all_documents.return_value = [{"id": "1"}, {"id": "2"}]
    docs = docsum.get_all_documents()
    cnt = len(docs)
    assert cnt > 0
    docsum.remove_document(doc_id)
    mock_host.return_value.get_all_documents.return_value = [{"id": "2"}]
    docs = docsum.get_all_documents()
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
    docsum: DocumentEmbedder,
    fn: str,
    expected_length: int,
):
    reader = DocumentReader()
    doc = reader.load_document(fn)
    splitted = docsum._split_document(doc)
    assert expected_length == len(splitted)

@patch("biochatter.vectorstore.OpenAIEmbeddings")
@patch('biochatter.vectorstore.VectorDatabaseHostMilvus')
@patch("biochatter.vectorstore.RecursiveCharacterTextSplitter")
def test_split_by_characters(mock_textsplitter, mock_host, mock_embeddings):
    # character splitter
    mock_textsplitter.from_huggingface_tokenizer.return_value = mock_textsplitter()
    mock_textsplitter.from_tiktoken_encoder.return_value = mock_textsplitter()
    mock_textsplitter.return_value.split_documents.return_value = splitted_docs
    docsum = DocumentEmbedder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    check_document_splitter(docsum, "test/bc_summary.pdf", len(splitted_docs))
    check_document_splitter(docsum, "test/dcn.pdf", len(splitted_docs))
    check_document_splitter(docsum, "test/bc_summary.txt", len(splitted_docs))
    
    # tiktoken
    docsum = DocumentEmbedder(
        split_by_characters=False,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    check_document_splitter(docsum, "test/bc_summary.pdf", len(splitted_docs))
    mock_textsplitter.from_tiktoken_encoder.assert_called_once()
    check_document_splitter(docsum, "test/dcn.pdf", len(splitted_docs))
    assert mock_textsplitter.from_tiktoken_encoder.call_count == 2
    check_document_splitter(docsum, "test/bc_summary.txt", len(splitted_docs))
    assert mock_textsplitter.from_tiktoken_encoder.call_count == 3

    # huggingface tokenizer
    docsum = DocumentEmbedder(
        split_by_characters=False,
        model="bigscience/bloom",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    check_document_splitter(docsum, "test/bc_summary.pdf", len(splitted_docs))
    mock_textsplitter.from_huggingface_tokenizer.assert_called_once()
    check_document_splitter(docsum, "test/dcn.pdf", len(splitted_docs))
    assert mock_textsplitter.from_huggingface_tokenizer.call_count == 2
    check_document_splitter(docsum, "test/bc_summary.txt", len(splitted_docs))
    assert mock_textsplitter.from_huggingface_tokenizer.call_count == 3

