import os, pytest, uuid
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from pymilvus import utility, Collection, connections

from biochatter.vectorstore import DocumentReader
from biochatter.vectorstore_host import VectorDatabaseHostMilvus

"""
This test needs OPENAI_API_KEY in the environment and a local milvus server. 
"""

# setup milvus connection
if os.getenv("DEVCONTAINER"):
    _HOST = "milvus-standalone"
else:
    _HOST = "127.0.0.1"
_PORT = "19530"

NAME_SUFFIX = uuid.uuid4().hex

EMBEDDING_NAME = f"DocumentEmbeddingTest_{NAME_SUFFIX}"
METADATA_NAME = f"DocumentMetadataTest_{NAME_SUFFIX}"


def prepare_splitted_documents(
    doc_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 20,
) -> List[Document]:
    pdf_path = doc_path
    reader = DocumentReader()
    docs = reader.load_document(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[" ", ",", "\n"],
    )
    return text_splitter.split_documents(docs)


@pytest.fixture
def dbHost():
    # create dbHost
    dbHost = VectorDatabaseHostMilvus(
        embedding_func=OpenAIEmbeddings(),
        connection_args={"host": _HOST, "port": _PORT},
        embedding_collection_name=EMBEDDING_NAME,
        metadata_collection_name=METADATA_NAME
    )
    dbHost.connect(_HOST, _PORT)
    assert dbHost._col_embeddings is not None
    assert dbHost._col_metadata is not None

    # save embeddings
    splitted_docs = prepare_splitted_documents("test/dcn.pdf")
    doc_id = dbHost.store_embeddings(splitted_docs)
    assert doc_id is not None
    splitted_docs = prepare_splitted_documents("test/bc_summary.pdf")
    doc_id = dbHost.store_embeddings(splitted_docs)
    assert doc_id is not None
    splitted_docs = prepare_splitted_documents("test/bc_summary.txt")
    doc_id = dbHost.store_embeddings(splitted_docs)
    assert doc_id is not None
    yield dbHost

    # clean up
    alias = uuid.uuid4().hex
    connections.connect(host=_HOST, port=_PORT, alias=alias)
    if utility.has_collection(EMBEDDING_NAME, using=alias):
        col = Collection(EMBEDDING_NAME, using=alias)
        col.drop()
    if utility.has_collection(METADATA_NAME, using=alias):
        col = Collection(METADATA_NAME, using=alias)
        col.drop()

def test_similarity_search(dbHost):    
    results = dbHost.similarity_search(
        query="What is Deep Counterfactual Networks?",
        k=3,
    )
    assert len(results) > 0


def test_remove_document(dbHost):    
    docs = dbHost.get_all_documents()
    if len(docs) == 0:
        return
    cnt = len(docs)
    res = dbHost.remove_document(docs[0]["id"])
    assert res
    assert (cnt - 1) == len(dbHost.get_all_documents())
