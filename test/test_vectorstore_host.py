import os
import pytest
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from pymilvus import utility, Collection

from biochatter.vectorstore import DocumentReader
from biochatter.vectorstore_host import (
    VectorDatabaseHostMilvus,
    base64_to_string,
    string_to_base64,
)

"""
This test need OPENAI API KEY and local milvus server
"""

_HOST = "127.0.0.1"
_PORT = "19530"
collection_names = []


def prepare_splitted_documents(
    doc_path: str, chunk_size: int = 1000, chunk_overlap: int = 20
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


def setup_module(module):
    pdf_path = "test/bc_summary.pdf"
    splitted_docs = prepare_splitted_documents(pdf_path)
    docname = string_to_base64(os.path.basename(pdf_path))
    db = Milvus.from_documents(
        documents=splitted_docs,
        embedding=OpenAIEmbeddings(),
        connection_args={"host": _HOST, "port": _PORT},
    )
    utility.create_alias(db.collection_name, f"{docname}_{db.collection_name}")
    collection_names.append(db.collection_name)


def teardown_module(module):
    for name in collection_names:
        col = Collection(name)
        col.drop()


def test_connect_host():
    # require local milvus server
    dbHost = VectorDatabaseHostMilvus()
    dbHost.connect(host=_HOST, port=_PORT)
    collections = dbHost.collections
    assert not collections is None
    assert len(collections) > 0


def test_store_embeddings_and_similarity_search():
    dbHost = VectorDatabaseHostMilvus(embeddings=OpenAIEmbeddings())
    dbHost.connect(host=_HOST, port=_PORT)
    pdf_path = "test/dcn.pdf"
    splitted_docs = prepare_splitted_documents(pdf_path)
    vector_collection = dbHost.store_embedding("dcn.pdf", splitted_docs)

    results = dbHost.similarity_search(
        collection_name=vector_collection["collection_name"],
        query="What is Deep Counterfactual Networks?",
        k=3,
    )
    assert len(results) > 0
    assert all(
        [
            "Deep Counterfactual Networks" in item.page_content
            for item in results
        ]
    )
    cnt = len(dbHost.collections)
    dbHost.drop_collection(vector_collection["collection_name"])
    assert (cnt - 1) == len(dbHost.collections)
