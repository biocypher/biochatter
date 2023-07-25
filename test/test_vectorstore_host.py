
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from pymilvus import utility, Collection

from biochatter.vectorstore import DocumentReader
from biochatter.vectorstore_host import VectorDatabaseHost, base64_to_string, string_to_base64

'''
This test need OPENAI API KEY and local milvus server
'''

_HOST = "127.0.0.1"
_PORT = "19530"
collection_names = []
def prepare_collection(doc_path: str) -> Milvus:
    pdf_path = doc_path
    docname = string_to_base64(os.path.basename(pdf_path))
    reader = DocumentReader()
    docs = reader.load_document(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap = 10,
        separators = [" ", ",", "\n"]
    )
    splitted_docs = text_splitter.split_documents(docs)
    db = Milvus.from_documents(
        documents=splitted_docs, 
        embedding=OpenAIEmbeddings(),
        connection_args={"host": _HOST, "port": _PORT}
    )
    
    utility.create_alias(db.collection_name, f"{docname}_{db.collection_name}")
    collection_names.append(db.collection_name)
def setup_module(module):
    prepare_collection("test/bc_summary.pdf")
    

def teardown_module(module):
    for name in collection_names:
        col = Collection(name)
        col.drop()


def test_connect_host():
    # require local milvus server
    dbHost = VectorDatabaseHost()
    dbHost.connect(host=_HOST, port=_PORT)
    collections = dbHost.get_collections()
    assert not collections is None
    cur_collection = dbHost.get_current_collection()
    assert cur_collection is None

def test_set_current_collection():
    dbHost = VectorDatabaseHost()
    dbHost.connect(host=_HOST, port=_PORT)
    collections = dbHost.get_collections()
    assert len(collections) > 0
    dbHost.set_current_collection(collections[0].collection_name)
    cur_collection = dbHost.get_current_collection()
    assert cur_collection is not None
    assert cur_collection.collection_name == collections[0].collection_name
