# Retrieval-Augmented Generation

To connect to a vector database for using semantic similarity search and
retrieval-augmented generation (RAG), we provide an implementation that connects
to a [Milvus](https://milvus.io) instance (local or remote).  These functions
are provided by the modules `vectorstore.py` (for performing embeddings) and `vectorstore_agent.py` (for maintaining the
connection and search).

This is implemented in the [ChatGSE](https://github.com/biocypher/ChatGSE)
Docker workflow and the BioChatter Docker compose found in this repository.  To
start Milvus on its own in these repositories, you can call `docker compose up
-d standalone` (`standalone` being the Milvus endpoint, which starts two other
services alongside it).

## Connecting

To connect to a vector DB host, we can use the corresponding class:

```python
from biochatter.vectorstore_agent import VectorDatabaseAgentMilvus

dbHost = VectorDatabaseAgentMilvus(
    embedding_func=OpenAIEmbeddings(),
    connection_args={"host": _HOST, "port": _PORT},
    embedding_collection_name=EMBEDDING_NAME,
    metadata_collection_name=METADATA_NAME
)
```

This establishes a connection with the vector database (using a host IP and
port) and uses two collections, one for the embeddings and one for the metadata
of embedded text (e.g. the title and authors of the paper that was embedded).

## Embedding documents

To embed text from documents, we use the LangChain and BioChatter
functionalities for processing and passing the text to the vector database.

```python
from biochatter.vectorstore import DocumentReader()
from langchain.text_splitter import RecursiveCharacterTextSplitter

# read and split document at `pdf_path`
reader = DocumentReader()
docs = reader.load_document(pdf_path)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=[" ", ",", "\n"],
)
split_text = text_splitter.split_documents(docs)

# embed and store embeddings in the connected vector DB
doc_id = dbHost.store_embeddings(splitted_docs)
```

The dbHost class takes care of calling an embedding model, storing the embedding
in the database, and returning a document ID that can be used to refer to the
stored document.

## Semantic search

To perform a semantic similarity search, all that is left to do is pass a
question or statement to the `dbHost`, which will be embedded and compared to
the present embeddings, returning a number `k` most similar text fragments.

```python
results = dbHost.similarity_search(
    query="Semantic similarity search query",
    k=3,
)
```

## Vectorstore management

Using the collections we created at setup, we can delete entries in the vector
database using their IDs. We can also return a list of all collected docs to
determine which we want to delete.

```python
docs = dbHost.get_all_documents()
res = dbHost.remove_document(docs[0]["id"])
```
