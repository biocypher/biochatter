import uuid
from typing import Any
from unittest.mock import Mock


class Document:
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict
    """
    Arbitrary metadata about the page content (e.g., source, relationships to
    other documents, etc.).
    """

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class OpenAIEmbeddings:
    pass


class Milvus:
    def __init__(
        self,
        embedding_function: OpenAIEmbeddings | None = None,
        collection_name: str | None = "default",
        connection_args: dict[str, Any] | None = None,
        documents: list[Document] | None = None,
    ) -> None:
        self.documents: dict[list[Document]] = {} if documents is None else {uuid.uuid4().hex: documents}
        self.col = Mock()
        self.col.query = Mock()
        self.col.query.return_value = []

    def connect(self):
        pass

    def store_embeddings(self, docs: list[Document]) -> str:
        id = uuid.uuid4().hex
        self.documents[id] = docs
        return id

    def get_all_documents(self) -> list[list[Document]]:
        return [self.documents[id] for id in self.documents.keys()]

    def remove_document(self, id: str):
        if id in self.documents.keys():
            self.documents.pop(id)

    def similarity_search(
        self,
        query: str,
        k: int,
        expr: str | None = None,
    ) -> list[Document]:
        from random import randint

        total_docs: list[Document] = [doc for id in self.documents.keys() for doc in self.documents[id]]
        if len(total_docs) < k:
            return [doc for doc in total_docs]
        ret_docs = []
        for i in range(k):
            random_ix = randint(0, len(total_docs) - 1)
            ret_docs.append(total_docs[random_ix])
        return ret_docs

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Any,
        **kwargs: Any,
    ):
        return Milvus(embedding_function=embedding, documents=documents)
