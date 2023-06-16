# ChatGSE document summarisation functionality
# split text
# connect to vector db
# do similarity search
# return x closes matches for in-context learning

from typing import List, Optional

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Milvus

import fitz  # this is PyMuPDF (PyPI pymupdf package, not fitz)


class DocumentEmbedder:
    def __init__(
        self,
        use_prompt: bool = True,
        used: bool = False,
        online: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        document: Optional[Document] = None,
        separators: Optional[list] = None,
        n_results: int = 3,
        model: Optional[str] = None,
        vector_db_vendor: Optional[str] = None,
        connection_args: Optional[dict] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.use_prompt = use_prompt
        self.used = used
        self.online = online
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [" ", ",", "\n"]
        self.n_results = n_results

        # TODO: variable embeddings depending on model
        # for now, just use ada-002
        # TODO API Key handling to central config
        if not self.online:
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        else:
            self.embeddings = None

        # connection arguments
        self.connection_args = connection_args or {
            "host": "127.0.0.1",
            "port": "19530",
        }
        # TODO: vector db selection
        self.vector_db_vendor = vector_db_vendor or "milvus"
        self.vector_db = None

        self.document = None

    def set_chunk_siue(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def set_chunk_overlap(self, chunk_overlap: int) -> None:
        self.chunk_overlap = chunk_overlap

    def set_separators(self, separators: list) -> None:
        self.separators = separators

    def set_document(self, document: List[Document]) -> None:
        self.document = document

    def _load_document(self, path: str) -> None:
        """
        Loads a document from a path; accepts txt and pdf files. Txt files are
        loaded as-is, pdf files are converted to text using fitz.

        Args:
            path (str): path to document

        Returns:
            List[Document]: list of documents
        """
        if path.endswith(".txt"):
            loader = TextLoader(path)
            self.document = loader.load()
        elif path.endswith(".pdf"):
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()

            meta = {k: v for k, v in doc.metadata.items() if v}
            meta.update({"source": path})

            self.document = [
                Document(
                    page_content=text,
                    metadata=meta,
                )
            ]

    def split_document(self) -> None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        self.split = text_splitter.split_documents(self.document)

    def store_embeddings(self) -> None:
        if self.vector_db_vendor == "milvus":
            self.vector_db = Milvus.from_documents(
                documents=self.split,
                embedding=self.embeddings,
                connection_args=self.connection_args,
            )
        else:
            raise NotImplementedError(self.vector_db_vendor)

    def similarity_search(self, query: str, k: int = 3):
        """
        Returns top n closest matches to query from vector store.

        Args:
            query (str): query string

            k (int, optional): number of closest matches to return. Defaults to
            3.

        """
        if self.vector_db_vendor == "milvus":
            if not self.vector_db:
                raise ValueError("No vector store loaded")
            return self.vector_db.similarity_search(
                query=query, k=k or self.n_results
            )
        else:
            raise NotImplementedError(self.vector_db_vendor)


def document_from_pdf(pdf) -> List[Document]:
    """
    Receive a byte representation of a pdf file and return a list of Documents
    with metadata.
    """
    doc = fitz.open(stream=pdf, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    meta = {k: v for k, v in doc.metadata.items() if v}
    meta.update({"source": "pdf"})

    return [
        Document(
            page_content=text,
            metadata=meta,
        )
    ]


def document_from_txt(txt) -> List[Document]:
    """
    Receive a byte representation of a txt file and return a list of Documents
    with metadata.
    """
    meta = {"source": "txt"}
    return [
        Document(
            page_content=txt,
            metadata=meta,
        )
    ]
