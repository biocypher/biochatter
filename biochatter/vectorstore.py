"""Module for handling document embedding, storage and retrieval.

This module provides classes for splitting documents into chunks, embedding them
using various LLM providers (OpenAI, Xinference, Ollama), storing them in vector
databases, and retrieving relevant passages through similarity search.
"""

import fitz  # this is PyMuPDF (PyPI pymupdf package, not fitz)
import openai
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import (
    OllamaEmbeddings,
    XinferenceEmbeddings,
)
from langchain_community.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from transformers import GPT2TokenizerFast

from biochatter.vectorstore_agent import VectorDatabaseAgentMilvus


class DocumentEmbedder:
    """Handle retrieval-augmented generation (RAG) functionality of BioChatter.

    This class is responsible for:
    - Splitting text documents into manageable chunks
    - Embedding these chunks using various LLM providers
    - Storing embeddings in vector databases
    - Performing similarity searches for retrieval
    """

    def __init__(
        self,
        used: bool = False,
        online: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        split_by_characters: bool = True,
        separators: list | None = None,
        n_results: int = 3,
        model: str | None = "text-embedding-ada-002",
        vector_db_vendor: str | None = None,
        connection_args: dict | None = None,
        embedding_collection_name: str | None = None,
        metadata_collection_name: str | None = None,
        base_url: str | None = None,
        embeddings: OpenAIEmbeddings | XinferenceEmbeddings | OllamaEmbeddings | AzureOpenAIEmbeddings | None = None,
        documentids_workspace: list[str] | None = None,
    ) -> None:
        r"""Initialize the DocumentEmbedder with the specified configuration.

        Args:
        ----
            used (bool, optional): whether RAG has been used (frontend setting).
                Defaults to False.

            online (bool, optional): whether we are running the frontend online.
                Defaults to False.

            chunk_size (int, optional): size of chunks to split text into.
                Defaults to 1000.

            chunk_overlap (int, optional): overlap between chunks.
                Defaults to 0.

            split_by_characters (bool, optional): whether to split by characters
                or tokens. Defaults to True.

            separators (Optional[list], optional): list of separators to use
                when splitting by characters. Defaults to [" ", ",", "\n"].

            n_results (int, optional): number of results to return from
                similarity search. Defaults to 3.

            model (Optional[str], optional): name of model to use for
                embeddings. Defaults to 'text-embedding-ada-002'.

            vector_db_vendor (Optional[str], optional): name of vector database
                to use. Defaults to Milvus.

            connection_args (Optional[dict], optional): arguments to pass to
                vector database connection. Defaults to None.

            base_url (Optional[str], optional): base url of OpenAI API.

            embeddings (Optional[OpenAIEmbeddings | XinferenceEmbeddings],
                optional): Embeddings object to use. Defaults to OpenAI.

            documentids_workspace (Optional[List[str]], optional): a list of
                document IDs that defines the scope within which RAG operations
                (remove, similarity search, and get all) occur. Defaults to
                None, which means the operations will be performed across all
                documents in the database.

        """
        self.used = used
        self.online = online
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [" ", ",", "\n"]
        self.n_results = n_results
        self.split_by_characters = split_by_characters
        self.model_name = model

        # TODO API Key handling to central config?
        if base_url:
            openai.api_base = base_url

        self.embeddings = embeddings

        # connection arguments
        self.connection_args = connection_args or {
            "host": "127.0.0.1",
            "port": "19530",
        }
        self.embedding_collection_name = embedding_collection_name
        self.metadata_collection_name = metadata_collection_name
        self.documentids_workspace = documentids_workspace

        # TODO: vector db selection
        self.vector_db_vendor = vector_db_vendor or "milvus"
        # instantiate VectorDatabaseHost
        self.database_host = None
        self._init_database_host()

    def _set_embeddings(
        self,
        embeddings: (OpenAIEmbeddings | XinferenceEmbeddings | OllamaEmbeddings | AzureOpenAIEmbeddings),
    ) -> None:
        print("setting embedder")
        self.embeddings = embeddings

    def _init_database_host(self) -> None:
        if self.vector_db_vendor == "milvus":
            self.database_host = VectorDatabaseAgentMilvus(
                embedding_func=self.embeddings,
                connection_args=self.connection_args,
                embedding_collection_name=self.embedding_collection_name,
                metadata_collection_name=self.metadata_collection_name,
            )
        else:
            raise NotImplementedError(self.vector_db_vendor)

    def set_chunk_size(self, chunk_size: int) -> None:
        """Set the chunk size for the text splitter."""
        self.chunk_size = chunk_size

    def set_chunk_overlap(self, chunk_overlap: int) -> None:
        """Set the chunk overlap for the text splitter."""
        self.chunk_overlap = chunk_overlap

    def set_separators(self, separators: list) -> None:
        """Set the separators for the text splitter."""
        self.separators = separators

    def _characters_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def _tokens_splitter(self) -> RecursiveCharacterTextSplitter:
        DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
        HUGGINGFACE_MODELS = ["bigscience/bloom"]
        if self.model_name and self.model_name in HUGGINGFACE_MODELS:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
            )

        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="",
            model_name=(DEFAULT_OPENAI_MODEL if not self.model_name else self.model_name),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def _text_splitter(self) -> RecursiveCharacterTextSplitter:
        return self._characters_splitter() if self.split_by_characters else self._tokens_splitter()

    def save_document(self, doc: list[Document]) -> str:
        """Save a list of documents to the vector database.

        Args:
        ----
            doc (List[Document]): document content, read with `DocumentReader`
                functions `load_document()`, `document_from_pdf()`, or
                `document_from_txt()`

        Returns:
        -------
            str: document id, which can be used to remove an uploaded document
                with `remove_document()`

        """
        splitted = self._split_document(doc)
        return self._store_embeddings(splitted)

    def _split_document(self, document: list[Document]) -> list[Document]:
        """Split a document into chunks."""
        text_splitter = self._text_splitter()
        return text_splitter.split_documents(document)

    def _store_embeddings(self, doc: list[Document]) -> str:
        """Store embeddings for a list of documents."""
        return self.database_host.store_embeddings(documents=doc)

    def connect(self) -> None:
        """Connect to the vector database."""
        self.database_host.connect()

    def get_all_documents(self) -> list[dict]:
        """Get all documents from the vector database."""
        return self.database_host.get_all_documents(
            doc_ids=self.documentids_workspace,
        )

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the vector database."""
        return self.database_host.remove_document(
            doc_id,
            self.documentids_workspace,
        )


class XinferenceDocumentEmbedder(DocumentEmbedder):
    """Extension of the DocumentEmbedder class to Xinference."""

    def __init__(
        self,
        used: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        split_by_characters: bool = True,
        separators: list | None = None,
        n_results: int = 3,
        model: str | None = "auto",
        vector_db_vendor: str | None = None,
        connection_args: dict | None = None,
        embedding_collection_name: str | None = None,
        metadata_collection_name: str | None = None,
        base_url: str | None = None,
        documentids_workspace: list[str] | None = None,
    ) -> None:
        """Initialize with the specified configuration.

        Args:
        ----
            used (bool, optional): whether RAG has been used (frontend setting).

            chunk_size (int, optional): size of chunks to split text into.

            chunk_overlap (int, optional): overlap between chunks.

            split_by_characters (bool, optional): whether to split by characters
                or tokens.

            separators (Optional[list], optional): list of separators to use
                when splitting by characters.

            n_results (int, optional): number of results to return from
                similarity search.

            model (Optional[str], optional): name of model to use for
                embeddings. Can be "auto" to use the first available model.

            vector_db_vendor (Optional[str], optional): name of vector database
                to use.

            connection_args (Optional[dict], optional): arguments to pass to
                vector database connection.

            embedding_collection_name (Optional[str], optional): name of
                collection to store embeddings in.

            metadata_collection_name (Optional[str], optional): name of
                collection to store metadata in.

            base_url (Optional[str], optional): base url of Xinference API.

            documentids_workspace (Optional[List[str]], optional): a list of
                document IDs that defines the scope within which RAG operations
                (remove, similarity search, and get all) occur. Defaults to
                None, which means the operations will be performed across all
                documents in the database.

        """
        from xinference.client import Client

        self.model_name = model
        self.client = Client(base_url=base_url)
        self.models = {}
        self.load_models()

        if self.model_name is None or self.model_name == "auto":
            self.model_name = self.list_models_by_type("embedding")[0]
        self.model_uid = self.models[self.model_name]["id"]

        super().__init__(
            used=used,
            online=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_by_characters=split_by_characters,
            separators=separators,
            n_results=n_results,
            model=model,
            vector_db_vendor=vector_db_vendor,
            connection_args=connection_args,
            embedding_collection_name=embedding_collection_name,
            metadata_collection_name=metadata_collection_name,
            base_url=base_url,
            embeddings=XinferenceEmbeddings(
                server_url=base_url,
                model_uid=self.model_uid,
            ),
            documentids_workspace=documentids_workspace,
        )

    def load_models(self) -> None:
        """Get all models that are currently available.

        Connect to the Xinference server and write the running models to
        `self.models`.
        """
        for _id, model in self.client.list_models().items():
            model["id"] = _id
            self.models[model["model_name"]] = model

    def list_models_by_type(self, model_type: str) -> list[str]:
        """Return all models of a certain type.

        Connect to the Xinference server and return all models of a certain
        type.

        Args:
        ----
            model_type (str): type of model to list (e.g. "embedding", "chat")

        Returns:
        -------
            List[str]: list of model names

        """
        names = []
        for name, model in self.models.items():
            if "model_ability" in model:
                if model_type in model["model_ability"]:
                    names.append(name)
            elif model["model_type"] == model_type:
                names.append(name)
        return names


class OllamaDocumentEmbedder(DocumentEmbedder):
    """Extension of the DocumentEmbedder class to Ollama."""

    def __init__(
        self,
        used: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        split_by_characters: bool = True,
        separators: list | None = None,
        n_results: int = 3,
        model: str | None = "nomic-embed-text",
        vector_db_vendor: str | None = None,
        connection_args: dict | None = None,
        embedding_collection_name: str | None = None,
        metadata_collection_name: str | None = None,
        base_url: str | None = None,
        documentids_workspace: list[str] | None = None,
    ) -> None:
        """Initialize with the specified configuration.

        Args:
        ----
            used (bool, optional): whether RAG has been used (frontend setting).

            chunk_size (int, optional): size of chunks to split text into.

            chunk_overlap (int, optional): overlap between chunks.

            split_by_characters (bool, optional): whether to split by characters
                or tokens.

            separators (Optional[list], optional): list of separators to use
                when splitting by characters.

            n_results (int, optional): number of results to return from
                similarity search.

            model (Optional[str], optional): name of model to use for
                embeddings. Can be "auto" to use the first available model.

            vector_db_vendor (Optional[str], optional): name of vector database
                to use.

            connection_args (Optional[dict], optional): arguments to pass to
                vector database connection.

            embedding_collection_name (Optional[str], optional): name of
                collection to store embeddings in.

            metadata_collection_name (Optional[str], optional): name of
                collection to store metadata in.

            base_url (Optional[str], optional): base url of Xinference API.

            documentids_workspace (Optional[List[str]], optional): a list of
                document IDs that defines the scope within which RAG operations
                (remove, similarity search, and get all) occur. Defaults to
                None, which means the operations will be performed across all
                documents in the database.

        """
        from langchain_community.embeddings import OllamaEmbeddings

        self.model_name = model

        super().__init__(
            used=used,
            online=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_by_characters=split_by_characters,
            separators=separators,
            n_results=n_results,
            model=model,
            vector_db_vendor=vector_db_vendor,
            connection_args=connection_args,
            embedding_collection_name=embedding_collection_name,
            metadata_collection_name=metadata_collection_name,
            base_url=base_url,
            embeddings=OllamaEmbeddings(
                base_url=base_url,
                model=self.model_name,
            ),
            documentids_workspace=documentids_workspace,
        )


class DocumentReader:
    """Class for reading documents from various sources."""

    def load_document(self, path: str) -> list[Document]:
        """Load a document from a path; accept txt and pdf files.

        Txt files are loaded as-is, pdf files are converted to text using
        `fitz`.

        Args:
        ----
            path (str): path to document

        Returns:
        -------
            List[Document]: list of documents

        Raises:
        ------
            ValueError: If file extension is not supported

        """
        if path.endswith(".txt"):
            loader = TextLoader(path)
            return loader.load()

        if path.endswith(".pdf"):
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()

            meta = {k: v for k, v in doc.metadata.items() if v}
            meta.update({"source": path})

            return [
                Document(
                    page_content=text,
                    metadata=meta,
                ),
            ]

        err_msg = f"Unsupported file extension in {path}. File must be .txt or .pdf"
        raise ValueError(err_msg)

    def document_from_pdf(self, pdf: bytes) -> list[Document]:
        """Return a list of Documents from a pdf file byte representation.

        Receive a byte representation of a pdf file and return a list of
        Documents with metadata.

        Args:
        ----
            pdf (bytes): byte representation of pdf file

        Returns:
        -------
            List[Document]: list of documents

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
            ),
        ]

    def document_from_txt(self, txt: bytes) -> list[Document]:
        """Return a list of Documents from a txt file byte representation.

        Receive a byte representation of a txt file and return a list of
        Documents with metadata.

        Args:
        ----
            txt (bytes): byte representation of txt file

        Returns:
        -------
            List[Document]: list of documents

        """
        meta = {"source": "txt"}
        return [
            Document(
                page_content=txt,
                metadata=meta,
            ),
        ]
