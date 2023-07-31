# ChatGSE document summarisation functionality
# split text
# connect to vector db
# do similarity search
# return x closes matches for in-context learning

from typing import List, Optional, Dict

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Milvus

import fitz  # this is PyMuPDF (PyPI pymupdf package, not fitz)
from transformers import GPT2TokenizerFast

from biochatter.vectorstore_host import VectorDatabaseHostMilvus


class DocumentEmbedder:
    def __init__(
        self,
        use_prompt: bool = True,
        used: bool = False,
        online: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        split_by_characters: bool = True,
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
        self.split_by_characters = split_by_characters
        self.model_name = model

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
        # instantiate VectorDatabaseHost
        self.database_host = None
        self._init_database_host()

    def _init_database_host(self):
        if self.vector_db_vendor == "milvus":
            self.database_host = VectorDatabaseHostMilvus(
                embedding_func=self.embeddings, connection_args=self.connection_args
            )
        else:
            raise NotImplementedError(self.vector_db_vendor)

    def set_chunk_siue(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def set_chunk_overlap(self, chunk_overlap: int) -> None:
        self.chunk_overlap = chunk_overlap

    def set_separators(self, separators: list) -> None:
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
        else:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="",
                model_name=DEFAULT_OPENAI_MODEL
                if not self.model_name
                else self.model_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
            )

    def _text_splitter(self) -> RecursiveCharacterTextSplitter:
        return (
            self._characters_splitter()
            if self.split_by_characters
            else self._tokens_splitter()
        )

    def save_document(self, doc: List[Document]) -> str:
        '''
        This function saves document to the vector database
        Args:
            doc List[Document]: document content, read with DocumentReader load_document(), 
                or document_from_pdf(), document_from_txt()
        Returns:
            str: document id, which can be used to remove an uploaded document with remove_document()
        '''
        splitted = self._split_document(doc)
        return self._store_embeddings(splitted)

    def _split_document(self, document: List[Document]) -> List[Document]:
        text_splitter = self._text_splitter()
        return text_splitter.split_documents(document)

    def _store_embeddings(
        self, doc: List[Document]
    ) -> str:
        return self.database_host.store_embeddings(
            documents=doc
        )

    def similarity_search(
        self,
        query: str,
        k: int = 3
    ):
        """
        Returns top n closest matches to query from vector store.

        Args:
            query (str): query string

            k (int, optional): number of closest matches to return. Defaults to
            3.

            collection_name (str): search in collection {collection_name}

        """
        return self.database_host.similarity_search(
            query=query, k=k or self.n_results
        )

    def connect(self, host: str, port: str) -> None:
        self.database_host.connect(host, port)

    def get_all_documents(self) -> List[Dict]:
        return self.database_host.get_all_documents()

    def remove_document(self, doc_id: str) -> None:
        self.database_host.remove_document(doc_id)


class DocumentReader:
    def load_document(self, path: str) -> List[Document]:
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
            return loader.load()

        elif path.endswith(".pdf"):
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
                )
            ]

    def document_from_pdf(self, pdf: bytes) -> List[Document]:
        """
        Receive a byte representation of a pdf file and return a list of Documents
        with metadata.

        Args:
            pdf (bytes): byte representation of pdf file

        Returns:
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
            )
        ]

    def document_from_txt(self, txt: bytes) -> List[Document]:
        """
        Receive a byte representation of a txt file and return a list of Documents
        with metadata.

        Args:
            txt (bytes): byte representation of txt file

        Returns:
            List[Document]: list of documents
        """
        meta = {"source": "txt"}
        return [
            Document(
                page_content=txt,
                metadata=meta,
            )
        ]
