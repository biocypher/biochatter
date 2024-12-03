import logging
import random
import uuid

from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

from .constants import MAX_AGENT_DESC_LENGTH

logger = logging.getLogger(__name__)

DOCUMENT_METADATA_COLLECTION_NAME = "DocumentMetadata1"
DOCUMENT_EMBEDDINGS_COLLECTION_NAME = "DocumentEmbeddings1"

METADATA_VECTOR_DIM = 2
METADATA_FIELDS = [
    "id",
    "name",
    "author",
    "title",
    "format",
    "subject",
    "creator",
    "producer",
    "creationDate",
    "modDate",
    "source",
]


def align_metadata(
    metadata: list[dict],
    isDeleted: bool | None = False,
) -> list[list]:
    """Ensure that specific metadata fields are present; if not provided, fill with
    "unknown". Also, add a random vector to each metadata item to simulate an
    embedding.

    Args:
    ----
        metadata (List[Dict]): List of metadata items

        isDeleted (Optional[bool], optional): Whether the document is deleted.
            Defaults to False.

    Returns:
    -------
        List[List]: List of metadata items, with each item being a list of
            metadata fields.

    """
    ret = []
    fields = METADATA_FIELDS.copy()
    fields.pop(0)
    for ix, k in enumerate(fields):
        ret.append([item[k] if k in item else "unknown" for item in metadata])

    ret.append(
        [[random.random() for _ in range(METADATA_VECTOR_DIM)] for _ in range(len(metadata))],
    )
    ret.append([isDeleted for _ in metadata])
    return ret


def align_embeddings(docs: list[Document], meta_id: int) -> list[Document]:
    """Ensure that the metadata id is present in each document.

    Args:
    ----
        docs (List[Document]): List of documents

        meta_id (int): Metadata id to assign to the documents

    Returns:
    -------
        List[Document]: List of documents, with each document having a metadata
            id.

    """
    ret = []
    for doc in docs:
        ret.append(
            Document(
                page_content=doc.page_content,
                metadata={"meta_id": meta_id},
            ),
        )
    return ret


def validate_connection_args(connection_args: dict | None = None):
    if connection_args is None:
        return {
            "host": "127.0.0.1",
            "port": "19530",
            "user": "",
            "password": "",
        }

    connection_args["user"] = connection_args.get("user", "")
    connection_args["password"] = connection_args.get("password", "")
    return connection_args


class VectorDatabaseAgentMilvus:
    """The VectorDatabaseAgentMilvus class manages vector databases in a connected
    host database. It manages an embedding collection
    `_col_embeddings:langchain.vectorstores.Milvus`, which is the main
    information on the embedded text fragments and the basis for similarity
    search, and a metadata collection `_col_metadata:pymilvus.Collection`, which
    stores the metadata of the embedded text fragments. A typical workflow
    includes the following operations:

    1. connect to a host using `connect()`
    2. get all documents in the active database using `get_all_documents()`
    3. save a number of fragments, usually from a specific document, using
        `store_embeddings()`
    4. do similarity search among all fragments of the currently active database
        using `similarity_search()`
    5. remove a document from the currently active database using
        `remove_document()`
    """

    def __init__(
        self,
        embedding_func: OpenAIEmbeddings,
        connection_args: dict | None = None,
        embedding_collection_name: str | None = None,
        metadata_collection_name: str | None = None,
    ):
        """Args:
        ----
            embedding_func OpenAIEmbeddings: Function used to embed the text

            connection_args Optional dict: args to connect Vector Database

            embedding_collection_name Optional str: exposed for test

            metadata_collection_name Optional str: exposed for test

        """
        self._embedding_func = embedding_func
        self._col_embeddings: Milvus | None = None
        self._col_metadata: Collection | None = None
        self._connection_args = validate_connection_args(connection_args)
        self._embedding_name = embedding_collection_name or DOCUMENT_EMBEDDINGS_COLLECTION_NAME
        self._metadata_name = metadata_collection_name or DOCUMENT_METADATA_COLLECTION_NAME

    def connect(self) -> None:
        """Connect to a host and read two document collections (the default names
        are `DocumentEmbeddings` and `DocumentMetadata`) in the currently active
        database (default database name is `default`); if those document
        collections don't exist, create the two collections.
        """
        self._connect(**self._connection_args)
        self._init_host()

    def _connect(self, host: str, port: str, user: str, password: str) -> None:
        self.alias = self._create_connection_alias(host, port, user, password)

    def _init_host(self) -> None:
        """Initialize host. Will read/create document collection inside currently
        active database.
        """
        self._create_collections()

    def _create_connection_alias(
        self,
        host: str,
        port: str,
        user: str,
        password: str,
    ) -> str:
        """Connect to host and create a connection alias for metadata collection
        using a random uuid.

        Args:
        ----
            host (str): host ip address
            port (str): host port

        Returns:
        -------
            str: connection alias

        """
        alias = uuid.uuid4().hex
        try:
            connections.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                alias=alias,
            )
            logger.debug(f"Created new connection using: {alias}")
            return alias
        except MilvusException as e:
            logger.error(f"Failed to create  new connection using: {alias}")
            raise e

    def _create_collections(self) -> None:
        """Create or load the embedding and metadata collections from the currently
        active database.
        """
        embedding_exists = utility.has_collection(
            self._embedding_name,
            using=self.alias,
        )
        meta_exists = utility.has_collection(
            self._metadata_name,
            using=self.alias,
        )

        if embedding_exists:
            self._load_embeddings_collection()
        else:
            self._create_embeddings_collection()

        if meta_exists:
            self._load_metadata_collection()
        else:
            self._create_metadata_collection()

        self._create_metadata_collection_index()
        self._col_metadata.load()

    def _load_embeddings_collection(self) -> None:
        """Load embeddings collection from currently active database."""
        try:
            self._col_embeddings = Milvus(
                embedding_function=self._embedding_func,
                collection_name=self._embedding_name,
                connection_args=self._connection_args,
            )
        except MilvusException as e:
            logger.error(
                f"Failed to load embeddings collection {self._embedding_name}.",
            )
            raise e

    def _create_embeddings_collection(self) -> None:
        """Create embedding collection.
        All fields: "meta_id", "vector"
        """
        try:
            self._col_embeddings = Milvus(
                embedding_function=self._embedding_func,
                collection_name=self._embedding_name,
                connection_args=self._connection_args,
            )
        except MilvusException as e:
            logger.error(
                f"Failed to create embeddings collection {self._embedding_name}",
            )
            raise e

    def _load_metadata_collection(self) -> None:
        """Load metadata collection from currently active database."""
        self._col_metadata = Collection(
            self._metadata_name,
            using=self.alias,
        )
        self._col_metadata.load()

    def _create_metadata_collection(self) -> None:
        """Create metadata collection.

        All fields: "id", "name", "author", "title", "format", "subject",
        "creator", "producer", "creationDate", "modDate", "source", "embedding",
        "isDeleted".

        As the vector database requires a vector field, we will create a fake
        vector "embedding". The field "isDeleted" is used to specify if the
        document is deleted.
        """
        MAX_LENGTH = 10000
        doc_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        doc_name = FieldSchema(
            name="name",
            dtype=DataType.VARCHAR,
            max_length=MAX_LENGTH,
        )
        doc_author = FieldSchema(
            name="author",
            dtype=DataType.VARCHAR,
            max_length=MAX_LENGTH,
        )
        doc_title = FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=MAX_LENGTH,
        )
        doc_format = FieldSchema(
            name="format",
            dtype=DataType.VARCHAR,
            max_length=255,
        )
        doc_subject = FieldSchema(
            name="subject",
            dtype=DataType.VARCHAR,
            max_length=MAX_LENGTH,
        )
        doc_creator = FieldSchema(
            name="creator",
            dtype=DataType.VARCHAR,
            max_length=MAX_LENGTH,
        )
        doc_producer = FieldSchema(
            name="producer",
            dtype=DataType.VARCHAR,
            max_length=MAX_LENGTH,
        )
        doc_creationDate = FieldSchema(
            name="creationDate",
            dtype=DataType.VARCHAR,
            max_length=1024,
        )
        doc_modDate = FieldSchema(
            name="modDate",
            dtype=DataType.VARCHAR,
            max_length=1024,
        )
        doc_source = FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=MAX_LENGTH,
        )
        embedding = FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=METADATA_VECTOR_DIM,
        )
        isDeleted = FieldSchema(
            name="isDeleted",
            dtype=DataType.BOOL,
        )
        fields = [
            doc_id,
            doc_name,
            doc_author,
            doc_title,
            doc_format,
            doc_subject,
            doc_creator,
            doc_producer,
            doc_creationDate,
            doc_modDate,
            doc_source,
            embedding,
            isDeleted,
        ]
        schema = CollectionSchema(fields=fields)
        try:
            self._col_metadata = Collection(
                name=self._metadata_name,
                schema=schema,
                using=self.alias,
            )
        except MilvusException as e:
            logger.error(f"Failed to create collection {self._metadata_name}")
            raise e

    def _create_metadata_collection_index(self) -> None:
        """Create index for metadata collection in currently active database."""
        if not isinstance(self._col_metadata, Collection) or len(self._col_metadata.indexes) > 0:
            return

        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }

        try:
            self._col_metadata.create_index(
                field_name="embedding",
                index_params=index_params,
                using=self.alias,
            )
        except MilvusException as e:
            logger.error(
                "Failed to create index for meta collection " f"{self._metadata_name}.",
            )
            raise e

    def _insert_data(self, documents: list[Document]) -> str:
        """Insert documents into the currently active database.

        Args:
        ----
            documents (List[Documents]): documents array, usually from
                DocumentReader.load_document, DocumentReader.document_from_pdf,
                DocumentReader.document_from_txt

        Returns:
        -------
            str: document id

        """
        if len(documents) == 0:
            return None
        metadata = [documents[0].metadata]
        aligned_metadata = align_metadata(metadata)
        try:
            result = self._col_metadata.insert(aligned_metadata)
            meta_id = str(result.primary_keys[0])
            self._col_metadata.flush()
        except MilvusException as e:
            logger.error("Failed to insert meta data")
            raise e
        aligned_docs = align_embeddings(documents, meta_id)
        try:
            # As we passed collection_name, documents will be added to existed collection
            self._col_embeddings = Milvus.from_documents(
                embedding=self._embedding_func,
                collection_name=self._embedding_name,
                connection_args=self._connection_args,
                documents=aligned_docs,
            )
        except MilvusException as e:
            logger.error(
                "Failed to insert data to embedding collection " f"{self._embedding_name}.",
            )
            raise e
        return meta_id

    def store_embeddings(self, documents: list[Document]) -> str:
        """Store documents in the currently active database.

        Args:
        ----
            documents (List[Documents]): documents array, usually from
                DocumentReader.load_document, DocumentReader.document_from_pdf,
                DocumentReader.document_from_txt

        Returns:
        -------
            str: document id

        """
        if len(documents) == 0:
            return None
        return self._insert_data(documents)

    def _build_embedding_search_expression(
        self,
        meta_ids: list[dict],
    ) -> str | None:
        """Build search expression for embedding collection. The generated
        expression follows the pattern: "meta_id in [{id1}, {id2}, ...]

        Args:
        ----
            meta_ids: the array of metadata id in metadata collection

        Returns:
        -------
            str: search expression or None

        """
        if len(meta_ids) == 0:
            return "meta_id in []"
        built_expr = """meta_id in ["""
        for item in meta_ids:
            id = f'"{item["id"]}",'
            built_expr += id
        built_expr = built_expr[:-1]
        built_expr += """]"""
        return built_expr

    def _join_embedding_and_metadata_results(
        self,
        result_embedding: list[Document],
        result_meta: list[dict],
    ) -> list[Document]:
        """Join the search results of embedding collection and results of metadata.

        Args:
        ----
            result_embedding (List[Document]): search result of embedding
                collection

            result_meta (List[Dict]): search result of metadata collection

        Returns:
        -------
            List[Document]: combined results like
                [{page_content: str, metadata: {...}}]

        """

        def _find_metadata_by_id(
            metadata: list[dict],
            id: str,
        ) -> dict | None:
            for d in metadata:
                if str(d["id"]) == id:
                    return d
            return None

        joined_docs = []
        for res in result_embedding:
            found = _find_metadata_by_id(result_meta, res.metadata["meta_id"])
            if found is None:  # discard
                logger.error(
                    f"Failed to join meta_id {res.metadata['meta_id']}",
                )
                continue
            joined_docs.append(
                Document(page_content=res.page_content, metadata=found),
            )
        return joined_docs

    @staticmethod
    def _build_meta_col_query_expr_for_all_documents(
        doc_ids: list[str] | None = None,
    ) -> str:
        """Build metadata collection query expression to obtain all documents.

        Args:
        ----
            doc_ids: the list of document ids (metadata ids), if thie argument is None,
                     that is, the query is to get all undeleted documents in metadata collection.
                     Otherwise, the query is to getr all undeleted documents form provided doc_ids

        Returns:
        -------
            query: str

        """
        expr = f"id in {doc_ids} and isDeleted == false" if doc_ids is not None else "isDeleted == false"
        return expr.replace('"', "").replace("'", "")

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        doc_ids: list[str] | None = None,
    ) -> list[Document]:
        """Perform similarity search insider the currently active database
        according to the input query.

        This method will:
        1. get all non-deleted meta_id and build to search expression for
            the currently active embedding collection
        2. do similarity search in the embedding collection
        3. combine metadata and embeddings

        Args:
        ----
            query (str): query string

            k (int): the number of results to return

            doc_ids (Optional[list[str]]): the list of document ids, do
                similarity search across the specified documents

        Returns:
        -------
            List[Document]: search results

        """
        result_metadata = []
        expr = VectorDatabaseAgentMilvus._build_meta_col_query_expr_for_all_documents(
            doc_ids,
        )
        result_metadata = self._col_metadata.query(
            expr=expr,
            output_fields=METADATA_FIELDS,
        )
        expr = self._build_embedding_search_expression(result_metadata)
        result_embedding = self._col_embeddings.similarity_search(
            query=query,
            k=k,
            expr=expr,
        )
        return self._join_embedding_and_metadata_results(
            result_embedding,
            result_metadata,
        )

    def remove_document(
        self,
        doc_id: str,
        doc_ids: list[str] | None = None,
    ) -> bool:
        """Remove the document include meta data and its embeddings.

        Args:
        ----
            doc_id (str): the document to be deleted

            doc_ids (Optional[list[str]]): the list of document ids, defines
                documents scope within which remove operation occurs.

        Returns:
        -------
            bool: True if the document is deleted, False otherwise

        """
        if not self._col_metadata:
            return False
        if doc_ids is not None and (len(doc_ids) == 0 or (len(doc_ids) > 0 and doc_id not in doc_ids)):
            return False
        try:
            expr = f"id in [{doc_id}]"
            res = self._col_metadata.query(
                expr=expr,
                output_fields=METADATA_FIELDS,
            )
            if len(res) == 0:
                return False
            del_res = self._col_metadata.delete(expr)
            self._col_metadata.flush()

            res = self._col_embeddings.col.query(f'meta_id in ["{doc_id}"]')
            if len(res) == 0:
                return True
            ids = [item["pk"] for item in res]
            embedding_expr = f"pk in {ids}"
            del_res = self._col_embeddings.col.delete(expr=embedding_expr)
            self._col_embeddings.col.flush()
            return True
        except MilvusException as e:
            logger.error(e)
            raise e

    def get_all_documents(
        self,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """Get all non-deleted documents from the currently active database.

        Args:
        ----
            doc_ids (List[str] optional): the list of document ids, defines
                documents scope within which the operation of obtaining all
                documents occurs

        Returns:
        -------
            List[Dict]: the metadata of all non-deleted documents in the form
                [{{id}, {author}, {source}, ...}]

        """
        try:
            expr = VectorDatabaseAgentMilvus._build_meta_col_query_expr_for_all_documents(
                doc_ids,
            )
            result_metadata = self._col_metadata.query(
                expr=expr,
                output_fields=METADATA_FIELDS,
            )
            return result_metadata
        except MilvusException as e:
            logger.error(e)
            raise e

    def get_description(self, doc_ids: list[str] | None = None):
        def get_name(meta: dict[str, str]):
            name_col = ["title", "name", "subject", "source"]
            for col in name_col:
                if meta[col] is not None and len(meta[col]) > 0:
                    return meta[col]
            return ""

        expr = VectorDatabaseAgentMilvus._build_meta_col_query_expr_for_all_documents(
            doc_ids,
        )
        result = self._col_metadata.query(
            expr=expr,
            output_fields=METADATA_FIELDS,
        )
        names = list(map(get_name, result))
        names_set = set(names)
        desc = f"This vector store contains the following articles: {names_set}"
        return desc[:MAX_AGENT_DESC_LENGTH]
