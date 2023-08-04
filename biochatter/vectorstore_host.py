import logging, uuid, random
from typing import List, Optional, Tuple, Dict
from pymilvus import (
    MilvusException,
    connections,
    Collection,
    utility,
    DataType,
    FieldSchema,
    CollectionSchema
)
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

DOCUMENT_METADATA_COLLECTION_NAME = "DocumentMetadata"
DOCUMENT_EMBEDDINGS_COLLECTION_NAME = "DocumentEmbeddings"

METADATA_VECTOR_DIM = 2
METADATA_FIELDS = [
    "id", "name", "author", "title", "format", "subject",
    "creator", "producer", "creationDate", "modDate", "source",
]
def align_metadata(metadata: List[Dict], isDeleted: Optional[bool]=False):
    ret = []
    fields = METADATA_FIELDS.copy()
    fields.pop(0)
    for ix, k in enumerate(fields):
        ret.append([item[k] if k in item else "unknown" for item in metadata])
    
    ret.append([[random.random() for _ in range(METADATA_VECTOR_DIM)] for _ in range(len(metadata))])
    ret.append([isDeleted for _ in metadata])
    return ret

def align_embeddings(docs: List[Document], meta_id: int) -> List[Document]:
    ret = []
    for doc in docs:
        ret.append(Document(page_content=doc.page_content, metadata={"meta_id": meta_id}))
    return ret

class VectorDatabaseHostMilvus:
    """
    VectorDatabaseHostMilvus is to manage vector databases in a host:
    In it, we will manage embedding collection `_col_embeddings:langchain.vectorstores.Milvus` and 
    metadata collection `_col_metadata:pymilvus.Collection`.
    A typical workflow is:
    1. connect to a host, connect()
    2. get all documents in the host, get_all_documents()
    3. save document, store_embeddings()
    4. do similarity search, similarity_search()
    5. remove a document, remove_document()
    """
    def __init__(
        self,
        embedding_func: Optional[OpenAIEmbeddings] = None,
        connection_args: Optional[Dict] = None,
        embedding_collection_name: Optional[str] = None,
        metadata_collection_name: Optional[str] = None
    ):
        self._embedding_func = embedding_func
        self._col_embeddings: Optional[Milvus] = None
        self._col_metadata: Optional[Collection] = None
        self._connection_args = connection_args or {
            "host": "127.0.0.1",
            "port": "19530"
        }
        self._embedding_name = embedding_collection_name or DOCUMENT_EMBEDDINGS_COLLECTION_NAME
        self._metadata_name = metadata_collection_name or DOCUMENT_METADATA_COLLECTION_NAME
    
    def connect(self, host: str, port: str) -> None:
        """
        Connects to a host and read two document collections (the default names are: DocumentMetadata 
        and DocumentEmbeddings), if document collections don't exist, we will create the two collections.
        Parameters:
            host str: host ip address
            port str: host port
        """
        self._connect(host, port)
        self._init_host()

    def _connect(self, host: str, port: str) -> None:
        self._connection_args = {"host": host, "port": port}
        self.alias = self._create_connection_alias(host, port)
    
    def _init_host(self) -> None:
        """
        Initialize host. Will read/create document collection
        """
        self._create_collections()


    def _create_connection_alias(self, host: str, port: str) -> str:
        """
        connect to host and create a connection alias for metadata collection
        Returns:
            str: connection alias
        """
        alias = uuid.uuid4().hex
        try:
            connections.connect(host=host, port=port, alias=alias)
            logger.debug(f"Created new connection using: {alias}")
            return alias
        except MilvusException as e:
            logger.error(f"Failed to create  new connection using: {alias}")
            raise e

    def _create_collections(self) -> None:
        '''
        This function handles creating or loading the embedding and metadata collections
        '''
        embedding_existed = utility.has_collection(self._embedding_name, using=self.alias)
        meta_existed = utility.has_collection(self._metadata_name, using=self.alias)
        if meta_existed:
            self._col_metadata = Collection(self._metadata_name, using=self.alias)
            self._col_metadata.load()
        else:
            self._create_metadata_collection_directly()
        if embedding_existed:
            self._col_embeddings = Milvus(
                embedding_function=self._embedding_func,
                collection_name=self._embedding_name,
                connection_args=self._connection_args
            )
        else:
            self._create_embeddings_collection_directly()
        
        self._create_metadata_collection_index()
        self._col_metadata.load()
        

    def _create_metadata_collection_index(self) -> None:
        """
        create index for metadata collection
        """
        if not isinstance(self._col_metadata, Collection) or \
            len(self._col_metadata.indexes) > 0:
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
                using=self.alias
            )
        except MilvusException as e:
            logger.error(f"Failed to create index for meta collection {self._metadata_name}")
            raise e
        
    def _create_metadata_collection_directly(self) -> None:
        """
        Create metadata collection.
        All fields: "id", "name", "author", "title", "format", "subject", "creator", "producer", 
        "creationDate", "modDate", "source", "embedding", "isDeleted"
        As vector database requires one vector field, we will create a fake vector "embedding".
        Field "isDeleted" is used to specify if the document is deleted.
        """
        doc_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        doc_name = FieldSchema(
            name="name",
            dtype=DataType.VARCHAR,
            max_length=255
        )
        doc_author = FieldSchema(
            name="author",
            dtype=DataType.VARCHAR,
            max_length=255
        )
        doc_title = FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=1000
        )
        doc_format = FieldSchema(
            name="format",
            dtype=DataType.VARCHAR,
            max_length=255
        )
        doc_subject = FieldSchema(
            name="subject",
            dtype=DataType.VARCHAR,
            max_length=255
        )
        doc_creator = FieldSchema(
            name="creator",
            dtype=DataType.VARCHAR,
            max_length=255
        )
        doc_producer = FieldSchema(
            name="producer",
            dtype=DataType.VARCHAR,
            max_length=255
        )
        doc_creationDate = FieldSchema(
            name="creationDate",
            dtype=DataType.VARCHAR,
            max_length=64
        )
        doc_modDate = FieldSchema(
            name="modDate",
            dtype=DataType.VARCHAR,
            max_length=64
        )
        doc_source = FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=1000
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
            doc_id,      doc_name,    doc_author,   doc_title,        doc_format,
            doc_subject, doc_creator, doc_producer, doc_creationDate, doc_modDate,
            doc_source,  embedding,   isDeleted,
        ]
        schema = CollectionSchema(fields=fields)
        try:
            self._col_metadata = Collection(name=self._metadata_name, schema=schema, using=self.alias)
        except MilvusException as e:
            logger.error(f"Failed to create collection {self._metadata_name}")
            raise e
            
    def _create_embeddings_collection_directly(self) -> None:
        """
        Create embedding collection.
        All fields: "meta_id", "vector"
        """
        try:
            self._col_embeddings = Milvus(
                embedding_function=self._embedding_func,
                collection_name=self._embedding_name,
                connection_args=self._connection_args
            )
        except MilvusException as e:
            logger.error(f"Failed to create embeddings collection {self._embedding_name}")
            raise e
        

    def _insert_data(self, documents: List[Document]) -> str:
        """
        Insert documents to database.
        Parameters:
            documents List[Documents]: documents array, usually from DocumentReader.load_document
                                       DocumentReader.document_from_pdf, DocumentReader.document_from_txt
        Returns:
            str: document id
        """
        if (len(documents) == 0):
            return None
        metadata = [documents[0].metadata]
        aligned_metadata = align_metadata(metadata)
        try:
            result = self._col_metadata.insert(aligned_metadata)
            meta_id = str(result.primary_keys[0])
        except MilvusException as e:
            logger.error(f"Failed to insert meta data")
            raise e
        aligned_docs = align_embeddings(documents, meta_id)
        try:
            self._col_embeddings = Milvus.from_documents(
                embedding=self._embedding_func,
                collection_name=self._embedding_name,
                connection_args=self._connection_args,
                documents=aligned_docs
            )
        except MilvusException as e:
            logger.error(f"Failed to insert data to embedding collection {self._embedding_name}")
            raise e
        return meta_id

    def store_embeddings(self, documents: List[Document]) -> str:
        """
        Store documents to database.
        Arguments;
            documents List[Documents]: documents array, usually from DocumentReader.load_document
                                       DocumentReader.document_from_pdf, DocumentReader.document_from_txt
        Returns:
            str: document id
        """
        if len(documents) == 0:
            return
        return self._insert_data(documents)
    def _build_embedding_search_expression(self, meta_ids: List[Dict]) -> Optional[str]:
        """
        Build search expression for embedding collection. The generated expression is like: "meta_id in [{id1}, {id2}, ...]
        Parameters:
            meta_ids: the array of metadata id in metadata collection
        Returns:
            str: search expression or None
        """
        if len(meta_ids) == 0:
            return None
        built_expr = '''meta_id in ['''
        for item in meta_ids:
            id = f'"{item["id"]}",'
            built_expr += id
        built_expr = built_expr[:-1]
        built_expr += ''']'''
        return built_expr
    
    def _join_embedding_and_metadata_results(self, result_embedding: List[Document], result_meta: List[Dict]) -> List[Document]:
        """
        Join the search results of embedding collection and results of metadata.
        Parameters:
            result_embedding List[Document]: search result of embedding collection
            result_meta List[Dict]: search result of metadata collection
        Returns:
            List[Document]: combined results like [{page_content: str, metadata: {...}}]
        """
        def _find_metadata_by_id(metadata: List[Dict], id: str) -> Optional[Dict]:
            for d in metadata:
                if str(d["id"]) == id:
                    return d
            return None

        joined_docs = []
        for res in result_embedding:
            found = _find_metadata_by_id(result_meta, res.metadata["meta_id"])
            if found is None: # discard
                logger.error(f"Failed to join meta_id {res.metadata['meta_id']}")
                continue
            joined_docs.append(
                Document(page_content=res.page_content, metadata=found)
            )
        return joined_docs
    
    def similarity_search(
        self, query: str, k: int=3
    ) -> List[Document]:
        """
        Similarity search according to query
        This method will:
        1). get all non-deleted meta_id and build to search expression for embedding collection
        2). do similarity search in embedding collection
        3). combine metadata and embeddings
        Parameters:
            query str: query string
            k: the number of result
        Returns:
            List[Document]: search results
        """
        result_metadata = self._col_metadata.query(
            expr="isDeleted == false"
        )
        expr = self._build_embedding_search_expression(result_metadata)
        result_embedding = self._col_embeddings.similarity_search(
            query=query,
            k=k,
            expr=expr
        )
        return self._join_embedding_and_metadata_results(result_embedding, result_metadata)
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Mark the document as deleted
        Parameters:
            doc_id str: the document to be deleted
        Returns: 
            bool: if the document is deleted
        """
        if not self._col_metadata:
            return False
        try:
            expr = f'id in [{doc_id}]'
            res = self._col_metadata.query(expr=expr, output_fields=METADATA_FIELDS)
            if len(res) == 0:
                return False
            del_res = self._col_metadata.delete(expr)
            aligned_metadata = align_metadata(res, True)
            self._col_metadata.insert(aligned_metadata)
            self._col_metadata.flush()
            return True
        except MilvusException as e:
            logger.error(e)
            raise e
        
    def get_all_documents(self) -> List[Dict]:
        """
        Returns:
            List[Dict]: the metadata of all non-deleted documents, [{{id}, {author}, {source}, ...}]
        """
        try:
            result_metadata = self._col_metadata.query(
                expr="isDeleted == false",
                output_fields=METADATA_FIELDS
            )
            return result_metadata
        except MilvusException as e:
            logger.error(e)
            raise e


