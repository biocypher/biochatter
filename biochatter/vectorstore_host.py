import logging
from typing import List, Optional, Tuple, Dict
import re
import base64
from pymilvus import MilvusException, connections, Collection, utility, FieldSchema
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

def _grab_text_field_name(fields: List[FieldSchema]) -> str | None:
    prev_field = None
    for x in fields:
        if x.auto_id:
            return prev_field.name
        prev_field = x
    return None

def is_valid_alias(alias: str) -> Optional[Tuple]:
    # valud alias is {base64}_c{uuid4.hex}
    pattern = r"^([0-9a-zA-Z_]+)_c([a-f0-9]{32})$"
    match = re.fullmatch(pattern, alias)
    return None if not match else match.groups()

def string_to_base64(txt: str) -> str:
    converted = base64.urlsafe_b64encode(txt.encode("utf-8")).decode("utf-8")
    return converted.replace('-', 'a_a').replace("=", "b_b")

def base64_to_string(b64_txt: str) -> str:
    converted = b64_txt.replace('a_a', '-').replace("b_b", "=")
    return base64.urlsafe_b64decode(converted).decode("utf-8")


class VectorCollection:
    def __init__(
            self,
            col_name: str,
            doc_name: str,
            text_field: Optional[str]="text"
        ) -> None:
        self.collection_name = col_name
        self.document_name = doc_name
        self.text_field = text_field

class VectorDatabaseHostMilvus:
    def __init__(
        self,
        embeddings: Optional[OpenAIEmbeddings]=None,
        connection_args: Optional[dict]=None
    ):
        self._collections: List[VectorCollection] = []
        self._embeddings: Optional[OpenAIEmbeddings] = embeddings
        self._connection_args = connection_args or {
            "host": "127.0.0.1",
            "port": "19530",
        }

    def connect(self, host: str, port: str) -> None:
        self._connection_args = {"host": host, "port": port}
        self._init_host(host, port)

    def _init_host(self, host: str, port: str):
        try:
            connections.connect(host=host, port=port)
        except MilvusException as e:
            logger.error(f"Failed to create connection to {host}:{port}")
            raise e
        
        self._collections = []
        collections = utility.list_collections()
        for col_name in collections:
            col = Collection(col_name)
            if len(col.aliases) == 0:
                continue
            
            for alias in col.aliases:
                matched_grp = is_valid_alias(alias)
                if not matched_grp:
                    continue
                text_field = _grab_text_field_name(col.schema.fields)
                if not text_field:
                    continue
                docname = matched_grp[0]
                docname = base64_to_string(docname)
                self._collections.append(
                    VectorCollection(
                        col_name=col_name,
                        doc_name=docname,
                        text_field=text_field
                    )
                )                    
                break
    
    @property
    def collections(self) -> List[Dict[str, str]]:
        return [
            {"document_name": col.document_name, "collection_name": col.collection_name}
            for col in self._collections
        ]

    def _find_vector_collection(self, collection_name: str) -> Optional[VectorCollection]:
        for obj in self._collections:
            if obj.collection_name == collection_name:
                return obj
        return None
    def _remove_vector_collection(self, collection_name: str) -> bool:
        for obj in self._collections:
            if obj.collection_name == collection_name:
                self._collections.remove(obj)
                return True
        return False
    def store_embedding(
            self, 
            doc_name: str,
            documents: List[Document],
        ) -> Dict[str, str]:
        db = Milvus.from_documents(
            documents=documents,
            embedding=self._embeddings,
            connection_args=self._connection_args
        )
        encoded_doc_name = string_to_base64(doc_name)
        alias = f"{encoded_doc_name}_{db.collection_name}"
        utility.create_alias(db.collection_name, alias=alias)
        vector_collection = VectorCollection(
            col_name=db.collection_name,
            doc_name=doc_name,
            text_field=db.text_field
        )
        self._collections.append(vector_collection)
        return {
            "document_name": doc_name, 
            "collection_name": db.collection_name
        }
    
    def similarity_search(self, collection_name: str, query: str, k: int=3) -> List[Document]:
        vector_coll = self._find_vector_collection(collection_name)
        if not vector_coll:
            raise ValueError("No current collection loaded")
        try:
            db = Milvus(
                embedding_function=self._embeddings, 
                collection_name=collection_name, 
                connection_args=self._connection_args,
                text_field=vector_coll.text_field
            )
            return db.similarity_search(query=query, k=k)
        except MilvusException as e:
            logger.error(e)
            raise e
    
    def drop_collection(self, collection_name: str) -> None:
        if not self._remove_vector_collection(collection_name):
            return
        try:
            coll = Collection(collection_name)
            coll.drop()
        except MilvusException as e:
            logger.error(e)
            raise e




                

