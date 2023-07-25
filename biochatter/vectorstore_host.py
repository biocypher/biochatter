import logging
from typing import List, Optional, Tuple
import re
import base64
from pymilvus import MilvusException, connections, Collection, utility, FieldSchema
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings

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

class VectorDatabaseHost:
    def __init__(
        self,
        embeddings: Optional[OpenAIEmbeddings]=None,
        connection_args: Optional[dict]=None
    ):
        self.collections: List[VectorCollection] = []
        self.vector_db: Optional[Milvus] = None
        self.embeddings: Optional[OpenAIEmbeddings] = embeddings
        self.connection_args = connection_args or {
            "host": "127.0.0.1",
            "port": "19530",
        }

    def connect(self, host, port):
        self.connection_args = {"host": host, "port": port}
        self._init_host(host, port)

    def _init_host(self, host, port):
        try:
            connections.connect(host=host, port=port)
        except MilvusException as e:
            logger.error(f"Failed to create connection to {host}:{port}")
            raise e
        
        self.collections = []
        self.vector_db = None
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
                self.collections.append(
                    VectorCollection(
                        col_name=col_name,
                        doc_name=docname,
                        text_field=text_field
                    )
                )                    
                break
    
    def get_collections(self):
        return self.collections
    def get_current_collection(self) -> Optional[VectorCollection]:
        if not self.vector_db:
            return None
        return self._find_collection(self.vector_db.collection_name)

    def _find_collection(self, collection_name: str) -> Optional[VectorCollection]:
        for col in self.collections:
            if col.collection_name == collection_name:
                return col
        return None
    def set_current_collection(self, collection_name: str):
        col = self._find_collection(collection_name)
        if not col:
            logger.error(f"Unknown collection name: {collection_name}")
            return
        if self.vector_db and self.vector_db.collection_name == collection_name:
            return
        self.vector_db = Milvus(self.embeddings, self.connection_args, collection_name=collection_name, text_field=col.text_field)
        
    



                

