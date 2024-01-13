from typing import Dict, Optional, List, Union, Any
import uuid

from pymilvus import DataType, FieldSchema


def has_collection(name, using: Optional[str] = None):
    return True


class CollectionSchema(object):
    def __init__(self, fields: List[FieldSchema]):
        pass


class CollectionRecord(object):
    def __init__(self, key: str) -> None:
        self.primary_keys = [key]


class Collection(object):
    def __init__(
        self,
        name: str,
        schema: CollectionSchema = None,
        using: str = "default",
        **kwargs,
    ):
        self.data: dict[str, list] = {}

    def query(self, expr: str, **kwargs):
        return [{"id": id} for id in self.data.keys()]

    def delete(self, expr: str):
        # remove one record
        if len(self.data.items()) == 0:
            return
        id = list(self.data.keys())[0]
        self.data.pop(id)

    def flush(self):
        pass

    def similarity_search(self, query: str, k: int, expr: Optional[str] = None):
        pass

    def load(self):
        pass

    def drop(self):
        pass

    @property
    def indexes(self):
        return []

    def create_index(
        self, field_name: str, index_params: Dict[str, Any], using: str
    ):
        pass

    def insert(
        self,
        data: Union[List, Dict],
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        id = uuid.uuid4().hex
        self.data[id] = data
        return CollectionRecord(id)


class Connections(object):
    def connect(
        self, 
        host: str, 
        port: str, 
        alias: Optional[str] = None, 
        user: Optional[str]="", 
        password: Optional[str]=""
    ):
        pass


connections = Connections()


class Utility:
    def has_collection(self, name: str, using: Optional[str] = None):
        return True


utility = Utility()

__all__ = [
    "DataType",
    "FieldSchema",
    "CollectionSchema",
    "Collection",
    "utility",
    "connections",
]
