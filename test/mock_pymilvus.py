import uuid
from typing import Any

from pymilvus import DataType, FieldSchema


def has_collection(name, using: str | None = None):
    return True


class CollectionSchema:
    def __init__(self, fields: list[FieldSchema]):
        pass


class CollectionRecord:
    def __init__(self, key: str) -> None:
        self.primary_keys = [key]


class Collection:
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

    def similarity_search(self, query: str, k: int, expr: str | None = None):
        pass

    def load(self):
        pass

    def drop(self):
        pass

    @property
    def indexes(self):
        return []

    def create_index(
        self,
        field_name: str,
        index_params: dict[str, Any],
        using: str,
    ):
        pass

    def insert(
        self,
        data: list | dict,
        partition_name: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ):
        id = uuid.uuid4().hex
        self.data[id] = data
        return CollectionRecord(id)


class Connections:
    def connect(
        self,
        host: str,
        port: str,
        alias: str | None = None,
        user: str | None = "",
        password: str | None = "",
    ):
        pass


connections = Connections()


class Utility:
    def has_collection(self, name: str, using: str | None = None):
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
