
from biochatter.vectorstore_host import VectorDatabaseHost

_HOST = "127.0.0.1"
_PORT = "19530"

def test_connect_host():
    # require local milvus server
    dbHost = VectorDatabaseHost()
    dbHost.connect(host=_HOST, port=_PORT)
    collections = dbHost.get_collections()
    assert not collections is None
    cur_collection = dbHost.get_current_collection()
    assert cur_collection is None
