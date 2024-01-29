import os, pytest, uuid
from unittest.mock import patch

# from langchain.schema import Document
# from pymilvus import utility, Collection, connections
# from langchain.embeddings import OpenAIEmbeddings
from biochatter.vectorstore_agent import VectorDatabaseAgentMilvus

from .mock_langchain import OpenAIEmbeddings, Document, Milvus
from .mock_pymilvus import (
    connections,
    utility,
    Collection,
    DataType,
    FieldSchema,
    CollectionSchema,
)


"""
This test needs OPENAI_API_KEY in the environment and a local milvus server. 
"""

# setup milvus connection
if os.getenv("DEVCONTAINER"):
    _HOST = "milvus-standalone"
else:
    _HOST = "127.0.0.1"
_PORT = "19530"

NAME_SUFFIX = uuid.uuid4().hex

EMBEDDING_NAME = f"DocumentEmbeddingTest_{NAME_SUFFIX}"
METADATA_NAME = f"DocumentMetadataTest_{NAME_SUFFIX}"

mocked_dcn_pdf_splitted_texts = [
    Document(
        page_content=(
            "arXiv:1706.05966v1  [cs.LG]  19 Jun 2017\nDeep Counterfactual "
            "Networks with Propensity-Dropout\nAhmed M. Alaa, 1 Michael Weisz, "
            "2 Mihaela van der Schaar 1 2 3\nAbstract\nWe propose a novel "
            "approach for inferring the\nindividualized causal effects of a "
            "treatment (in-\ntervention)"
        ),
        metadata={
            "format": "PDF 1.4",
            "title": "Deep Counterfactual Networks with Propensity-Dropout",
            "author": "Ahmed M. Alaa,, Michael Weisz,, Mihaela van der Schaar",
            "subject": "Proceedings of the International Conference on Machine Learning 2017",
            "creator": "LaTeX with hyperref package",
            "producer": "dvips + GPL Ghostscript GIT PRERELEASE 9.08",
            "creationDate": "D:20170619204746-04'00'",
            "modDate": "D:20170619204746-04'00'",
            "source": "test/dcn.pdf",
        },
    ),
    Document(
        page_content=(
            "to update the weights of the\nshared layers and the respective "
            "outcome-speciﬁc\nlayers. Experiments conducted on data based "
            "on\na real-world observational study show that our al-\ngorithm "
            "outperforms the state-of-the-art.\n1. Introduction\nThe problem "
            "of inferring individualized treatment effects"
        ),
        metadata={
            "format": "PDF 1.4",
            "title": "Deep Counterfactual Networks with Propensity-Dropout",
            "author": "Ahmed M. Alaa,, Michael Weisz,, Mihaela van der Schaar",
            "subject": "Proceedings of the International Conference on Machine Learning 2017",
            "creator": "LaTeX with hyperref package",
            "producer": "dvips + GPL Ghostscript GIT PRERELEASE 9.08",
            "creationDate": "D:20170619204746-04'00'",
            "modDate": "D:20170619204746-04'00'",
            "source": "test/dcn.pdf",
        },
    ),
]
mocked_bc_summary_pdf_splitted_texts = [
    Document(
        page_content=(
            "Democratising knowledge representation with BioCypher\nSebastian "
            "Lobentanzer1,*, Patrick Aloy2,3, Jan Baumbach4, Balazs Bohar5,6, "
            "Pornpimol\nCharoentong8,9, Katharina Danhauser10, Tunca "
            "Doğan11,12, Johann Dreo13,14, Ian Dunham15,16,\nAdrià "
            "Fernandez-Torras2, Benjamin M. Gyori17"
        ),
        metadata={
            "format": "PDF 1.4",
            "title": "Accepted_Version_Editor_edits_May_2023",
            "producer": "Skia/PDF m115 Google Docs Renderer",
            "source": "test/bc_summary.pdf",
        },
    ),
    Document(
        page_content=(
            "i Estudis Avançats (ICREA), Barcelona, Catalonia, Spain\n4 "
            "Institute for Computational Systems Biology, University of "
            "Hamburg, Germany\n5 Earlham Institute, Norwich, UK\n6 Biological "
            "Research Centre, Szeged, Hungary\n7\n8 Centre for Quantitative "
            "Analysis of Molecular and Cellular Biosystems (Bioquant),\n"
            "Heidelberg University, Im Neuenheimer Feld 267, 69120, "
            "Heidelberg, Germany\n9 Department of Medical Oncology, National "
            "Centre for Tumour Diseases (NCT), Heidelberg\nUniversity Hospital "
            "(UKHD), Im Neuenheimer Feld 460, 69120, Heidelberg, Germany\n10 "
            "Department of Pediatrics, Dr. von Hauner Children’s Hospital, "
            "University Hospital, LMU\nMunich, Germany\n11 Biological Data "
            "Science Lab, Department of Computer Engineering, Hacettepe "
            "University,\nAnkara, Turkey\n12 Department of Bioinformatics, "
            "Graduate School of Health Sciences, Hacettepe University,\n"
            "Ankara, Turkey\n13 Computational Systems Biomedicine Lab, "
            "Department of Computational Biology, Institut\nPasteur, "
            "Université Paris Cité, Paris, France\n14"
        ),
        metadata={
            "format": "PDF 1.4",
            "title": "Accepted_Version_Editor_edits_May_2023",
            "producer": "Skia/PDF m115 Google Docs Renderer",
            "source": "test/bc_summary.pdf",
        },
    ),
]
mocked_bc_summary_txt_splitted_texts = [
    Document(
        page_content=(
            "Biomedical data are amassed at an ever-increasing rate, and "
            "machine learning tools that use prior knowledge in combination "
            "with biomedical big data are gaining much traction 1,2. Knowledge "
            "graphs (KGs) are rapidly becoming the dominant form of knowledge "
            "representation. KGs are d"
        ),
        metadata={"source": "test/bc_summary.txt"},
    )
]


@pytest.fixture
def dbHost():
    with patch(
        "biochatter.vectorstore_agent.OpenAIEmbeddings", OpenAIEmbeddings
    ), patch("biochatter.vectorstore_agent.Document", Document), patch(
        "biochatter.vectorstore_agent.Milvus", Milvus
    ), patch(
        "biochatter.vectorstore_agent.connections", connections
    ), patch(
        "biochatter.vectorstore_agent.utility", utility
    ), patch(
        "biochatter.vectorstore_agent.Collection", Collection
    ), patch(
        "biochatter.vectorstore_agent.DataType", DataType
    ), patch(
        "biochatter.vectorstore_agent.FieldSchema", FieldSchema
    ), patch(
        "biochatter.vectorstore_agent.CollectionSchema", CollectionSchema
    ):
        # create dbHost
        dbHost = VectorDatabaseAgentMilvus(
            embedding_func=OpenAIEmbeddings(),
            connection_args={"host": _HOST, "port": _PORT},
            embedding_collection_name=EMBEDDING_NAME,
            metadata_collection_name=METADATA_NAME,
        )
        dbHost.connect()
        assert dbHost._col_embeddings is not None
        assert dbHost._col_metadata is not None

        # save embeddings
        doc_id = dbHost.store_embeddings(mocked_dcn_pdf_splitted_texts)
        assert doc_id is not None
        doc_id = dbHost.store_embeddings(mocked_bc_summary_pdf_splitted_texts)
        assert doc_id is not None
        doc_id = dbHost.store_embeddings(mocked_bc_summary_txt_splitted_texts)
        assert doc_id is not None
        yield dbHost

        # clean up
        alias = uuid.uuid4().hex
        connections.connect(host=_HOST, port=_PORT, alias=alias)
        if utility.has_collection(EMBEDDING_NAME, using=alias):
            col = Collection(EMBEDDING_NAME, using=alias)
            col.drop()
        if utility.has_collection(METADATA_NAME, using=alias):
            col = Collection(METADATA_NAME, using=alias)
            col.drop()


def test_similarity_search(dbHost):
    results = dbHost.similarity_search(
        query="What is Deep Counterfactual Networks?",
        k=3,
    )
    assert len(results) > 0


def test_remove_document(dbHost):
    docs = dbHost.get_all_documents()
    if len(docs) == 0:
        return
    cnt = len(docs)
    res = dbHost.remove_document(docs[0]["id"])
    assert res
    assert (cnt - 1) == len(dbHost.get_all_documents())


def test_build_meta_col_query_expr_for_all_documents():
    data = [
        [[], "id in [] and isDeleted == false"],
        [["1234", "5678"], "id in [1234, 5678] and isDeleted == false"],
        [["1234", "5678"], "id in [1234, 5678] and isDeleted == false"],
        [None, "isDeleted == false"],
        [
            [12345678901234, 43210987654321],
            "id in [12345678901234, 43210987654321] and isDeleted == false",
        ],
    ]
    for test_data in data:
        expr = VectorDatabaseAgentMilvus._build_meta_col_query_expr_for_all_documents(
            test_data[0]
        )
        assert expr == test_data[1]
