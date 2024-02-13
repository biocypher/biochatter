from unittest.mock import patch
from biochatter.vectorstore_agent import VectorDatabaseAgentMilvus
from biochatter.vectorstore import DocumentEmbedder, DocumentReader, Document
from biochatter.rag_agent import RagAgent, RagAgentModeEnum
import os
import pytest
from benchmark.conftest import calculate_test_score

# setup milvus connection
if os.getenv("DEVCONTAINER"):
    _HOST = "milvus-standalone"
else:
    _HOST = "127.0.0.1"
_PORT = "19530"

EMBEDDING_MODELS = [
    "text-embedding-ada-002",
]
CHUNK_SIZES = [50, 1000]

splitted_docs = [
    Document(
        page_content="Democratising",
        metadata={
            "format": "PDF 1.4",
            "title": "Accepted_Version_Editor_edits_May_2023",
            "producer": "Skia/PDF m115 Google Docs Renderer",
            "source": "pdf",
        },
    ),
    Document(
        page_content="",
        metadata={
            "format": "PDF 1.4",
            "title": "Accepted_Version_Editor_edits_May_2023",
            "producer": "Skia/PDF m115 Google Docs Renderer",
            "source": "pdf",
        },
    ),
    Document(
        page_content="",
        metadata={
            "format": "PDF 1.4",
            "title": "Accepted_Version_Editor_edits_May_2023",
            "producer": "Skia/PDF m115 Google Docs Renderer",
            "source": "pdf",
        },
    ),
]

search_docs = [
    Document(
        page_content="Democratising knowledge representation with BioCypher\n"
        "Sebastian Lobentanzer1,*, Patrick Aloy2,3, Jan Baumbach4, Balazs "
        "Bohar5,6, Pornpimol\nCharoentong8,9, Katharina Danhauser10, Tunca "
        "Doğan11,12, Johann Dreo13,14, Ian Dunham15,16,\nAdrià "
        "Fernandez-Torras2, Benjamin M. Gyori17, Michael",
        metadata={id: "1"},
    ),
    Document(
        page_content="BioCypher has been built with continuous consideration "
        "of the FAIR and TRUST",
        metadata={id: "1"},
    ),
    Document(
        page_content="adopting their own, arbitrary formats of representation. "
        "To our knowledge, no\nframework provides easy access to "
        "tate-of-the-art KGs to the average biomedical researcher,\na gap that "
        "BioCypher aims to fill. We demonstrate some key advantages of "
        "BioCypher by\ncase studies in Supplementary Note 5.\n5\nFigure ",
        metadata={id: "1"},
    ),
]

@pytest.mark.parametrize("model", EMBEDDING_MODELS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_retrieval_augmented_generation(model, chunk_size):
    pdf_path = "test/bc_summary.pdf"
    with open(pdf_path, "rb") as f:
        doc_bytes = f.read()

    reader = DocumentReader()
    doc = reader.document_from_pdf(doc_bytes)

    with patch(
        "biochatter.vectorstore.OpenAIEmbeddings"
    ) as mock_openaiembeddings, patch(
        "biochatter.vectorstore_agent.VectorDatabaseAgentMilvus"
    ) as mock_host_1, patch(
        "biochatter.vectorstore.VectorDatabaseAgentMilvus"
    ) as mock_host, patch(
        "biochatter.vectorstore.RecursiveCharacterTextSplitter"
    ) as mock_textsplitter:
        # mocking
        mock_textsplitter.from_huggingface_tokenizer.return_value = (
            mock_textsplitter()
        )
        mock_textsplitter.from_tiktoken_encoder.return_value = (
            mock_textsplitter()
        )
        mock_textsplitter.return_value.split_documents.return_value = (
            splitted_docs
        )
        mock_host.return_value.store_embeddings.return_value = "1"
        mock_host_1.return_value.similarity_search.return_value = search_docs

        doc_ids = []
        doc_embedder = DocumentEmbedder(
            model=model,
            chunk_size=chunk_size,
            connection_args={"host": _HOST, "port": _PORT},
        )
        rag_agent = RagAgent(
            use_prompt=True,
            mode=RagAgentModeEnum.VectorStore,
            model_name=model,
            connection_args={"host": _HOST, "port": _PORT},
            embedding_func=mock_openaiembeddings
        )
        doc_embedder.connect()
        doc_ids.append(doc_embedder.save_document(doc))

        query = "What is BioCypher?"
        results = rag_agent.generate_responses(query)
        correct = ["BioCypher" in result[0] for result in results]

        # delete embeddings
        [doc_embedder.database_host.remove_document(doc_id) for doc_id in doc_ids]

        # record sum in CSV file
        assert calculate_test_score(correct) == (3, 3)
