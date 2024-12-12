"""Module for interacting with the `scanpy` API for data tools (`tl`)."""

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING

from langchain_core.output_parsers import PydanticToolsParser

from .abc import BaseAPIModel, BaseQueryBuilder
from .generate_pydantic_classes_from_module import generate_pydantic_classes

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation
from biochatter.llm_connect import Conversation

SCANPY_QUERY_PROMPT = """

You are a world class algorithm for creating queries in structured formats. Your
task is to use the scanpy python package to provide the user with the
appropriate function call to answer their question. You focus on the scanpy.tl
module, which has the following overview: Any transformation of the data matrix
that is not *preprocessing*. In contrast to a *preprocessing* function, a *tool*
usually adds an easily interpretable annotation to the data matrix, which can
then be visualized with a corresponding plotting function.

### Embeddings

   pp.pca
   tl.tsne
   tl.umap
   tl.draw_graph
   tl.diffmap

Compute densities on embeddings.

   tl.embedding_density

### Clustering and trajectory inference

   tl.leiden
   tl.louvain
   tl.dendrogram
   tl.dpt
   tl.paga

### Data integration

   tl.ingest

### Marker genes

   tl.rank_genes_groups
   tl.filter_rank_genes_groups
   tl.marker_gene_overlap

### Gene scores, Cell cycle

   tl.score_genes
   tl.score_genes_cell_cycle

### Simulations

   tl.sim
"""


class ScanpyTlQueryBuilder(BaseQueryBuilder):
    """A class for building an ScanpyTlQuery object."""

    def create_runnable(
        self,
        query_parameters: list["BaseAPIModel"],
        conversation: Conversation,
    ) -> Callable:
        runnable = conversation.chat.bind_tools(query_parameters, tool_choice="required")
        return runnable | PydanticToolsParser(tools=query_parameters)

    def parameterise_query(
        self,
        question: str,
        conversation: "Conversation",
        module: ModuleType,
        generated_classes=None,  # Allow external injection of classes for testing purposes
    ):
        """Generate an ScanpyTLQuery object.

        Generate a ScanpyTLQuery object based on the given question, prompt, and
        BioChatter conversation. Uses a Pydantic model to define the API fields.
        Using langchains .bind_tools method to allow the LLM to parameterise the
        function call, based on the functions available in thescanpy.tl module.

        Args:
        ----
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                BioToolsQuery.

        Returns:
        -------
            BioToolsQueryParameters: the parameterised query object (Pydantic
                model)

        """
        if generated_classes is None:
            tools = generate_pydantic_classes(module)

        runnable = self.create_runnable(
            conversation=conversation,
            query_parameters=tools,
        )

        query = [
            ("system", SCANPY_QUERY_PROMPT),
            ("human", question),
        ]

        return runnable.invoke(query)
