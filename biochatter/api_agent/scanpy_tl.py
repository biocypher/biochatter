"""Module for interacting with the `scanpy` API for data tools (`tl`)."""

from typing import TYPE_CHECKING

from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel

from .abc import BaseQueryBuilder
from .generate_pydantic_classes_from_module import generate_pydantic_classes

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

SCANPY_QUERY_PROMPT = """
You are a world class algorithm for creating queries in structured formats. Your task is to use the scanpy python package
to provide the user with the appropriate function call to answer their question. You focus on the scanpy.tl module, which has 
the following overview:
Any transformation of the data matrix that is not *preprocessing*. In contrast to a *preprocessing* function, a *tool* usually adds an easily interpretable annotation to the data matrix, which can then be visualized with a corresponding plotting function.

### Embeddings

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pp.pca
   tl.tsne
   tl.umap
   tl.draw_graph
   tl.diffmap
```

Compute densities on embeddings.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.embedding_density
```

### Clustering and trajectory inference

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.leiden
   tl.louvain
   tl.dendrogram
   tl.dpt
   tl.paga
```

### Data integration

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.ingest
```

### Marker genes

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.rank_genes_groups
   tl.filter_rank_genes_groups
   tl.marker_gene_overlap
```

### Gene scores, Cell cycle

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.score_genes
   tl.score_genes_cell_cycle
```

### Simulations

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.sim

```
"""


class ScanpyTLQueryBuilder(BaseQueryBuilder):
    """A class for building an ScanpyTLQuery object."""

    def create_runnable(
        self,
        query_parameters: BaseModel,
        conversation: "Conversation",
    ):
        pass

    def parameterise_query(
        self,
        question: str,
        conversation: "Conversation",
    ):
        """Generate an ScanpyTLQuery object.

        Generate a ScanpyTLQuery object based on the given question, prompt,
        and BioChatter conversation. Uses a Pydantic model to define the API
        fields. Using langchains .bind_tools method to allow the LLM to parameterise
        the function call, based on the functions available in thescanpy.tl module.

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
        import scanpy as sc

        module = sc.tl
        generated_classes = generate_pydantic_classes(module)
        llm = conversation.chat
        llm_with_tools = llm.bind_tools(generated_classes)
        query = [
            ("system", "You're an expert data scientist"),
            ("human", {question}),
        ]
        chain = llm_with_tools | PydanticToolsParser(tools=generated_classes)
        return chain.invoke(query)
