"""Module for interacting with the `scanpy` API for plotting (`pl`)."""

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.pydantic_v1 import BaseModel, Field

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

from .abc import BaseQueryBuilder

SCANPY_PL_QUERY_PROMPT = """
You are a world class algorithm for creating queries in structured formats.
Your task is to use the Python API of `scanpy` to answer questions about the `scanpy` plotting module.
You should prefix the function calls with `scanpy.pl.`, for instance, for a scatter plot, you should use
`scanpy.pl.scatter`.

Extract information from the following documentation to answer the user's question:

The plotting module scanpy.pl largely parallels the tl.* and a few of the pp.* functions. For most tools and for some
preprocessing functions, you’ll find a plotting function with the same name.

Generic
pl.scatter
Scatter plot along observations or variables axes.
pl.heatmap
Heatmap of the expression values of genes.
pl.dotplot
Makes a dot plot of the expression values of var_names.
pl.tracksplot
In this type of plot each var_name is plotted as a filled line plot where the y values correspond to the var_name values
and x is each of the cells.
pl.violin
Violin plot.
pl.stacked_violin
Stacked violin plots.
pl.matrixplot
Creates a heatmap of the mean expression values per group of each var_names.
pl.clustermap
Hierarchically-clustered heatmap.
pl.ranking
Plot rankings.
pl.dendrogram
Plots a dendrogram of the categories defined in groupby.
Classes
These classes allow fine tuning of visual parameters.

pl.DotPlot
Allows the visualization of two values that are encoded as dot size and color.
pl.MatrixPlot
Allows the visualization of values using a color map.
pl.StackedViolin
Stacked violin plots.
Preprocessing
Methods for visualizing quality control and results of preprocessing functions.

pl.highest_expr_genes
Fraction of counts assigned to each gene over all cells.
pl.filter_genes_dispersion
Plot dispersions versus means for genes.
pl.highly_variable_genes
Plot dispersions or normalized variance versus means for genes.
pl.scrublet_score_distribution
Plot histogram of doublet scores for observed transcriptomes and simulated doublets.
Tools
Methods that extract and visualize tool-specific annotation in an AnnData object. For any method in module tl, there is a method with the same name in pl.

PCA

pl.pca
Scatter plot in PCA coordinates.
pl.pca_loadings
Rank genes according to contributions to PCs.
pl.pca_variance_ratio
Plot the variance ratio.
pl.pca_overview
Plot PCA results.
Embeddings

pl.tsne
Scatter plot in tSNE basis.
pl.umap
Scatter plot in UMAP basis.
pl.diffmap
Scatter plot in Diffusion Map basis.
pl.draw_graph
Scatter plot in graph-drawing basis.
pl.spatial
Scatter plot in spatial coordinates.
pl.embedding
Scatter plot for user specified embedding basis (e.g. umap, pca, etc).
Compute densities on embeddings.

pl.embedding_density
Plot the density of cells in an embedding (per condition).
Branching trajectories and pseudotime, clustering

Visualize clusters using one of the embedding methods passing color='louvain'.

pl.dpt_groups_pseudotime
Plot groups and pseudotime.
pl.dpt_timeseries
Heatmap of pseudotime series.
pl.paga
Plot the PAGA graph through thresholding low-connectivity edges.
pl.paga_path
Gene expression and annotation changes along paths in the abstracted graph.
pl.paga_compare
Scatter and PAGA graph side-by-side.
Marker genes

pl.rank_genes_groups
Plot ranking of genes.
pl.rank_genes_groups_violin
Plot ranking of genes for all tested comparisons.
pl.rank_genes_groups_stacked_violin
Plot ranking of genes using stacked_violin plot (see stacked_violin())
pl.rank_genes_groups_heatmap
Plot ranking of genes using heatmap plot (see heatmap())
pl.rank_genes_groups_dotplot
Plot ranking of genes using dotplot plot (see dotplot())
pl.rank_genes_groups_matrixplot
Plot ranking of genes using matrixplot plot (see matrixplot())
pl.rank_genes_groups_tracksplot
Plot ranking of genes using heatmap plot (see heatmap())
Simulations

pl.sim
Plot results of simulation.
"""


class ScanpyPlScatterQueryParameters(BaseModel):
    """Parameters for querying the scanpy `pl.scatter` API."""

    question_uuid: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )
    adata: str = Field(description="Annotated data matrix.",)
    x: str | None = Field(default=None, description="x coordinate.",)
    y: str | None = Field(default=None, description="y coordinate.",)
    color: str | tuple[float, ...] | list[str | tuple[float, ...]] | None = Field(
        default=None,
        description="Keys for annotations of observations/cells or variables/genes, or a hex color specification.",
    )
    use_raw: bool | None = Field(
        default=None,
        description="Whether to use raw attribute of adata. Defaults to True if .raw is present.",
    )
    layers: str | list[str] | None = Field(
        default=None,
        description="Layer(s) to use from adata's layers attribute.",
    )
    basis: str | None = Field(
        default=None,
        description="String that denotes a plotting tool that computed coordinates (e.g., 'pca', 'tsne', 'umap').",
    )
    sort_order: bool = Field(
        default=True,
        description="For continuous annotations used as color parameter, plot data points with higher values on top.",
    )
    groups: str | list[str] | None = Field(
        default=None,
        description="Restrict to specific categories in categorical observation annotation.",
    )
    projection: str = Field(
        default="2d",
        description="Projection of plot ('2d' or '3d').",
    )
    legend_loc: str | None = Field(
        default="right margin",
        description="Location of legend ('none', 'right margin', 'on data', etc.).",
    )
    size: int | float | None = Field(
        default=None,
        description="Point size. If None, automatically computed as 120000 / n_cells.",
    )
    color_map: str | None = Field(
        default=None,
        description="Color map to use for continuous variables (e.g., 'magma', 'viridis').",
    )
    title: str | list[str] | None = Field(
        default=None,
        description="Title for panels either as string or list of strings.",
    )
    show: bool | None = Field(
        default=None,
        description="Show the plot, do not return axis.",
    )
    save: str | bool | None = Field(
        default=None,
        description="If True or a str, save the figure. String is appended to default filename.",
    )

class ScanpyPlQueryBuilder(BaseQueryBuilder):
    """A class for building an ScanpyPlQuery object."""

    def create_runnable(
        self,
        query_parameters: "ScanpyPlScatterQueryParameters",
        conversation: "Conversation",
    ) -> Callable:
        """Create a runnable object for executing queries.

        Create runnable using the LangChain `create_structured_output_runnable`
        method.

        Args:
        ----
            query_parameters: A Pydantic data model that specifies the fields of
                the API that should be queried.

            conversation: A BioChatter conversation object.

        Returns:
        -------
            A Callable object that can execute the query.

        """
        return create_structured_output_runnable(
            output_schema=query_parameters,
            llm=conversation.chat,
            prompt=self.structured_output_prompt,
        )

    def parameterise_query(
        self,
        question: str,
        conversation: "Conversation",
    ) -> ScanpyPlScatterQueryParameters:
        """Generate an ScanpyPlQuery object.

        Generate a ScanpyPlQuery object based on the given question, prompt,
        and BioChatter conversation. Uses a Pydantic model to define the API
        fields.  Creates a runnable that can be invoked on LLMs that are
        qualified to parameterise functions.

        Args:
        ----
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                ScanpyPlQuery.

        Returns:
        -------
            ScanpyPlQueryParameters: the parameterised query object (Pydantic
                model)

        """
        runnable = self.create_runnable(
            query_parameters=ScanpyPlScatterQueryParameters,
            conversation=conversation,
        )
        scanpy_pl_call_obj = runnable.invoke(
            {
                "input": f"Answer:\n{question} based on:\n {SCANPY_PL_QUERY_PROMPT}",
            },
        )
        scanpy_pl_call_obj.question_uuid = str(uuid.uuid4())
        return scanpy_pl_call_obj
