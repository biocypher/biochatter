"""Module for interacting with the `scanpy` API for plotting (`pl`)."""

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

from biochatter.api_agent.base.agent_abc import BaseAPIModel, BaseQueryBuilder

SCANPY_PL_QUERY_PROMPT = """
You are a world class algorithm for creating queries in structured formats.
Your task is to use the Python API of `scanpy` to answer questions about the `scanpy` plotting module.
You should prefix the function calls with `scanpy.pl.`, for instance, for a scatter plot, you should use
`scanpy.pl.scatter`.

Extract information from the following documentation to answer the user's question:

The plotting module scanpy.pl largely parallels the tl.* and a few of the pp.* functions. For most tools and for some
preprocessing functions, you will find a plotting function with the same name.

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
Methods that extract and visualize tool-specific annotation in an AnnData object. For any method in module tl, there is
a method with the same name in pl.

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

    method_name: str = Field(
        default="sc.pl.scatter",
        description="The name of the method to call.",
    )
    question_uuid: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question.",
    )
    adata: str = Field(description="Annotated data matrix.")
    x: str | None = Field(default=None, description="x coordinate.")
    y: str | None = Field(default=None, description="y coordinate.")
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


### Embeddings
class ScanpyPlPcaQueryParameters(BaseModel):
    """Parameters for querying the scanpy `pl.pca` API."""

    method_name: str = Field(
        default="sc.pl.pca",
        description="The name of the method to call.",
    )
    question_uuid: str | None = Field(
        default=None,
        description="Unique identifier for the question.",
    )
    adata: str = Field(
        ...,
        description="Annotated data matrix.",
    )
    color: str | list[str] | None = Field(
        default=None,
        description="Keys for annotations of observations/cells or variables/genes.",
    )
    color_map: str | None = Field(
        default=None,
        description="String denoting matplotlib color map.",
    )


class ScanpyPlTsneQueryParameters(BaseModel):
    """Parameters for querying the Scanpy `pl.tsne` API."""

    method_name: str = Field(
        default="sc.pl.tsne",
        description="The name of the method to call.",
    )
    question_uuid: str | None = Field(
        default=None,
        description="Unique identifier for the question.",
    )
    adata: str = Field(
        ...,
        description="Annotated data matrix.",
    )
    color: str | list[str] | None = Field(
        default=None,
        description="Keys for annotations of observations/cells or variables/genes.",
    )
    gene_symbols: str | None = Field(
        default=None,
        description="Column name in `.var` DataFrame that stores gene symbols.",
    )
    groups: str | None = Field(
        default=None,
        description="Restrict to specific categories in categorical observation annotation.",
    )
    vmin: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="Lower limit of the color scale.",
    )
    vmax: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="Upper limit of the color scale.",
    )
    vcenter: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="Center of the color scale, useful for diverging colormaps.",
    )


class ScanpyPlUmapQueryParameters(BaseModel):
    """Parameters for querying the Scanpy `pl.umap` API."""

    method_name: str = Field(
        default="sc.pl.umap",
        description="The name of the method to call.",
    )
    question_uuid: str | None = Field(
        default=None,
        description="Unique identifier for the question.",
    )
    adata: str = Field(
        ...,
        description="Annotated data matrix.",
    )
    color: str | list[str] | None = Field(
        default=None,
        description="Keys for annotations of observations/cells or variables/genes.",
    )
    gene_symbols: str | None = Field(
        default=None,
        description="Column name in `.var` DataFrame that stores gene symbols.",
    )
    layer: str | None = Field(
        default=None,
        description="Name of the AnnData object layer to plot.",
    )
    vmax: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="Upper limit of the color scale.",
    )
    vmin: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="Lower limit of the color scale.",
    )
    vcenter: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="Center of the color scale, useful for diverging colormaps.",
    )


class ScanpyPlDrawGraphQueryParameters(BaseModel):
    """Parameters for querying the Scanpy `pl.draw_graph` API."""

    method_name: str = Field(
        default="sc.pl.draw_graph",
        description="The name of the method to call.",
    )
    question_uuid: str | None = Field(
        default=None,
        description="Unique identifier for the question.",
    )
    adata: str = Field(
        ...,
        description="Annotated data matrix.",
    )
    color: str | list[str] | None = Field(
        default=None,
        description="Keys for annotations of observations/cells or variables/genes.",
    )
    gene_symbols: str | None = Field(
        default=None,
        description="Column name in `.var` DataFrame that stores gene symbols.",
    )
    color_map: str | Any | None = Field(
        default=None,
        description="Color map to use for continuous variables.",
    )
    palette: str | list[str] | Any | None = Field(
        default=None,
        description="Colors to use for plotting categorical annotation groups.",
    )
    vmin: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="The value representing the lower limit of the color scale.",
    )
    vmax: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="The value representing the upper limit of the color scale.",
    )
    vcenter: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="The value representing the center of the color scale.",
    )


class ScanpyPlSpatialQueryParameters(BaseModel):
    """Parameters for querying the Scanpy `pl.spatial` API."""

    method_name: str = Field(
        default="sc.pl.spatial",
        description="The name of the method to call.",
    )
    question_uuid: str | None = Field(
        default=None,
        description="Unique identifier for the question.",
    )
    adata: str = Field(
        ...,
        description="Annotated data matrix.",
    )
    color: str | list[str] | None = Field(
        default=None,
        description="Keys for annotations of observations/cells or variables/genes.",
    )
    gene_symbols: str | None = Field(
        default=None,
        description="Column name in `.var` DataFrame that stores gene symbols.",
    )
    layer: str | None = Field(
        default=None,
        description="Name of the AnnData object layer to plot.",
    )
    library_id: str | None = Field(
        default=None,
        description="Library ID for Visium data, e.g., key in `adata.uns['spatial']`.",
    )
    img_key: str | None = Field(
        default=None,
        description=(
            "Key for image data, used to get `img` and `scale_factor` from "
            "'images' and 'scalefactors' entries for this library."
        ),
    )
    img: Any | None = Field(
        default=None,
        description="Image data to plot, overrides `img_key`.",
    )
    scale_factor: float | None = Field(
        default=None,
        description="Scaling factor used to map from coordinate space to pixel space.",
    )
    spot_size: float | None = Field(
        default=None,
        description="Diameter of spot (in coordinate space) for each point.",
    )
    vmin: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="The value representing the lower limit of the color scale.",
    )
    vmax: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="The value representing the upper limit of the color scale.",
    )
    vcenter: str | float | Any | list[str | float | Any] | None = Field(
        default=None,
        description="The value representing the center of the color scale.",
    )


class ScanpyPlQueryBuilder(BaseQueryBuilder):
    """A class for building a AnndataIO query object."""

    def create_runnable(
        self,
        query_parameters: list["BaseAPIModel"],
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
        runnable = conversation.chat.bind_tools(query_parameters, tool_choice="required")
        return runnable | PydanticToolsParser(tools=query_parameters)

    def parameterise_query(
        self,
        question: str,
        conversation: "Conversation",
    ) -> list["BaseModel"]:
        """Generate a AnnDataIOQuery object.

        Generates the object based on the given question, prompt, and
        BioChatter conversation. Uses a Pydantic model to define the API fields.
        Creates a runnable that can be invoked on LLMs that are qualified to
        parameterise functions.

        Args:
        ----
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                ScanpyPlQuery.

        Returns:
        -------
            ScanpyPlQuery: the parameterised query object (Pydantic model)

        """
        tools = [
            ScanpyPlScatterQueryParameters,
            ScanpyPlPcaQueryParameters,
            ScanpyPlTsneQueryParameters,
            ScanpyPlUmapQueryParameters,
            ScanpyPlDrawGraphQueryParameters,
            ScanpyPlSpatialQueryParameters,
        ]
        runnable = self.create_runnable(conversation=conversation, query_parameters=tools)
        return runnable.invoke(question)
