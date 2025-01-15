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
    show: bool | None = Field(
        default=None,
        description="Show the plot, do not return axis.",
    )
    save: str | bool | None = Field(
        default=None,
        description="If True or a str, save the figure. String is appended to default filename.",
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
    components: str | list[str] = Field(
        default="1,2",
        description="For example, ['1,2', '2,3']. To plot all available components use 'all'.",
    )
    projection: str = Field(
        default="2d",
        description="Projection of plot.",
    )
    legend_loc: str = Field(
        default="right margin",
        description="Location of legend.",
    )
    legend_fontsize: int | float | str | None = Field(
        default=None,
        description="Font size for legend.",
    )
    legend_fontweight: int | str | None = Field(
        default=None,
        description="Font weight for legend.",
    )
    color_map: str | None = Field(
        default=None,
        description="String denoting matplotlib color map.",
    )
    palette: str | list[str] | dict | None = Field(
        default=None,
        description="Colors to use for plotting categorical annotation groups.",
    )
    frameon: bool | None = Field(
        default=None,
        description="Draw a frame around the scatter plot.",
    )
    size: int | float | None = Field(
        default=None,
        description="Point size. If `None`, is automatically computed as 120000 / n_cells.",
    )
    show: bool | None = Field(
        default=None,
        description="Show the plot, do not return axis.",
    )
    save: str | bool | None = Field(
        default=None,
        description="If `True` or a `str`, save the figure.",
    )
    ax: str | None = Field(
        default=None,
        description="A matplotlib axes object.",
    )
    return_fig: bool = Field(
        default=False,
        description="Return the matplotlib figure object.",
    )
    marker: str | None = Field(
        default=".",
        description="Marker symbol.",
    )
    annotate_var_explained: bool = Field(
        default=False,
        description="Annotate the percentage of explained variance.",
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
    use_raw: bool | None = Field(
        default=None,
        description="Use `.raw` attribute of `adata` for coloring with gene expression.",
    )
    sort_order: bool = Field(
        default=True,
        description="Plot data points with higher values on top for continuous annotations.",
    )
    edges: bool = Field(
        default=False,
        description="Show edges.",
    )
    edges_width: float = Field(
        default=0.1,
        description="Width of edges.",
    )
    edges_color: str | list[float] | list[str] = Field(
        default="grey",
        description="Color of edges.",
    )
    neighbors_key: str | None = Field(
        default=None,
        description="Key for neighbors connectivities.",
    )
    arrows: bool = Field(
        default=False,
        description="Show arrows (deprecated in favor of `scvelo.pl.velocity_embedding`).",
    )
    arrows_kwds: dict[str, Any] | None = Field(
        default=None,
        description="Arguments passed to `quiver()`.",
    )
    groups: str | None = Field(
        default=None,
        description="Restrict to specific categories in categorical observation annotation.",
    )
    components: str | list[str] | None = Field(
        default=None,
        description="Components to plot, e.g., ['1,2', '2,3']. Use 'all' to plot all available components.",
    )
    projection: str = Field(
        default="2d",
        description="Projection of plot ('2d' or '3d').",
    )
    legend_loc: str = Field(
        default="right margin",
        description="Location of legend.",
    )
    legend_fontsize: int | float | str | None = Field(
        default=None,
        description="Font size for legend.",
    )
    legend_fontweight: int | str = Field(
        default="bold",
        description="Font weight for legend.",
    )
    legend_fontoutline: int | None = Field(
        default=None,
        description="Line width of the legend font outline in pt.",
    )
    size: float | list[float] | None = Field(
        default=None,
        description="Point size. If `None`, computed as 120000 / n_cells.",
    )
    color_map: str | Any | None = Field(
        default=None,
        description="Color map for continuous variables.",
    )
    palette: str | list[str] | Any | None = Field(
        default=None,
        description="Colors for plotting categorical annotation groups.",
    )
    na_color: str | tuple[float, ...] = Field(
        default="lightgray",
        description="Color for null or masked values.",
    )
    na_in_legend: bool = Field(
        default=True,
        description="Include missing values in the legend.",
    )
    frameon: bool | None = Field(
        default=None,
        description="Draw a frame around the scatter plot.",
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
    norm: Any | None = Field(
        default=None,
        description="Normalization for the colormap.",
    )
    add_outline: bool = Field(
        default=False,
        description="Add a thin border around groups of dots.",
    )
    outline_width: tuple[float, ...] = Field(
        default=(0.3, 0.05),
        description="Width of the outline as a fraction of the scatter dot size.",
    )
    outline_color: tuple[str, ...] = Field(
        default=("black", "white"),
        description="Colors for the outline: border color and gap color.",
    )
    ncols: int = Field(
        default=4,
        description="Number of panels per row.",
    )
    hspace: float = Field(
        default=0.25,
        description="Height of the space between multiple panels.",
    )
    wspace: float | None = Field(
        default=None,
        description="Width of the space between multiple panels.",
    )
    return_fig: bool | None = Field(
        default=None,
        description="Return the matplotlib figure.",
    )
    show: bool | None = Field(
        default=None,
        description="Show the plot; do not return axis.",
    )
    save: str | bool | None = Field(
        default=None,
        description="If `True` or a `str`, save the figure.",
    )
    ax: Any | None = Field(
        default=None,
        description="A matplotlib axes object.",
    )
    kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Additional arguments passed to `matplotlib.pyplot.scatter()`.",
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
    mask_obs: str | None = Field(
        default=None,
        description="Mask for observations.",
    )
    gene_symbols: str | None = Field(
        default=None,
        description="Column name in `.var` DataFrame that stores gene symbols.",
    )
    use_raw: bool | None = Field(
        default=None,
        description="Use `.raw` attribute of `adata` for coloring with gene expression.",
    )
    sort_order: bool = Field(
        default=True,
        description="Plot data points with higher values on top for continuous annotations.",
    )
    edges: bool = Field(
        default=False,
        description="Show edges.",
    )
    edges_width: float = Field(
        default=0.1,
        description="Width of edges.",
    )
    edges_color: str | list[float] | list[str] = Field(
        default="grey",
        description="Color of edges.",
    )
    neighbors_key: str | None = Field(
        default=None,
        description="Key for neighbors connectivities.",
    )
    arrows: bool = Field(
        default=False,
        description="Show arrows (deprecated in favor of `scvelo.pl.velocity_embedding`).",
    )
    arrows_kwds: dict[str, Any] | None = Field(
        default=None,
        description="Arguments passed to `quiver()`.",
    )
    groups: str | None = Field(
        default=None,
        description="Restrict to specific categories in categorical observation annotation.",
    )
    components: str | list[str] | None = Field(
        default=None,
        description="Components to plot, e.g., ['1,2', '2,3']. Use 'all' to plot all available components.",
    )
    dimensions: int | None = Field(
        default=None,
        description="Number of dimensions to plot.",
    )
    layer: str | None = Field(
        default=None,
        description="Name of the AnnData object layer to plot.",
    )
    projection: str = Field(
        default="2d",
        description="Projection of plot ('2d' or '3d').",
    )
    scale_factor: float | None = Field(
        default=None,
        description="Scale factor for the plot.",
    )
    color_map: str | Any | None = Field(
        default=None,
        description="Color map for continuous variables.",
    )
    cmap: str | Any | None = Field(
        default=None,
        description="Alias for `color_map`.",
    )
    palette: str | list[str] | Any | None = Field(
        default=None,
        description="Colors for plotting categorical annotation groups.",
    )
    na_color: str | tuple[float, ...] = Field(
        default="lightgray",
        description="Color for null or masked values.",
    )
    na_in_legend: bool = Field(
        default=True,
        description="Include missing values in the legend.",
    )
    size: float | list[float] | None = Field(
        default=None,
        description="Point size. If `None`, computed as 120000 / n_cells.",
    )
    frameon: bool | None = Field(
        default=None,
        description="Draw a frame around the scatter plot.",
    )
    legend_fontsize: int | float | str | None = Field(
        default=None,
        description="Font size for legend.",
    )
    legend_fontweight: int | str = Field(
        default="bold",
        description="Font weight for legend.",
    )
    legend_loc: str = Field(
        default="right margin",
        description="Location of legend.",
    )
    legend_fontoutline: int | None = Field(
        default=None,
        description="Line width of the legend font outline in pt.",
    )
    colorbar_loc: str = Field(
        default="right",
        description="Location of the colorbar.",
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
    norm: Any | None = Field(
        default=None,
        description="Normalization for the colormap.",
    )
    add_outline: bool = Field(
        default=False,
        description="Add a thin border around groups of dots.",
    )
    outline_width: tuple[float, ...] = Field(
        default=(0.3, 0.05),
        description="Width of the outline as a fraction of the scatter dot size.",
    )
    outline_color: tuple[str, ...] = Field(
        default=("black", "white"),
        description="Colors for the outline: border color and gap color.",
    )
    ncols: int = Field(
        default=4,
        description="Number of panels per row.",
    )
    hspace: float = Field(
        default=0.25,
        description="Height of the space between multiple panels.",
    )
    wspace: float | None = Field(
        default=None,
        description="Width of the space between multiple panels.",
    )
    show: bool | None = Field(
        default=None,
        description="Show the plot; do not return axis.",
    )
    save: str | bool | None = Field(
        default=None,
        description="If `True` or a `str`, save the figure.",
    )
    ax: Any | None = Field(
        default=None,
        description="A matplotlib axes object.",
    )
    return_fig: bool | None = Field(
        default=None,
        description="Return the matplotlib figure.",
    )
    marker: str = Field(
        default=".",
        description="Marker symbol.",
    )
    kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Additional arguments passed to `matplotlib.pyplot.scatter()`.",
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
    use_raw: bool | None = Field(
        default=None,
        description="Use `.raw` attribute of `adata` for coloring with gene expression.",
    )
    sort_order: bool = Field(
        default=True,
        description=(
            "For continuous annotations used as color parameter, "
            "plot data points with higher values on top of others."
        ),
    )
    edges: bool = Field(
        default=False,
        description="Show edges.",
    )
    edges_width: float = Field(
        default=0.1,
        description="Width of edges.",
    )
    edges_color: str | list[float] | list[str] = Field(
        default="grey",
        description="Color of edges.",
    )
    neighbors_key: str | None = Field(
        default=None,
        description="Where to look for neighbors connectivities.",
    )
    arrows: bool = Field(
        default=False,
        description="Show arrows (deprecated in favor of `scvelo.pl.velocity_embedding`).",
    )
    arrows_kwds: dict[str, Any] | None = Field(
        default=None,
        description="Arguments passed to `quiver()`.",
    )
    groups: str | list[str] | None = Field(
        default=None,
        description="Restrict to a few categories in categorical observation annotation.",
    )
    components: str | list[str] | None = Field(
        default=None,
        description="For instance, ['1,2', '2,3']. To plot all available components use components='all'.",
    )
    projection: str = Field(
        default="2d",
        description="Projection of plot.",
    )
    legend_loc: str = Field(
        default="right margin",
        description="Location of legend.",
    )
    legend_fontsize: int | float | str | None = Field(
        default=None,
        description="Numeric size in pt or string describing the size.",
    )
    legend_fontweight: int | str = Field(
        default="bold",
        description="Legend font weight.",
    )
    legend_fontoutline: int | None = Field(
        default=None,
        description="Line width of the legend font outline in pt.",
    )
    colorbar_loc: str | None = Field(
        default="right",
        description="Where to place the colorbar for continuous variables.",
    )
    size: float | list[float] | None = Field(
        default=None,
        description="Point size. If None, is automatically computed as 120000 / n_cells.",
    )
    color_map: str | Any | None = Field(
        default=None,
        description="Color map to use for continuous variables.",
    )
    palette: str | list[str] | Any | None = Field(
        default=None,
        description="Colors to use for plotting categorical annotation groups.",
    )
    na_color: str | tuple[float, ...] = Field(
        default="lightgray",
        description="Color to use for null or masked values.",
    )
    na_in_legend: bool = Field(
        default=True,
        description="If there are missing values, whether they get an entry in the legend.",
    )
    frameon: bool | None = Field(
        default=None,
        description="Draw a frame around the scatter plot.",
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
    norm: Any | None = Field(
        default=None,
        description="Normalization for the colormap.",
    )
    add_outline: bool = Field(
        default=False,
        description="Add a thin border around groups of dots.",
    )
    outline_width: tuple[float, ...] = Field(
        default=(0.3, 0.05),
        description="Width of the outline as a fraction of the scatter dot size.",
    )
    outline_color: tuple[str, ...] = Field(
        default=("black", "white"),
        description="Colors for the outline: border color and gap color.",
    )
    ncols: int = Field(
        default=4,
        description="Number of panels per row.",
    )
    hspace: float = Field(
        default=0.25,
        description="Height of the space between multiple panels.",
    )
    wspace: float | None = Field(
        default=None,
        description="Width of the space between multiple panels.",
    )
    return_fig: bool | None = Field(
        default=None,
        description="Return the matplotlib figure.",
    )
    show: bool | None = Field(
        default=None,
        description="Show the plot; do not return axis.",
    )
    save: str | bool | None = Field(
        default=None,
        description="If `True` or a `str`, save the figure.",
    )
    ax: Any | None = Field(
        default=None,
        description="A matplotlib axes object.",
    )
    layout: str | None = Field(
        default=None,
        description="One of the `draw_graph()` layouts.",
    )
    kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Additional arguments passed to `matplotlib.pyplot.scatter()`.",
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
    use_raw: bool | None = Field(
        default=None,
        description="Use `.raw` attribute of `adata` for coloring with gene expression.",
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
    crop_coord: tuple[int, ...] | None = Field(
        default=None,
        description="Coordinates to use for cropping the image (left, right, top, bottom).",
    )
    alpha_img: float = Field(
        default=1.0,
        description="Alpha value for image.",
    )
    bw: bool = Field(
        default=False,
        description="Plot image data in grayscale.",
    )
    sort_order: bool = Field(
        default=True,
        description=(
            "For continuous annotations used as color parameter, plot data points "
            "with higher values on top of others."
        ),
    )
    groups: str | list[str] | None = Field(
        default=None,
        description="Restrict to specific categories in categorical observation annotation.",
    )
    components: str | list[str] | None = Field(
        default=None,
        description="For example, ['1,2', '2,3']. To plot all available components, use 'all'.",
    )
    projection: str = Field(
        default="2d",
        description="Projection of plot.",
    )
    legend_loc: str = Field(
        default="right margin",
        description="Location of legend.",
    )
    legend_fontsize: int | float | str | None = Field(
        default=None,
        description="Numeric size in pt or string describing the size.",
    )
    legend_fontweight: int | str = Field(
        default="bold",
        description="Legend font weight.",
    )
    legend_fontoutline: int | None = Field(
        default=None,
        description="Line width of the legend font outline in pt.",
    )
    colorbar_loc: str | None = Field(
        default="right",
        description="Where to place the colorbar for continuous variables.",
    )
    size: float = Field(
        default=1.0,
        description="Point size. If None, automatically computed as 120000 / n_cells.",
    )
    color_map: str | Any | None = Field(
        default=None,
        description="Color map to use for continuous variables.",
    )
    palette: str | list[str] | Any | None = Field(
        default=None,
        description="Colors to use for plotting categorical annotation groups.",
    )
    na_color: str | tuple[float, ...] | None = Field(
        default=None,
        description="Color to use for null or masked values.",
    )
    na_in_legend: bool = Field(
        default=True,
        description="If there are missing values, whether they get an entry in the legend.",
    )
    frameon: bool | None = Field(
        default=None,
        description="Draw a frame around the scatter plot.",
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
    norm: Any | None = Field(
        default=None,
        description="Normalization for the colormap.",
    )
    add_outline: bool = Field(
        default=False,
        description="Add a thin border around groups of dots.",
    )
    outline_width: tuple[float, ...] = Field(
        default=(0.3, 0.05),
        description="Width of the outline as a fraction of the scatter dot size.",
    )
    outline_color: tuple[str, ...] = Field(
        default=("black", "white"),
        description="Colors for the outline: border color and gap color.",
    )
    ncols: int = Field(
        default=4,
        description="Number of panels per row.",
    )
    hspace: float = Field(
        default=0.25,
        description="Height of the space between multiple panels.",
    )
    wspace: float | None = Field(
        default=None,
        description="Width of the space between multiple panels.",
    )
    return_fig: bool | None = Field(
        default=None,
        description="Return the matplotlib figure.",
    )
    show: bool | None = Field(
        default=None,
        description="Show the plot; do not return axis.",
    )
    save: str | bool | None = Field(
        default=None,
        description="If `True` or a `str`, save the figure.",
    )
    ax: Any | None = Field(
        default=None,
        description="A matplotlib axes object.",
    )
    kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Additional arguments passed to `matplotlib.pyplot.scatter()`.",
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
        runnable = conversation.chat.bind_tools(query_parameters)
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
