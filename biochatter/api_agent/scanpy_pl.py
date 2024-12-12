"""Module for interacting with the `scanpy` API for plotting (`pl`)."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

from .abc import BaseAPIModel, BaseQueryBuilder, BaseTools

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

class ScanpyPlTools(BaseTools):
    """A class containing parameters for Scanpy plotting functions."""

    tools_params = {}

    # Parameters for sc.pl.scatter
    tools_params["sc.pl.scatter"] = {
        "method_name": (str, Field(default="sc.pl.scatter", description="The name of the method to call")),
        "adata": (str, Field(description="Annotated data matrix")),
        "x": (Optional[str], Field(default=None, description="x coordinate")),
        "y": (Optional[str], Field(default=None, description="y coordinate")),
        "color": (Optional[str | tuple[float, ...] | list[str | tuple[float, ...]]], Field(default=None, description="Keys for annotations of observations/cells or variables/genes")),
        "use_raw": (Optional[bool], Field(default=None, description="Whether to use raw attribute of adata")),
        "layers": (Optional[str | list[str]], Field(default=None, description="Layer(s) to use from adata's layers attribute")),
        "basis": (Optional[str], Field(default=None, description="String that denotes a plotting tool that computed coordinates")),
        "sort_order": (bool, Field(default=True, description="Plot data points with higher values on top")),
        "groups": (Optional[str | list[str]], Field(default=None, description="Restrict to specific categories")),
        "projection": (str, Field(default="2d", description="Projection of plot ('2d' or '3d')")),
        "legend_loc": (Optional[str], Field(default="right margin", description="Location of legend")),
        "size": (Optional[int | float], Field(default=None, description="Point size")),
        "color_map": (Optional[str], Field(default=None, description="Color map to use")),
        "show": (Optional[bool], Field(default=None, description="Show the plot")),
        "save": (Optional[str | bool], Field(default=None, description="Save the figure"))
    }

    # Parameters for sc.pl.pca
    tools_params["sc.pl.pca"] = {
        "method_name": (str, Field(default="sc.pl.pca", description="The name of the method to call")),
        "adata": (str, Field(..., description="Annotated data matrix")),
        "color": (Optional[str | list[str]], Field(default=None, description="Keys for annotations")),
        "components": (str | list[str], Field(default="1,2", description="Components to plot")),
        "projection": (str, Field(default="2d", description="Projection of plot")),
        "legend_loc": (str, Field(default="right margin", description="Location of legend")),
        "legend_fontsize": (Optional[int | float | str], Field(default=None, description="Font size for legend")),
        "legend_fontweight": (Optional[int | str], Field(default=None, description="Font weight for legend")),
        "color_map": (Optional[str], Field(default=None, description="Color map")),
        "palette": (Optional[str | list[str] | dict], Field(default=None, description="Colors for categorical groups")),
        "frameon": (Optional[bool], Field(default=None, description="Draw frame around plot")),
        "size": (Optional[int | float], Field(default=None, description="Point size")),
        "show": (Optional[bool], Field(default=None, description="Show the plot")),
        "save": (Optional[str | bool], Field(default=None, description="Save the figure")),
        "ax": (Optional[str], Field(default=None, description="Matplotlib axes object")),
        "return_fig": (bool, Field(default=False, description="Return figure object")),
        "marker": (Optional[str], Field(default=".", description="Marker symbol")),
        "annotate_var_explained": (bool, Field(default=False, description="Annotate explained variance"))
    }

    tools_params["sc.pl.tsne"] = {
        "method_name": (str, Field(default="sc.pl.tsne", description="The name of the method to call")),
        "adata": (str, Field(..., description="Annotated data matrix")),
        "color": (Optional[str | list[str]], Field(default=None, description="Keys for annotations of observations/cells or variables/genes")),
        "gene_symbols": (Optional[str], Field(default=None, description="Column name in `.var` DataFrame that stores gene symbols")),
        "use_raw": (Optional[bool], Field(default=None, description="Use `.raw` attribute of `adata` for coloring with gene expression")),
        "sort_order": (bool, Field(default=True, description="Plot data points with higher values on top for continuous annotations")),
        "edges": (bool, Field(default=False, description="Show edges")),
        "edges_width": (float, Field(default=0.1, description="Width of edges")),
        "edges_color": (str | list[float] | list[str], Field(default="grey", description="Color of edges")),
        "neighbors_key": (Optional[str], Field(default=None, description="Key for neighbors connectivities")),
        "arrows": (bool, Field(default=False, description="Show arrows")),
        "arrows_kwds": (Optional[dict[str, Any]], Field(default=None, description="Arguments passed to `quiver()`")),
        "groups": (Optional[str], Field(default=None, description="Restrict to specific categories in categorical observation annotation")),
        "components": (Optional[str | list[str]], Field(default=None, description="Components to plot, e.g., ['1,2', '2,3']")),
        "projection": (str, Field(default="2d", description="Projection of plot")),
        "legend_loc": (str, Field(default="right margin", description="Location of legend")),
        "legend_fontsize": (Optional[int | float | str], Field(default=None, description="Font size for legend")),
        "legend_fontweight": (str | int, Field(default="bold", description="Font weight for legend")),
        "legend_fontoutline": (Optional[int], Field(default=None, description="Line width of the legend font outline in pt")),
        "size": (Optional[float | list[float]], Field(default=None, description="Point size")),
        "color_map": (Optional[str | Any], Field(default=None, description="Color map for continuous variables")),
        "palette": (Optional[str | list[str] | Any], Field(default=None, description="Colors for plotting categorical annotation groups")),
        "na_color": (str | tuple[float, ...], Field(default="lightgray", description="Color for null or masked values")),
        "na_in_legend": (bool, Field(default=True, description="Include missing values in the legend")),
        "frameon": (Optional[bool], Field(default=None, description="Draw a frame around the scatter plot")),
        "vmin": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="Lower limit of the color scale")),
        "vmax": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="Upper limit of the color scale")),
        "vcenter": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="Center of the color scale")),
        "norm": (Optional[Any], Field(default=None, description="Normalization for the colormap")),
        "add_outline": (bool, Field(default=False, description="Add a thin border around groups of dots")),
        "outline_width": (tuple[float, ...], Field(default=(0.3, 0.05), description="Width of the outline")),
        "outline_color": (tuple[str, ...], Field(default=("black", "white"), description="Colors for the outline")),
        "ncols": (int, Field(default=4, description="Number of panels per row")),
        "hspace": (float, Field(default=0.25, description="Height of the space between multiple panels")),
        "wspace": (Optional[float], Field(default=None, description="Width of the space between multiple panels")),
        "return_fig": (Optional[bool], Field(default=None, description="Return the matplotlib figure")),
        "show": (Optional[bool], Field(default=None, description="Show the plot")),
        "save": (Optional[str | bool], Field(default=None, description="Save the figure")),
        "ax": (Optional[Any], Field(default=None, description="A matplotlib axes object")),
        "kwargs": (Optional[dict[str, Any]], Field(default=None, description="Additional arguments passed to scatter()"))
    }

    # Parameters for sc.pl.umap
    tools_params["sc.pl.umap"] = {
        "method_name": (str, Field(default="sc.pl.umap", description="The name of the method to call")),
        "adata": (str, Field(..., description="Annotated data matrix")),
        "color": (Optional[str | list[str]], Field(default=None, description="Keys for annotations of observations/cells or variables/genes")),
        "mask_obs": (Optional[str], Field(default=None, description="Mask for observations")),
        "gene_symbols": (Optional[str], Field(default=None, description="Column name in `.var` DataFrame that stores gene symbols")),
        "use_raw": (Optional[bool], Field(default=None, description="Use `.raw` attribute of `adata` for coloring with gene expression")),
        "sort_order": (bool, Field(default=True, description="Plot data points with higher values on top for continuous annotations")),
        "edges": (bool, Field(default=False, description="Show edges")),
        "edges_width": (float, Field(default=0.1, description="Width of edges")),
        "edges_color": (str | list[float] | list[str], Field(default="grey", description="Color of edges")),
        "neighbors_key": (Optional[str], Field(default=None, description="Key for neighbors connectivities")),
        "arrows": (bool, Field(default=False, description="Show arrows")),
        "arrows_kwds": (Optional[dict[str, Any]], Field(default=None, description="Arguments passed to `quiver()`")),
        "groups": (Optional[str], Field(default=None, description="Restrict to specific categories")),
        "components": (Optional[str | list[str]], Field(default=None, description="Components to plot")),
        "dimensions": (Optional[int], Field(default=None, description="Number of dimensions to plot")),
        "layer": (Optional[str], Field(default=None, description="Name of the AnnData object layer to plot")),
        "projection": (str, Field(default="2d", description="Projection of plot")),
        "scale_factor": (Optional[float], Field(default=None, description="Scale factor for the plot")),
        "color_map": (Optional[str | Any], Field(default=None, description="Color map for continuous variables")),
        "cmap": (Optional[str | Any], Field(default=None, description="Alias for `color_map`")),
        "palette": (Optional[str | list[str] | Any], Field(default=None, description="Colors for plotting categorical annotation groups")),
        "na_color": (str | tuple[float, ...], Field(default="lightgray", description="Color for null or masked values")),
        "na_in_legend": (bool, Field(default=True, description="Include missing values in the legend")),
        "size": (Optional[float | list[float]], Field(default=None, description="Point size")),
        "frameon": (Optional[bool], Field(default=None, description="Draw a frame around the scatter plot")),
        "legend_fontsize": (Optional[int | float | str], Field(default=None, description="Font size for legend")),
        "legend_fontweight": (str | int, Field(default="bold", description="Font weight for legend")),
        "legend_loc": (str, Field(default="right margin", description="Location of legend")),
        "legend_fontoutline": (Optional[int], Field(default=None, description="Line width of the legend font outline in pt")),
        "colorbar_loc": (str, Field(default="right", description="Location of the colorbar")),
        "vmax": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="Upper limit of the color scale")),
        "vmin": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="Lower limit of the color scale")),
        "vcenter": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="Center of the color scale")),
        "norm": (Optional[Any], Field(default=None, description="Normalization for the colormap")),
        "add_outline": (bool, Field(default=False, description="Add a thin border around groups of dots")),
        "outline_width": (tuple[float, ...], Field(default=(0.3, 0.05), description="Width of the outline")),
        "outline_color": (tuple[str, ...], Field(default=("black", "white"), description="Colors for the outline")),
        "ncols": (int, Field(default=4, description="Number of panels per row")),
        "hspace": (float, Field(default=0.25, description="Height of the space between multiple panels")),
        "wspace": (Optional[float], Field(default=None, description="Width of the space between multiple panels")),
        "show": (Optional[bool], Field(default=None, description="Show the plot")),
        "save": (Optional[str | bool], Field(default=None, description="Save the figure")),
        "ax": (Optional[Any], Field(default=None, description="A matplotlib axes object")),
        "return_fig": (Optional[bool], Field(default=None, description="Return the matplotlib figure")),
        "marker": (str, Field(default=".", description="Marker symbol")),
        "kwargs": (Optional[dict[str, Any]], Field(default=None, description="Additional arguments passed to scatter()"))
    }

    # Parameters for sc.pl.draw_graph
    tools_params["sc.pl.draw_graph"] = {
        "method_name": (str, Field(default="sc.pl.draw_graph", description="The name of the method to call")),
        "adata": (str, Field(..., description="Annotated data matrix")),
        "color": (Optional[str | list[str]], Field(default=None, description="Keys for annotations of observations/cells or variables/genes")),
        "gene_symbols": (Optional[str], Field(default=None, description="Column name in `.var` DataFrame that stores gene symbols")),
        "use_raw": (Optional[bool], Field(default=None, description="Use `.raw` attribute of `adata` for coloring with gene expression")),
        "sort_order": (bool, Field(default=True, description="For continuous annotations used as color parameter, plot data points with higher values on top")),
        "edges": (bool, Field(default=False, description="Show edges")),
        "edges_width": (float, Field(default=0.1, description="Width of edges")),
        "edges_color": (str | list[float] | list[str], Field(default="grey", description="Color of edges")),
        "neighbors_key": (Optional[str], Field(default=None, description="Where to look for neighbors connectivities")),
        "arrows": (bool, Field(default=False, description="Show arrows")),
        "arrows_kwds": (Optional[dict[str, Any]], Field(default=None, description="Arguments passed to `quiver()`")),
        "groups": (Optional[str | list[str]], Field(default=None, description="Restrict to a few categories in categorical observation annotation")),
        "components": (Optional[str | list[str]], Field(default=None, description="For instance, ['1,2', '2,3']. To plot all available components use components='all'")),
        "projection": (str, Field(default="2d", description="Projection of plot")),
        "legend_loc": (str, Field(default="right margin", description="Location of legend")),
        "legend_fontsize": (Optional[int | float | str], Field(default=None, description="Numeric size in pt or string describing the size")),
        "legend_fontweight": (str | int, Field(default="bold", description="Legend font weight")),
        "legend_fontoutline": (Optional[int], Field(default=None, description="Line width of the legend font outline in pt")),
        "colorbar_loc": (Optional[str], Field(default="right", description="Where to place the colorbar for continuous variables")),
        "size": (Optional[float | list[float]], Field(default=None, description="Point size")),
        "color_map": (Optional[str | Any], Field(default=None, description="Color map to use for continuous variables")),
        "palette": (Optional[str | list[str] | Any], Field(default=None, description="Colors to use for plotting categorical annotation groups")),
        "na_color": (str | tuple[float, ...], Field(default="lightgray", description="Color to use for null or masked values")),
        "na_in_legend": (bool, Field(default=True, description="If there are missing values, whether they get an entry in the legend")),
        "frameon": (Optional[bool], Field(default=None, description="Draw a frame around the scatter plot")),
        "vmin": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="The value representing the lower limit of the color scale")),
        "vmax": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="The value representing the upper limit of the color scale")),
        "vcenter": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="The value representing the center of the color scale")),
        "norm": (Optional[Any], Field(default=None, description="Normalization for the colormap")),
        "add_outline": (bool, Field(default=False, description="Add a thin border around groups of dots")),
        "outline_width": (tuple[float, ...], Field(default=(0.3, 0.05), description="Width of the outline")),
        "outline_color": (tuple[str, ...], Field(default=("black", "white"), description="Colors for the outline")),
        "ncols": (int, Field(default=4, description="Number of panels per row")),
        "hspace": (float, Field(default=0.25, description="Height of the space between multiple panels")),
        "wspace": (Optional[float], Field(default=None, description="Width of the space between multiple panels")),
        "return_fig": (Optional[bool], Field(default=None, description="Return the matplotlib figure")),
        "show": (Optional[bool], Field(default=None, description="Show the plot")),
        "save": (Optional[str | bool], Field(default=None, description="Save the figure")),
        "ax": (Optional[Any], Field(default=None, description="A matplotlib axes object")),
        "layout": (Optional[str], Field(default=None, description="One of the `draw_graph()` layouts")),
        "kwargs": (Optional[dict[str, Any]], Field(default=None, description="Additional arguments passed to scatter()"))
    }

    # Parameters for sc.pl.spatial
    tools_params["sc.pl.spatial"] = {
        "method_name": (str, Field(default="sc.pl.spatial", description="The name of the method to call")),
        "adata": (str, Field(..., description="Annotated data matrix")),
        "color": (Optional[str | list[str]], Field(default=None, description="Keys for annotations of observations/cells or variables/genes")),
        "gene_symbols": (Optional[str], Field(default=None, description="Column name in `.var` DataFrame that stores gene symbols")),
        "use_raw": (Optional[bool], Field(default=None, description="Use `.raw` attribute of `adata` for coloring with gene expression")),
        "layer": (Optional[str], Field(default=None, description="Name of the AnnData object layer to plot")),
        "library_id": (Optional[str], Field(default=None, description="Library ID for Visium data")),
        "img_key": (Optional[str], Field(default=None, description="Key for image data")),
        "img": (Optional[Any], Field(default=None, description="Image data to plot, overrides `img_key`")),
        "scale_factor": (Optional[float], Field(default=None, description="Scaling factor used to map from coordinate space to pixel space")),
        "spot_size": (Optional[float], Field(default=None, description="Diameter of spot for each point")),
        "crop_coord": (Optional[tuple[int, ...]], Field(default=None, description="Coordinates to use for cropping the image")),
        "alpha_img": (float, Field(default=1.0, description="Alpha value for image")),
        "bw": (bool, Field(default=False, description="Plot image data in grayscale")),
        "sort_order": (bool, Field(default=True, description="Plot data points with higher values on top")),
        "groups": (Optional[str | list[str]], Field(default=None, description="Restrict to specific categories")),
        "components": (Optional[str | list[str]], Field(default=None, description="Components to plot")),
        "projection": (str, Field(default="2d", description="Projection of plot")),
        "legend_loc": (str, Field(default="right margin", description="Location of legend")),
        "legend_fontsize": (Optional[int | float | str], Field(default=None, description="Font size for legend")),
        "legend_fontweight": (str | int, Field(default="bold", description="Legend font weight")),
        "legend_fontoutline": (Optional[int], Field(default=None, description="Line width of the legend font outline")),
        "colorbar_loc": (Optional[str], Field(default="right", description="Where to place the colorbar")),
        "size": (float, Field(default=1.0, description="Point size")),
        "color_map": (Optional[str | Any], Field(default=None, description="Color map for continuous variables")),
        "palette": (Optional[str | list[str] | Any], Field(default=None, description="Colors for plotting categorical groups")),
        "na_color": (Optional[str | tuple[float, ...]], Field(default=None, description="Color for null values")),
        "na_in_legend": (bool, Field(default=True, description="Include missing values in legend")),
        "frameon": (Optional[bool], Field(default=None, description="Draw frame around plot")),
        "vmin": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="Lower limit of color scale")),
        "vmax": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="Upper limit of color scale")),
        "vcenter": (Optional[str | float | Any | list[str | float | Any]], Field(default=None, description="Center of color scale")),
        "norm": (Optional[Any], Field(default=None, description="Normalization for colormap")),
        "add_outline": (bool, Field(default=False, description="Add thin border around groups")),
        "outline_width": (tuple[float, ...], Field(default=(0.3, 0.05), description="Width of outline")),
        "outline_color": (tuple[str, ...], Field(default=("black", "white"), description="Colors for outline")),
        "ncols": (int, Field(default=4, description="Number of panels per row")),
        "hspace": (float, Field(default=0.25, description="Height of space between panels")),
        "wspace": (Optional[float], Field(default=None, description="Width of space between panels")),
        "return_fig": (Optional[bool], Field(default=None, description="Return matplotlib figure")),
        "show": (Optional[bool], Field(default=None, description="Show the plot")),
        "save": (Optional[str | bool], Field(default=None, description="Save the figure")),
        "ax": (Optional[Any], Field(default=None, description="Matplotlib axes object")),
        "kwargs": (Optional[dict[str, Any]], Field(default=None, description="Additional arguments for scatter()"))
    }

    def __init__(self):
        """Initialize the tools by creating Pydantic models from the parameters."""
        self.tools = self.make_pydantic_tools()

class ScanpyPlQueryBuilder(BaseQueryBuilder):
    """A class for building a ScanpyPl query object."""

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
        tool_maker = ScanpyPlTools()
        tools = tool_maker.make_pydantic_tools()
        runnable = self.create_runnable(conversation=conversation, query_parameters=tools)
        return runnable.invoke(question)
