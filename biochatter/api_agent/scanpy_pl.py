"""Module for interacting with the `scanpy` API for plotting (`pl`)."""

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

import requests
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

from .abc import BaseFetcher, BaseInterpreter, BaseQueryBuilder

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


class ScanpyPlQueryParameters(BaseModel):
    """Parameters for querying the scanpy plotting API."""

    scatter: str = Field(
        default=None,
        description="scanpy.pl.scatter(adata, x=None, y=None, *, color=None, use_raw=None, layers=None, sort_order=True, alpha=None, basis=None, groups=None, components=None, projection='2d', legend_loc='right margin', legend_fontsize=None, legend_fontweight=None, legend_fontoutline=None, color_map=None, palette=None, frameon=None, right_margin=None, left_margin=None, size=None, marker='.', title=None, show=None, save=None, ax=None)[source]"
    )
    heatmap: str = Field(
        default=None,
        description="scanpy.pl.heatmap(adata, var_names, groupby, *, use_raw=None, log=False, num_categories=7, dendrogram=False, gene_symbols=None, var_group_positions=None, var_group_labels=None, var_group_rotation=None, layer=None, standard_scale=None, swap_axes=False, show_gene_labels=None, show=None, save=None, figsize=None, vmin=None, vmax=None, vcenter=None, norm=None, **kwds)"
    )
    dotplot: str = Field(
        default=None,
        description="scanpy.pl.dotplot(adata, var_names, groupby, *, use_raw=None, log=False, num_categories=7, categories_order=None, expression_cutoff=0.0, mean_only_expressed=False, standard_scale=None, title=None, colorbar_title='Mean expression\\nin group', size_title='Fraction of cells\\nin group (%)', figsize=None, dendrogram=False, gene_symbols=None, var_group_positions=None, var_group_labels=None, var_group_rotation=None, layer=None, swap_axes=False, dot_color_df=None, show=None, save=None, ax=None, return_fig=False, vmin=None, vmax=None, vcenter=None, norm=None, cmap='Reds', dot_max=None, dot_min=None, smallest_dot=0.0, **kwds)"
    )


class ScanpyPlQueryBuilder(BaseQueryBuilder):
    """A class for building an ScanpyPlQuery object."""

    def create_runnable(
        self,
        query_parameters: "ScanpyPlQueryParameters",
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
    ) -> ScanpyPlQueryParameters:
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
            query_parameters=ScanpyPlQueryParameters,
            conversation=conversation,
        )
        scanpy_pl_call_obj = runnable.invoke(
            {
                "input": f"Answer:\n{question} based on:\n {SCANPY_PL_QUERY_PROMPT}",
            },
        )
        scanpy_pl_call_obj.question_uuid = str(uuid.uuid4())
        return scanpy_pl_call_obj
