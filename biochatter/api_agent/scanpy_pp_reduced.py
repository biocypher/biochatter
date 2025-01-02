from collections.abc import Callable
from typing import TYPE_CHECKING
from langchain_core.output_parsers import PydanticToolsParser
from biochatter.llm_connect import Conversation
from .abc import BaseAPIModel, BaseQueryBuilder, BaseTools
from typing import Union, Collection, Optional
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation




SCANPY_PL_QUERY_PROMPT = """
Scanpy Preprocessing (scanpy.pp) Query Guide

You are a world-class algorithm for creating queries in structured formats. Your task is to use the Python API of scanpy to answer questions about the scanpy.pp (preprocessing) module. All function calls should be prefixed with scanpy.pp.. For example, to normalize the data, you should use scanpy.pp.normalize_total.

Use the following documentation to craft queries for preprocessing tasks:
Preprocessing Functions in scanpy.pp

These are the primary preprocessing functions in scanpy.pp used for data cleaning, transformation, and filtering.

    pp.filter_cells
    Filters cells based on the minimum and maximum number of counts or genes.

    pp.filter_genes
    Filters genes based on the minimum and maximum number of counts or cells in which the gene is expressed.

    pp.normalize_total
    Normalizes the data by scaling each cell to a specified total count.

    pp.log1p
    Applies a natural logarithm transformation to the data (adds 1 before log transformation).

    pp.regress_out
    Removes the effects of unwanted sources of variation by regressing out specific variables.

    pp.scale
    Scales the data by zero-centering and (optionally) scaling each feature.

    pp.subsample
    Subsamples the data by randomly selecting a fraction of the observations or by a fixed number of observations.

    pp.highly_variable_genes
    Identifies and selects highly variable genes based on mean and dispersion criteria.

    pp.calculate_qc_metrics
    Computes quality control metrics such as the number of genes detected per cell, total counts per cell, and more.

Specialized Preprocessing Methods

These functions are used for specialized preprocessing workflows:

    pp.recipe_zhen17
    Implements a preprocessing recipe as described in Zheng et al., 2017 for single-cell RNA-seq data.

    pp.recipe_weinreb17
    Implements a preprocessing recipe as described in Weinreb et al., 2017 for single-cell RNA-seq data.

    pp.recipe_seurat
    Implements a preprocessing recipe for integration with the Seurat workflow.

    pp.combat
    Removes batch effects using the ComBat method.

    pp.scrublet
    Simulates and detects doublets in the dataset.

General Functions

    pp.dummy
    Placeholder function for dummy operations or custom preprocessing routines.

    pp.projection
    Projects the data into a new space after dimensionality reduction techniques like PCA.

Use the provided documentation to craft precise queries for any preprocessing needs in Scanpy. Ensure that your function call starts with scanpy.pp. and that you include relevant parameters based on the query.

This prompt guides the user to query the scanpy.pp module for preprocessing tasks, assisting with the construction of specific preprocessing operations, such as filtering, normalization, scaling, and more.
"""


class ScanpyPpFuncs(BaseTools):
    tools_params = {}
    tools_descriptions = {}

    # Parameters for sc.pp.filter_cells
    tool_name = "filter_cells"
    tools_descriptions[tool_name] = "Filters cells based on the minimum and maximum number of counts or genes. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "data": (str, Field(..., description="The (annotated) data matrix.")),
        "min_counts": (Optional[int], Field(None, description="Minimum counts per cell.")),
        "min_genes": (Optional[int], Field(None, description="Minimum genes expressed in a cell.")),
        "max_counts": (Optional[int], Field(None, description="Maximum counts per cell.")),
        "max_genes": (Optional[int], Field(None, description="Maximum genes expressed in a cell.")),
        "inplace": (bool, Field(True, description="Whether to modify the data in place."))
    }

    # Parameters for sc.pp.filter_genes
    tool_name = "filter_genes"
    tools_descriptions[tool_name] = "Filters genes based on the minimum and maximum number of counts or cells. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "data": (str, Field(..., description="The (annotated) data matrix.")),
        "min_counts": (Optional[int], Field(None, description="Minimum counts per gene.")),
        "min_cells": (Optional[int], Field(None, description="Minimum number of cells expressing the gene.")),
        "max_counts": (Optional[int], Field(None, description="Maximum counts per gene.")),
        "max_cells": (Optional[int], Field(None, description="Maximum number of cells expressing the gene.")),
        "inplace": (bool, Field(True, description="Whether to modify the data in place."))
    }
    # Parameters for sc.pp.highly_variable_genes
    tool_name = "highly_variable_genes"
    tools_descriptions[tool_name] = "Identifies and selects highly variable genes based on mean and dispersion criteria. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="Annotated data matrix.")),
        "n_top_genes": (Optional[int], Field(None, description="Number of highly-variable genes to keep.")),
        "min_mean": (float, Field(0.0125, description="Minimum mean expression for highly-variable genes.")),
        "max_mean": (float, Field(3, description="Maximum mean expression for highly-variable genes.")),
        "flavor": (str, Field('seurat', description="Method for identifying highly-variable genes.")),
        "inplace": (bool, Field(True, description="Whether to place metrics in .var or return them."))
    }
    
    # Parameters for sc.pp.log1p
    tool_name = "log1p"
    tools_descriptions[tool_name] = "Logarithmize the data matrix by computing log(data + 1). Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "data": (str, Field(..., description="The data matrix.")),
        "base": (Optional[float], Field(None, description="Base of the logarithm.")), 
        "copy": (bool, Field(False, description="If True, return a copy of the data.")),
        "chunked": (Optional[bool], Field(None, description="Process data in chunks."))
    }
    # Parameters for sc.pp.pca
    tool_name = "pca"
    tools_descriptions[tool_name] = "Principal component analysis. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "data": (str, Field(..., description="The (annotated) data matrix.")),
        "n_comps": (Optional[int], Field(None, description="Number of principal components to compute.")),
        "layer": (Optional[str], Field(None, description="Element of layers to use for PCA.")),
        "zero_center": (bool, Field(True, description="Whether to zero-center the data.")),
        "svd_solver": (Optional[str], Field(None, description="SVD solver to use.")),
        "copy": (bool, Field(False, description="If True, return a copy_param of the data."))
    }

    # Parameters for sc.pp.normalize_total
    tool_name = "normalize_total"
    tools_descriptions[tool_name] = "Normalize counts per cell. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="The annotated data matrix.")),
        "target_sum": (Optional[float], Field(None, description="Target sum after normalization.")),
        "exclude_highly_expressed": (bool, Field(False, description="Whether to exclude highly expressed genes.")),
        "inplace": (bool, Field(True, description="Whether to update adata or return normalized data."))
    }

    # Parameters for sc.pp.regress_out
    tool_name = "regress_out"
    tools_descriptions[tool_name] = "Regress out unwanted sources of variation. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="The annotated data matrix.")),
        "keys": (Union[str, Collection[str]], Field(..., description="Keys for regression.")),
        "copy": (bool, Field(False, description="If True, return a copy of the data."))
    }

    # Parameters for sc.pp.scale
    tool_name = "scale"
    tools_descriptions[tool_name] = "Scale data to unit variance and zero mean. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "data": (str, Field(..., description="The data matrix.")),
        "zero_center": (bool, Field(True, description="Whether to zero-center the data.")),
        "copy": (bool, Field(False, description="Whether to perform operation inplace."))
    }

    # Parameters for sc.pp.subsample
    tool_name = "subsample"
    tools_descriptions[tool_name] = "Subsample to a fraction of the number of observations. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "data": (str, Field(..., description="The data matrix.")),
        "fraction": (Optional[float], Field(None, description="Fraction of observations to subsample.")),
        "n_obs": (Optional[int], Field(None, description="Number of observations to subsample.")),
        "copy": (bool, Field(False, description="If True, return a copy_param of the data."))
    }

    # Parameters for sc.pp.downsample_counts
    tool_name = "downsample_counts"
    tools_descriptions[tool_name] = "Downsample counts per cell. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="The annotated data matrix.")),
        "counts_per_cell": (Optional[Union[int, str]], Field(None, description="Target total counts per cell.")),
        "replace": (bool, Field(False, description="Whether to sample with replacement.")),
        "copy": (bool, Field(False, description="If True, return a copy_param of the data."))
    }

    # Parameters for sc.pp.combat
    tool_name = "combat"
    tools_descriptions[tool_name] = "Remove batch effects using the ComBat method. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="The annotated data matrix.")),
        "key": (str, Field('batch', description="Key for batch effect removal.")),
        "inplace": (bool, Field(True, description="Whether to replace the data inplace."))
    }

    # Parameters for sc.pp.scrublet
    tool_name = "scrublet"
    tools_descriptions[tool_name] = "Simulate and detect doublets in the dataset. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="Annotated data matrix.")),
        "sim_doublet_ratio": (float, Field(2.0, description="Number of doublets to simulate.")),
        "threshold": (Optional[float], Field(None, description="Doublet score threshold.")),
        "copy": (bool, Field(False, description="If True, return a copy_param of the data."))
    }

    # Parameters for sc.pp.scrublet_simulate_doublets
    tool_name = "scrublet_simulate_doublets"
    tools_descriptions[tool_name] = "Simulate doublets in the dataset. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="Annotated data matrix.")),
        "sim_doublet_ratio": (float, Field(2.0, description="Number of doublets to simulate.")),
        "random_seed": (int, Field(0, description="Random seed for reproducibility."))
    }

    # Parameters for sc.pp.calculate_qc_metrics
    tool_name = "calculate_qc_metrics"
    tools_descriptions[tool_name] = "Compute quality control metrics. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="The annotated data matrix.")),
        "expr_type": (str, Field('counts', description="Name of kind of values in X.")),
        "var_type": (str, Field('genes', description="The kind of thing the variables are.")),
        "qc_vars": (Collection[str], Field((),
                                           description="Keys for boolean columns of .var which identify variables you could want to control for (e.g., “ERCC” or “mito”).")),
        "percent_top": (Collection[int], Field((50, 100, 200, 500),
                                               description="List of ranks at which cumulative proportion of expression will be reported as a percentage.")),
        "layer": (Optional[str], Field(None,
                                       description="If provided, use adata.layers[layer] for expression values instead of adata.X.")),
        "use_raw": (
        bool, Field(False, description="If True, use adata.raw.X for expression values instead of adata.X.")),
        "inplace": (bool, Field(False, description="Whether to place calculated metrics in adata’s .obs and .var.")),
        "log1p": (bool, Field(True, description="Set to False to skip computing log1p transformed annotations."))
    }

    # Parameters for sc.pp.recipe_zheng17
    tool_name = "recipe_zheng17"
    tools_descriptions[tool_name] = "Recipe for preprocessing as of Zheng et al. (2017). Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="The annotated data matrix.")),
        "n_top_genes": (int, Field(1000, description="Number of genes to keep.")),
        "log": (bool, Field(True, description="Take logarithm of the data.")),
        "plot": (bool, Field(False, description="Show a plot of the gene dispersion vs. mean relation.")),
        "copy": (bool, Field(False, description="Return a copy of adata instead of updating it."))
    }

    # Parameters for sc.pp.recipe_weinreb17
    tool_name = "recipe_weinreb17"
    tools_descriptions[tool_name] = "Recipe for preprocessing as of Weinreb et al. (2017). Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="The annotated data matrix.")),
        "log": (bool, Field(True, description="Logarithmize the data?")),
        "mean_threshold": (float, Field(0.01, description="Mean expression threshold for gene selection.")),
        "cv_threshold": (float, Field(2, description="Coefficient of variation threshold for gene selection.")),
        "n_pcs": (int, Field(50, description="Number of principal components to compute.")),
        "svd_solver": (str, Field('randomized', description="SVD solver to use for PCA.")),
        "random_state": (int, Field(0, description="Random seed for reproducibility.")),
        "copy": (bool, Field(False, description="Return a copy if true, otherwise modifies the original adata object."))
    }

    # Parameters for sc.pp.recipe_seurat
    tool_name = "recipe_seurat"
    tools_descriptions[tool_name] = "Recipe for preprocessing with the Seurat flavor. Available in the scanpy.pp module"
    tools_params[tool_name] = {
        "adata": (str, Field(..., description="The annotated data matrix.")),
        "log": (bool, Field(True, description="Logarithmize the data?")),
        "plot": (bool, Field(False, description="Show a plot of the gene dispersion vs. mean relation.")),
        "copy": (bool, Field(False, description="Return a copy if true, otherwise modifies the original adata object."))
    }

    def __init__(self, tools_params: dict = tools_params, tools_descriptions: dict = tools_descriptions):
        """Initialize the tools by creating Pydantic models from the parameters."""
        self.tools_params = tools_params
        self.tools_descriptions = tools_descriptions

class ScanpyPpQueryBuilder(BaseQueryBuilder):
    """A class for building a ScanpyPp query object."""

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
        """Generate a ScanpyPp query object.

        Generates the object based on the given question, prompt, and
        BioChatter conversation. Uses a Pydantic model to define the API fields.
        Creates a runnable that can be invoked on LLMs that are qualified to
        parameterise functions.

        Args:
        ----
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                ScanpyPpQuery.

        Returns:
        -------
            ScanpyPpQuery: the parameterised query object (Pydantic model)

        """
        tool_maker = ScanpyPpFuncs()
        tools = tool_maker.make_pydantic_tools()
        runnable = self.create_runnable(
            conversation=conversation, query_parameters=tools
        )
        scanpy_pp_call_obj = runnable.invoke(
            question,
        )
        return scanpy_pp_call_obj