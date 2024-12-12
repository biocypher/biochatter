import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Collection, Optional, Union, Literal
from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseAPIModel, Field
if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

from .abc import BaseAPIModel, BaseQueryBuilde


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


class FilterCellsParams(BaseAPIModel):
    data: "np.ndarray" | "spmatrix" | "AnnData" = Field(..., description="The (annotated) data matrix.")
    min_counts: int | None = Field(None, description="Minimum counts per cell.")
    min_genes: int | None = Field(None, description="Minimum genes expressed in a cell.")
    max_counts: int | None = Field(None, description="Maximum counts per cell.")
    max_genes: int | None = Field(None, description="Maximum genes expressed in a cell.")
    inplace: bool = Field(True, description="Whether to modify the data in place.")

class FilterGenesParams(BaseAPIModel):
    data: "np.ndarray" | "spmatrix" | "AnnData" = Field(..., description="The (annotated) data matrix.")
    min_counts: int | None = Field(None, description="Minimum counts per gene.")
    min_cells: int | None = Field(None, description="Minimum number of cells expressing the gene.")
    max_counts: int | None = Field(None, description="Maximum counts per gene.")
    max_cells: int | None = Field(None, description="Maximum number of cells expressing the gene.")
    inplace: bool = Field(True, description="Whether to modify the data in place.")

class HighlyVariableGenesParams(BaseAPIModel):
    adata: "AnnData" = Field(..., description="Annotated data matrix.")
    n_top_genes: int | None = Field(None, description="Number of highly-variable genes to keep.")
    min_mean: float = Field(0.0125, description="Minimum mean expression for highly-variable genes.")
    max_mean: float = Field(3, description="Maximum mean expression for highly-variable genes.")
    flavor: Literal['seurat', 'cell_ranger', 'seurat_v3', 'seurat_v3_paper'] = Field('seurat', description="Method for identifying highly-variable genes.")
    inplace: bool = Field(True, description="Whether to place metrics in .var or return them.")

class Log1pParams(BaseAPIModel):
    data: "AnnData" | "np.ndarray" | "spmatrix" = Field(..., description="The data matrix.")
    base: float | None = Field(None, description="Base of the logarithm.")
    copy: bool = Field(False, description="If True, return a copy of the data.")
    chunked: bool | None = Field(None, description="Process data in chunks.")
    
class PCAParams(BaseAPIModel):
    data: "AnnData" | "np.ndarray" | "spmatrix" = Field(..., description="The (annotated) data matrix.")
    n_comps: int | None = Field(None, description="Number of principal components to compute.")
    layer: str | None = Field(None, description="Element of layers to use for PCA.")
    zero_center: bool = Field(True, description="Whether to zero-center the data.")
    svd_solver: str | None = Field(None, description="SVD solver to use.")
    copy: bool = Field(False, description="If True, return a copy of the data.")

class NormalizeTotalParams(BaseAPIModel):
    adata: "AnnData" = Field(..., description="The annotated data matrix.")
    target_sum: float | None = Field(None, description="Target sum after normalization.")
    exclude_highly_expressed: bool = Field(False, description="Whether to exclude highly expressed genes.")
    inplace: bool = Field(True, description="Whether to update adata or return normalized data.")
    
class RegressOutParams(BaseAPIModel):
    adata: "AnnData" = Field(..., description="The annotated data matrix.")
    keys: str | Collection[str] = Field(..., description="Keys for regression.")
    copy: bool = Field(False, description="If True, return a copy of the data.")

class ScaleParams(BaseAPIModel):
    data: "AnnData" | "spmatrix" | "np.ndarray" = Field(..., description="The data matrix.")
    zero_center: bool = Field(True, description="Whether to zero-center the data.")
    copy: bool = Field(False, description="Whether to perform operation inplace.")
    
class SubsampleParams(BaseAPIModel):
    data: "AnnData" | "np.ndarray" | "spmatrix" = Field(..., description="The data matrix.")
    fraction: float | None = Field(None, description="Fraction of observations to subsample.")
    n_obs: int | None = Field(None, description="Number of observations to subsample.")
    copy: bool = Field(False, description="If True, return a copy of the data.")

class DownsampleCountsParams(BaseAPIModel):
    adata: "AnnData" = Field(..., description="The annotated data matrix.")
    counts_per_cell: int | "np.ndarray" | None = Field(None, description="Target total counts per cell.")
    replace: bool = Field(False, description="Whether to sample with replacement.")
    copy: bool = Field(False, description="If True, return a copy of the data.")

class CombatParams(BaseAPIModel):
    adata: "AnnData" = Field(..., description="The annotated data matrix.")
    key: str = Field('batch', description="Key for batch effect removal.")
    inplace: bool = Field(True, description="Whether to replace the data inplace.")

class ScrubletParams(BaseAPIModel):
    adata: "AnnData" = Field(..., description="Annotated data matrix.")
    sim_doublet_ratio: float = Field(2.0, description="Number of doublets to simulate.")
    threshold: float | None = Field(None, description="Doublet score threshold.")
    copy: bool = Field(False, description="If True, return a copy of the data.")

class ScrubletSimulateDoubletsParams(BaseAPIModel):
    adata: "AnnData" = Field(..., description="Annotated data matrix.")
    sim_doublet_ratio: float = Field(2.0, description="Number of doublets to simulate.")
    random_seed: int = Field(0, description="Random seed for reproducibility.")


class ScanpyPPQueryBuilder(BaseQueryBuilder):
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
        
        
        """Generate a ScanpyPPIOQuery object.

        Generates the object based on the given question, prompt, and
        BioChatter conversation. Uses a Pydantic model to define the API fields.
        Creates a runnable that can be invoked on LLMs that are qualified to
        parameterise functions.

        Args:
        ----
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                AnnDataIOQuery.

        Returns:
        -------
            AnnDataIOQuery: the parameterised query object (Pydantic model)

        """
        tools = [
        "FilterCellsParams",
        "FilterGenesParams",
        "HighlyVariableGenesParams",
        "Log1pParams",
        "PCAParams",
        "NormalizeTotalParams",
        "RegressOutParams",
        "ScaleParams",
        "SubsampleParams",
        "DownsampleCountsParams",
        "CombatParams",
        "ScrubletParams",
        "ScrubletSimulateDoubletsParams",
        ]

        runnable = self.create_runnable(
            query_parameters=tools,
            conversation=conversation,
        )
        oncokb_call_obj = runnable.invoke(
            {
                "input": f"Answer:\n{question} based on:\n {SCANPY_PL_QUERY_PROMPT}",
            },
        )
        oncokb_call_obj.question_uuid = str(uuid.uuid4())
        return [oncokb_call_obj]
