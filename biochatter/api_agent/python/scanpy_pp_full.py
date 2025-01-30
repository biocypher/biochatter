from collections.abc import Collection
from typing import Literal

from pydantic import BaseModel, Field


class CalculateQCMetricsParams(BaseModel):
    adata: str = Field(..., description="Annotated data matrix")
    expr_type: str = Field("counts", description="Name of kind of values in X")
    var_type: str = Field("genes", description="The kind of thing the variables are")
    qc_vars: str = Field("", description="Keys for boolean columns of .var for control variables")
    percent_top: str = Field("50,100,200,500", description="Ranks for library complexity assessment")
    layer: str = Field(None, description="Layer to use for expression values")
    use_raw: bool = Field(False, description="Use adata.raw.X instead of adata.X")
    inplace: bool = Field(False, description="Place calculated metrics in adata's .obs and .var")
    log1p: bool = Field(True, description="Compute log1p transformed annotations")
    parallel: bool | None = Field(None, description="Parallel computation flag")

    class Config:
        arbitrary_types_allowed = True


class FilterCellsParams(BaseModel):
    data: str = Field(
        ...,
        description="The (annotated) data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.",
    )
    min_counts: int | None = Field(None, description="Minimum number of counts required for a cell to pass filtering.")
    min_genes: int | None = Field(
        None, description="Minimum number of genes expressed required for a cell to pass filtering."
    )
    max_counts: int | None = Field(None, description="Maximum number of counts required for a cell to pass filtering.")
    max_genes: int | None = Field(
        None, description="Maximum number of genes expressed required for a cell to pass filtering."
    )
    inplace: bool = Field(True, description="Perform computation inplace or return result.")
    copy: bool = Field(False, description="Whether to copy the data or modify it inplace.")

    class Config:
        arbitrary_types_allowed = True


class FilterGenesParams(BaseModel):
    data: str = Field(
        ...,
        description="An annotated data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.",
    )
    min_counts: int | None = Field(None, description="Minimum number of counts required for a gene to pass filtering.")
    min_cells: int | None = Field(
        None,
        description="Minimum number of cells in which the gene is expressed required for the gene to pass filtering.",
    )
    max_counts: int | None = Field(None, description="Maximum number of counts allowed for a gene to pass filtering.")
    max_cells: int | None = Field(
        None,
        description="Maximum number of cells in which the gene is expressed allowed for the gene to pass filtering.",
    )
    inplace: bool = Field(True, description="Perform computation inplace or return result.")
    copy: bool = Field(False, description="Whether to return a copy of the data (not modifying the original).")

    class Config:
        arbitrary_types_allowed = True


class HighlyVariableGenesParams(BaseModel):
    adata: str = Field(
        ..., description="Annotated data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes."
    )
    layer: str | None = Field(None, description="Use adata.layers[layer] for expression values instead of adata.X.")
    n_top_genes: int | None = Field(
        None, description="Number of highly-variable genes to keep. Mandatory if flavor='seurat_v3'."
    )
    min_mean: float = Field(
        0.0125,
        description="Minimum mean expression threshold for highly variable genes. Ignored if flavor='seurat_v3'.",
    )
    max_mean: float = Field(
        3, description="Maximum mean expression threshold for highly variable genes. Ignored if flavor='seurat_v3'."
    )
    min_disp: float = Field(
        0.5, description="Minimum dispersion threshold for highly variable genes. Ignored if flavor='seurat_v3'."
    )
    max_disp: float = Field(
        float("inf"),
        description="Maximum dispersion threshold for highly variable genes. Ignored if flavor='seurat_v3'.",
    )
    span: float = Field(
        0.3,
        description="Fraction of the data (cells) used in variance estimation for the loess model fit if flavor='seurat_v3'.",
    )
    n_bins: int = Field(
        20, description="Number of bins for binning the mean gene expression. Normalization is done per bin."
    )
    flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"] = Field(
        "seurat", description="The method to use for identifying highly variable genes."
    )
    subset: bool = Field(False, description="If True, subset to highly-variable genes, otherwise just indicate them.")
    inplace: bool = Field(True, description="Whether to place calculated metrics in .var or return them.")
    batch_key: str | None = Field(
        None, description="If specified, highly-variable genes are selected separately within each batch and merged."
    )
    check_values: bool = Field(
        True, description="Whether to check if counts in selected layer are integers (relevant for flavor='seurat_v3')."
    )

    class Config:
        arbitrary_types_allowed = True


class Log1pParams(BaseModel):
    data: str = Field(
        ...,
        description="The (annotated) data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.",
    )
    base: float | None = Field(None, description="Base of the logarithm. Natural logarithm is used by default.")
    copy: bool = Field(
        False, description="If True, a copy of the data is returned. Otherwise, the operation is done inplace."
    )
    chunked: bool | None = Field(
        None, description="Process the data matrix in chunks, which will save memory. Applies only to AnnData."
    )
    chunk_size: int | None = Field(None, description="Number of observations (n_obs) per chunk to process the data in.")
    layer: str | None = Field(None, description="Entry of layers to transform.")
    obsm: str | None = Field(None, description="Entry of obsm to transform.")

    class Config:
        arbitrary_types_allowed = True


class PCAParams(BaseModel):
    data: str = Field(
        ...,
        description="The (annotated) data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.",
    )
    n_comps: int | None = Field(
        None,
        description="Number of principal components to compute. Defaults to 50, or 1 - minimum dimension size of selected representation.",
    )
    layer: str | None = Field(None, description="If provided, which element of layers to use for PCA.")
    zero_center: bool = Field(
        True,
        description="If True, compute standard PCA from covariance matrix. If False, omit zero-centering variables.",
    )
    svd_solver: str | None = Field(
        None, description="SVD solver to use. Options: 'auto', 'arpack', 'randomized', 'lobpcg', or 'tsqr'."
    )
    random_state: int | None = Field(0, description="Change to use different initial states for the optimization.")
    return_info: bool = Field(
        False, description="Only relevant when not passing an AnnData. Whether to return PCA info."
    )
    mask_var: str | None = Field(
        None, description="To run PCA only on certain genes. Default is .var['highly_variable'] if available."
    )
    use_highly_variable: bool | None = Field(
        None,
        description="Whether to use highly variable genes only, stored in .var['highly_variable']. Deprecated in 1.10.0.",
    )
    dtype: str = Field("float32", description="Numpy data type string to which to convert the result.")
    chunked: bool = Field(
        False, description="If True, perform incremental PCA using sklearn IncrementalPCA or dask-ml IncrementalPCA."
    )
    chunk_size: int | None = Field(
        None, description="Number of observations to include in each chunk. Required if chunked=True."
    )
    copy: bool = Field(
        False,
        description="If True, a copy of the data is returned when AnnData is passed. Otherwise, the operation is done inplace.",
    )

    class Config:
        arbitrary_types_allowed = True


class NormalizeTotalParams(BaseModel):
    adata: str = Field(
        ...,
        description="The annotated data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.",
    )
    target_sum: float | None = Field(
        None,
        description="Target sum after normalization. If None, each cell will have total counts equal to the median before normalization.",
    )
    exclude_highly_expressed: bool = Field(
        False, description="If True, exclude highly expressed genes from normalization computation."
    )
    max_fraction: float = Field(
        0.05,
        description="If exclude_highly_expressed=True, consider a gene as highly expressed if it has more than max_fraction of the total counts in at least one cell.",
    )
    key_added: str | None = Field(
        None, description="Name of the field in adata.obs where the normalization factor is stored."
    )
    layer: str | None = Field(None, description="Layer to normalize instead of X. If None, normalize X.")
    inplace: bool = Field(
        True, description="Whether to update adata or return normalized copies of adata.X and adata.layers."
    )
    copy: bool = Field(
        False, description="Whether to modify the copied input object. Not compatible with inplace=False."
    )

    class Config:
        arbitrary_types_allowed = True


class RegressOutParams(BaseModel):
    adata: str = Field(..., description="The annotated data matrix.")
    keys: str | Collection[str] = Field(
        ...,
        description="Keys for observation annotation on which to regress on. Can be a single key or a collection of keys.",
    )
    layer: str | None = Field(None, description="Layer to regress on, if provided.")
    n_jobs: int | None = Field(
        None, description="Number of jobs for parallel computation. None means using default n_jobs."
    )
    copy: bool = Field(False, description="If True, a copy of the data will be returned. Otherwise, modifies in-place.")

    class Config:
        arbitrary_types_allowed = True


class ScaleParams(BaseModel):
    data: str = Field(
        ...,
        description="The (annotated) data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.",
    )
    zero_center: bool = Field(
        True, description="If False, omit zero-centering variables, which allows to handle sparse input efficiently."
    )
    max_value: float | None = Field(
        None, description="Clip (truncate) to this value after scaling. If None, do not clip."
    )
    copy: bool = Field(False, description="Whether this function should be performed inplace.")
    layer: str | None = Field(None, description="If provided, which element of layers to scale.")
    obsm: str | None = Field(None, description="If provided, which element of obsm to scale.")
    mask_obs: str | None = Field(
        None,
        description="Restrict the scaling to a certain set of observations. The mask is specified as a boolean array or a string referring to an array in obs.",
    )

    class Config:
        arbitrary_types_allowed = True


class SubsampleParams(BaseModel):
    data: str = Field(
        ...,
        description="The (annotated) data matrix of shape n_obs × n_vars. Rows correspond to cells and columns to genes.",
    )
    fraction: float | None = Field(None, description="Subsample to this fraction of the number of observations.")
    n_obs: int | None = Field(None, description="Subsample to this number of observations.")
    random_state: int | None = Field(0, description="Random seed to change subsampling.")
    copy: bool = Field(False, description="If an AnnData is passed, determines whether a copy is returned.")

    class Config:
        arbitrary_types_allowed = True


class DownsampleCountsParams(BaseModel):
    adata: str = Field(..., description="Annotated data matrix.")
    counts_per_cell: int | None = Field(
        None,
        description="Target total counts per cell. If a cell has more than ‘counts_per_cell’, it will be downsampled to this number. Can be an integer or integer ndarray with same length as number of observations.",
    )
    total_counts: int | None = Field(
        None,
        description="Target total counts. If the count matrix has more than total_counts, it will be downsampled to this number.",
    )
    random_state: int | None = Field(0, description="Random seed for subsampling.")
    replace: bool = Field(False, description="Whether to sample the counts with replacement.")
    copy: bool = Field(False, description="Determines whether a copy of adata is returned.")

    class Config:
        arbitrary_types_allowed = True


class RecipeZheng17Params(BaseModel):
    adata: str = Field(..., description="Annotated data matrix.")
    n_top_genes: int = Field(1000, description="Number of genes to keep.")
    log: bool = Field(True, description="Take logarithm. If True, log-transform data after filtering.")
    plot: bool = Field(False, description="Show a plot of the gene dispersion vs. mean relation.")
    copy: bool = Field(False, description="Return a copy of adata instead of updating it.")

    class Config:
        arbitrary_types_allowed = True


class RecipeWeinreb17Params(BaseModel):
    adata: str = Field(..., description="Annotated data matrix.")
    log: bool = Field(True, description="Logarithmize data? If True, log-transform the data.")
    mean_threshold: float = Field(0.01, description="Threshold for mean expression of genes.")
    cv_threshold: float = Field(2, description="Threshold for coefficient of variation (CV) for gene dispersion.")
    n_pcs: int = Field(50, description="Number of principal components to use.")
    svd_solver: str = Field("randomized", description="SVD solver to use.")
    random_state: int = Field(0, description="Random state for reproducibility of results.")
    copy: bool = Field(False, description="Return a copy if True, else modifies the original AnnData.")

    class Config:
        arbitrary_types_allowed = True


class RecipeSeuratParams(BaseModel):
    adata: str = Field(..., description="Annotated data matrix.")
    log: bool = Field(True, description="Logarithmize data? If True, log-transform the data.")
    plot: bool = Field(False, description="Show a plot of the gene dispersion vs. mean relation.")
    copy: bool = Field(False, description="Return a copy if True, else modifies the original AnnData.")

    class Config:
        arbitrary_types_allowed = True


class CombatParams(BaseModel):
    adata: str = Field(..., description="Annotated data matrix.")
    key: str = Field(
        "batch", description="Key to a categorical annotation from obs that will be used for batch effect removal."
    )
    covariates: list[str] | None = Field(
        None, description="Additional covariates such as adjustment variables or biological conditions."
    )
    inplace: bool = Field(True, description="Whether to replace adata.X or to return the corrected data.")

    class Config:
        arbitrary_types_allowed = True


class ScrubletParams(BaseModel):
    adata: str = Field(..., description="Annotated data matrix (n_obs × n_vars).")
    adata_sim: str | None = Field(
        None, description="Optional AnnData object from scrublet_simulate_doublets() with same number of vars as adata."
    )
    batch_key: str | None = Field(None, description="Optional obs column name discriminating between batches.")
    sim_doublet_ratio: float = Field(
        2.0, description="Number of doublets to simulate relative to the number of observed transcriptomes."
    )
    expected_doublet_rate: float = Field(0.05, description="Estimated doublet rate for the experiment.")
    stdev_doublet_rate: float = Field(0.02, description="Uncertainty in the expected doublet rate.")
    synthetic_doublet_umi_subsampling: float = Field(
        1.0, description="Rate for sampling UMIs when creating synthetic doublets."
    )
    knn_dist_metric: str = Field("euclidean", description="Distance metric used for nearest neighbor search.")
    normalize_variance: bool = Field(True, description="Normalize the data such that each gene has a variance of 1.")
    log_transform: bool = Field(False, description="Whether to log-transform the data prior to PCA.")
    mean_center: bool = Field(True, description="If True, center the data such that each gene has a mean of 0.")
    n_prin_comps: int = Field(
        30,
        description="Number of principal components used to embed the transcriptomes prior to KNN graph construction.",
    )
    use_approx_neighbors: bool = Field(
        False, description="Use approximate nearest neighbor method (annoy) for KNN classifier."
    )
    get_doublet_neighbor_parents: bool = Field(
        False, description="If True, return parent transcriptomes that generated the doublet neighbors."
    )
    n_neighbors: int | None = Field(None, description="Number of neighbors used to construct the KNN graph.")
    threshold: float | None = Field(None, description="Doublet score threshold for calling a transcriptome a doublet.")
    verbose: bool = Field(True, description="If True, log progress updates.")
    copy: bool = Field(False, description="If True, return a copy of adata with Scrublet results added.")
    random_state: int = Field(0, description="Initial state for doublet simulation and nearest neighbors.")

    class Config:
        arbitrary_types_allowed = True


class ScrubletSimulateDoubletsParams(BaseModel):
    adata: str = Field(
        ..., description="Annotated data matrix of shape n_obs × n_vars. Rows correspond to cells, columns to genes."
    )
    layer: str | None = Field(
        None, description="Layer of adata where raw values are stored, or 'X' if values are in .X."
    )
    sim_doublet_ratio: float = Field(
        2.0, description="Number of doublets to simulate relative to the number of observed transcriptomes."
    )
    synthetic_doublet_umi_subsampling: float = Field(
        1.0,
        description="Rate for sampling UMIs when creating synthetic doublets. If 1.0, simply add UMIs from two randomly sampled transcriptomes.",
    )
    random_seed: int = Field(0, description="Random seed for reproducibility.")

    class Config:
        arbitrary_types_allowed = True
