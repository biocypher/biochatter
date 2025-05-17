"""API Agent package for BioChatter.

This package provides various API agents for interacting with bioinformatics
tools and services, including BLAST, OncoKB, BioTools, and Scanpy interfaces.
"""

from .base.agent_abc import BaseFetcher, BaseInterpreter, BaseQueryBuilder
from .base.api_agent import APIAgent
from .base.formatters import format_as_python_call, format_as_rest_call
from .python.anndata_agent import AnnDataIOQueryBuilder
from .python.generic_agent import GenericQueryBuilder
from .python.scanpy_pl_full import ScanpyPlQueryBuilder
from .python.scanpy_pl_reduced import ScanpyPlQueryBuilder as ScanpyPlQueryBuilderReduced
from .python.scanpy_pp_reduced import ScanpyPpQueryBuilder as ScanpyPpQueryBuilderReduced
from .web.bio_tools import BioToolsFetcher, BioToolsInterpreter, BioToolsQueryBuilder
from .web.blast import (
    BlastFetcher,
    BlastInterpreter,
    BlastQueryBuilder,
    BlastQueryParameters,
)
from .web.oncokb import OncoKBFetcher, OncoKBInterpreter, OncoKBQueryBuilder

__all__ = [
    "APIAgent",
    "AnnDataIOQueryBuilder",
    "BaseFetcher",
    "BaseInterpreter",
    "BaseQueryBuilder",
    "BioToolsFetcher",
    "BioToolsInterpreter",
    "BioToolsQueryBuilder",
    "BlastFetcher",
    "BlastInterpreter",
    "BlastQueryBuilder",
    "BlastQueryParameters",
    "GenericQueryBuilder",
    "OncoKBFetcher",
    "OncoKBInterpreter",
    "OncoKBQueryBuilder",
    "ScanpyPlQueryBuilder",
    "ScanpyPlQueryBuilderReduced",
    "ScanpyPpQueryBuilderReduced",
    "format_as_python_call",
    "format_as_rest_call",
]
