"""API Agent package for BioChatter.

This package provides various API agents for interacting with bioinformatics
tools and services, including BLAST, OncoKB, BioTools, and Scanpy interfaces.
"""

from .abc import BaseFetcher, BaseInterpreter, BaseQueryBuilder
from .anndata import AnnDataIOQueryBuilder
from .api_agent import APIAgent
from .bio_tools import BioToolsFetcher, BioToolsInterpreter, BioToolsQueryBuilder
from .blast import (
    BlastFetcher,
    BlastInterpreter,
    BlastQueryBuilder,
    BlastQueryParameters,
)
from .formatters import format_as_python_call, format_as_rest_call
from .oncokb import OncoKBFetcher, OncoKBInterpreter, OncoKBQueryBuilder
from .scanpy_pl import ScanpyPlQueryBuilder
from .scanpy_tl import ScanpyTlQueryBuilder

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
    "OncoKBFetcher",
    "OncoKBInterpreter",
    "OncoKBQueryBuilder",
    "ScanpyPlQueryBuilder",
    "ScanpyTlQueryBuilder",
    "format_as_python_call",
    "format_as_rest_call",
]
