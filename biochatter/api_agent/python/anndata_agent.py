"""Module for generating anndata queries using LLM tools."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel, Field

from biochatter.api_agent.base.agent_abc import BaseAPIModel, BaseQueryBuilder
from biochatter.llm_connect import Conversation

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

ANNDATA_IO_QUERY_PROMPT = """
You are a world class algorithm, computational biologist with world leading knowledge
of the anndata package.
You are also an expert algorithm to return structured output formats.

You will be asked to provide code to answer a specific questions involving the anndata package.
NEVER return a code snippet or code itself, instead you have to return a structured output format.
You will have to create a structured output formats containing method:argument fields.

You will be asked to read in an anndata object from any anndata api supported format OR
to concatenate the anndata objects.
FOR THE MapAnnData, BE SURE TO ALWAYS USE THE variable of the anndata GIVEN IN THE INPUT, REPLACE IT IN THE method_name
Use the tools available:
"""


class ConcatenateAnnData(BaseAPIModel):
    """Concatenate AnnData objects along an axis."""

    method_name: str = Field(default="anndata.concat", description="NEVER CHANGE")
    adatas: list | dict = Field(
        ...,
        description=(
            "The objects to be concatenated. "
            "Either a list of AnnData objects or a mapping of keys to AnnData objects."
        ),
    )
    axis: str = Field(
        default="obs",
        description="Axis to concatenate along. Can be 'obs' (0) or 'var' (1). Default is 'obs'.",
    )
    join: str = Field(
        default="inner",
        description="How to align values when concatenating. Options: 'inner' or 'outer'. Default is 'inner'.",
    )
    merge: str | Callable | None = Field(
        default=None,
        description=(
            "How to merge elements not aligned to the concatenated axis. "
            "Strategies include 'same', 'unique', 'first', 'only', or a callable function."
        ),
    )
    uns_merge: str | Callable | None = Field(
        default=None,
        description="How to merge the .uns elements. Uses the same strategies as 'merge'.",
    )
    label: str | None = Field(
        default=None,
        description="Column in axis annotation (.obs or .var) to place batch information. Default is None.",
    )
    keys: list | None = Field(
        default=None,
        description=(
            "Names for each object being concatenated. "
            "Used for column values or appended to the index if 'index_unique' is not None. "
            "Default is None."
        ),
    )
    index_unique: str | None = Field(
        default=None,
        description="Delimiter for making the index unique. When None, original indices are kept.",
    )
    fill_value: Any | None = Field(
        default=None,
        description="Value used to fill missing indices when join='outer'. Default behavior depends on array type.",
    )
    pairwise: bool = Field(
        default=False,
        description="Include pairwise elements along the concatenated dimension. Default is False.",
    )


class MapAnnData(BaseAPIModel):
    """Apply mapping functions to elements of AnnData."""

    method_name: str = Field(
        default="anndata.obs|var['annotation_name'].map",
        description=(
            "ALWAYS ALWAYS ALWAYS REPLACE THE anndata BY THE ONE GIVEN BY THE INPUT"
            "Specifies the AnnData attribute and operation being performed. "
            "For example, 'obs.map' applies a mapping function or dictionary to the specified column in `adata.obs`. "
            "This must always include the AnnData component and the `.map` operation. "
            "Adapt the component (e.g., 'obs', 'var', etc.) to the specific use case."
        ),
    )
    dics: dict | None = Field(default=None, description="Dictionary to map over.")


class ReadH5AD(BaseAPIModel):
    """Read .h5ad-formatted hdf5 file."""

    method_name: str = Field(default="io.read_h5ad", description="NEVER CHANGE")
    filename: str = Field(default="dummy.h5ad", description="Path to the .h5ad file")
    backed: str | None = Field(
        default=None,
        description="Mode to access file: None, 'r' for read-only",
    )
    as_sparse: str | None = Field(
        default=None,
        description="Convert to sparse format: 'csr', 'csc', or None",
    )
    as_sparse_fmt: str | None = Field(
        default=None,
        description="Sparse format if converting, e.g., 'csr'",
    )
    index_unique: str | None = Field(
        default=None,
        description="Make index unique by appending suffix if needed",
    )


class ReadZarr(BaseAPIModel):
    """Read from a hierarchical Zarr array store."""

    method_name: str = Field(default="io.read_zarr", description="NEVER CHANGE")
    filename: str = Field(
        default="placeholder.zarr",
        description="Path or URL to the Zarr store",
    )


class ReadCSV(BaseAPIModel):
    """Read .csv file."""

    method_name: str = Field(default="io.read_csv", description="NEVER CHANGE")
    filename: str = Field(
        default="placeholder.csv",
        description="Path to the .csv file",
    )
    delimiter: str | None = Field(
        None,
        description="Delimiter used in the .csv file",
    )
    first_column_names: bool | None = Field(
        None,
        description="Whether the first column contains names",
    )


class ReadExcel(BaseAPIModel):
    """Read .xlsx (Excel) file."""

    method_name: str = Field(default="io.read_excel", description="NEVER CHANGE")
    filename: str = Field(
        default="placeholder.xlsx",
        description="Path to the .xlsx file",
    )
    sheet: str | None = Field(None, description="Sheet name or index to read from")
    dtype: str | None = Field(
        None,
        description="Data type for the resulting dataframe",
    )


class ReadHDF(BaseAPIModel):
    """Read .h5 (hdf5) file."""

    method_name: str = Field(default="io.read_hdf", description="NEVER CHANGE")
    filename: str = Field(default="placeholder.h5", description="Path to the .h5 file")
    key: str | None = Field(None, description="Group key within the .h5 file")


class ReadLoom(BaseAPIModel):
    """Read .loom-formatted hdf5 file."""

    method_name: str = Field(default="io.read_loom", description="NEVER CHANGE")
    filename: str = Field(
        default="placeholder.loom",
        description="Path to the .loom file",
    )
    sparse: bool | None = Field(None, description="Whether to read data as sparse")
    cleanup: bool | None = Field(None, description="Clean up invalid entries")
    X_name: str | None = Field(None, description="Name to use for X matrix")
    obs_names: str | None = Field(
        None,
        description="Column to use for observation names",
    )
    var_names: str | None = Field(
        None,
        description="Column to use for variable names",
    )


class ReadMTX(BaseAPIModel):
    """Read .mtx file."""

    method_name: str = Field(default="io.read_mtx", description="NEVER CHANGE")
    filename: str = Field(
        default="placeholder.mtx",
        description="Path to the .mtx file",
    )
    dtype: str | None = Field(None, description="Data type for the matrix")


class ReadText(BaseAPIModel):
    """Read .txt, .tab, .data (text) file."""

    method_name: str = Field(default="io.read_text", description="NEVER CHANGE")
    filename: str = Field(
        default="placeholder.txt",
        description="Path to the text file",
    )
    delimiter: str | None = Field(None, description="Delimiter used in the file")
    first_column_names: bool | None = Field(
        None,
        description="Whether the first column contains names",
    )


class AnnDataIOQueryBuilder(BaseQueryBuilder):
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
                AnnDataIOQuery.

        Returns:
        -------
            AnnDataIOQuery: the parameterised query object (Pydantic model)

        """
        tools = [
            ReadCSV,
            ReadExcel,
            ReadH5AD,
            ReadHDF,
            ReadLoom,
            ReadMTX,
            ReadText,
            ReadZarr,
            ConcatenateAnnData,
            MapAnnData,
        ]
        runnable = self.create_runnable(
            conversation=conversation,
            query_parameters=tools,
        )
        query = [
            ("system", ANNDATA_IO_QUERY_PROMPT),
            ("human", f"{question}"),
        ]
        return runnable.invoke(
            query,
        )
