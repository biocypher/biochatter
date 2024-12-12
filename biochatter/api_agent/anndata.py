# Aim
# For any question about anndata:
# return the code to answer the question

# 1. Read in anndata object from any anndata api supported format -> built-in anndata api
# 2. Concatenate the anndata object -> built-in anndata api
# 3. Filter the anndata object -> NumPy or SciPy sparse matrix api
# 4. Write the anndata object to [xxx] format -> built-in anndata api

from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.output_parsers import PydanticToolsParser

# from langchain_core.pydantic_v1 import BaseModel, Field
from biochatter.llm_connect import Conversation

from .abc import BaseAPIModel, BaseQueryBuilder

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation


from pydantic import BaseModel, Field

ANNDATA_IO_QUERY_PROMPT = """
You are a world class algorithm, computational biologist with world leading knowledge
of the anndata package.
You are also an expert algorithm to return structured output formats.

You will be asked to provide code to answer a specific questions involving the anndata package.
NEVER return a code snippet or code itself, instead you have to return a structured output format.
You will have to create a structured output formats containing method:argument fields.

Here are the possible questions you might be asked:
<question: output> TBD
<question: output> TBD
<question: output> TBD

BASED ON THE DOCUMENTATION below:
### 1. Reading AnnData Native Formats
- **HDF5 (.h5ad):**
  `io.read_h5ad(filename[, backed, as_sparse, ...])`
  - Reads `.h5ad`-formatted HDF5 file.

- **Zarr:**
  `io.read_zarr(store)`
  - Reads from a hierarchical Zarr array store.

### 2. Reading Specific Portions of AnnData
- **Individual Elements (e.g., obs, varm, etc.):**
  `io.read_elem(elem)`
  - Reads an individual element from a store.

- **Backed Mode-Compatible Sparse Dataset:**
  `io.sparse_dataset(group)`
  - Generates a sparse dataset class compatible with backed mode.

### 3. Reading Non-Native Formats
#### 3.1 General Tips
- Non-native formats may not represent all aspects of AnnData objects.
- Assembling the AnnData object manually from individual parts may be more successful.

#### 3.2 Supported Formats
- **CSV:**
  `io.read_csv(filename[, delimiter, ...])`
  - Reads `.csv` file.

- **Excel (.xlsx):**
  `io.read_excel(filename, sheet[, dtype])`
  - Reads `.xlsx` (Excel) file.

- **HDF5 (.h5):**
  `io.read_hdf(filename, key)`
  - Reads `.h5` (HDF5) file.

- **Loom:**
  `io.read_loom(filename, *[, sparse, cleanup, ...])`
  - Reads `.loom`-formatted HDF5 file.

- **Matrix Market (.mtx):**
  `io.read_mtx(filename[, dtype])`
  - Reads `.mtx` file.

- **Text (.txt, .tab, .data):**
  `io.read_text(filename[, delimiter, ...])`
  - Reads `.txt`, `.tab`, or `.data` text files.

- **UMI Tools Matrix:**
  `io.read_umi_tools(filename[, dtype])`
  - Reads a gzipped condensed count matrix from UMI Tools.
"""


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
        ]
        runnable = self.create_runnable(
            conversation=conversation,
            query_parameters=tools,
        )
        anndata_io_call_obj = runnable.invoke(
            f"{question}",
        )
        return anndata_io_call_obj
