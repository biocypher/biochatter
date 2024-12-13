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

from biochatter.api_agent.abc import BaseAPIModel, BaseQueryBuilder, BaseTools

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation


from typing import Optional
# Careful as this is not the same as the langchain_core.pydantic_v1
from langchain_core.pydantic_v1 import BaseModel, Field, create_model

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
class Tools(BaseTools):
    tools_params = {}
    tools_descriptions = {}
    
    tool_name = "read_h5ad"
    tools_descriptions[tool_name] = "Function to read h5ad files. Available in the anndata.io module"
    tools_params[tool_name] = {
        "filename": (str, Field(default="dummy.h5ad", description="Path to the .h5ad file")),
        "backed": (Optional[str], Field(default=None, description="Mode to access file: None, 'r' for read-only")),
        "as_sparse": (Optional[str], Field(default=None, description="Convert to sparse format: 'csr', 'csc', or None")),
        "as_sparse_fmt": (Optional[str], Field(default=None, description="Sparse format if converting, e.g., 'csr'")),
        "index_unique": (Optional[str], Field(default=None, description="Make index unique by appending suffix if needed"))
    }

    tool_name = "read_zarr" 
    tools_descriptions[tool_name] = "Function to read Zarr stores. Available in the anndata.io module"
    tools_params[tool_name] = {
        "filename": (str, Field(default="placeholder.zarr", description="Path or URL to the Zarr store"))
    }

    tool_name = "read_csv"
    tools_descriptions[tool_name] = "Function to read CSV files. Available in the anndata.io module"
    tools_params[tool_name] = {
        "filename": (str, Field(default="placeholder.csv", description="Path to the .csv file")),
        "delimiter": (Optional[str], Field(default=None, description="Delimiter used in the .csv file")),
        "first_column_names": (Optional[bool], Field(default=None, description="Whether the first column contains names"))
    }

    tool_name = "read_excel"
    tools_descriptions[tool_name] = "Function to read Excel files. Available in the anndata.io module"
    tools_params[tool_name] = {
        "filename": (str, Field(default="placeholder.xlsx", description="Path to the .xlsx file")),
        "sheet": (Optional[str], Field(default=None, description="Sheet name or index to read from")),
        "dtype": (Optional[str], Field(default=None, description="Data type for the resulting dataframe"))
    }

    tool_name = "read_hdf"
    tools_descriptions[tool_name] = "Function to read HDF5 files. Available in the anndata.io module"
    tools_params[tool_name] = {
        "filename": (str, Field(default="placeholder.h5", description="Path to the .h5 file")),
        "key": (Optional[str], Field(default=None, description="Group key within the .h5 file"))
    }

    tool_name = "read_loom"
    tools_descriptions[tool_name] = "Function to read Loom files. Available in the anndata.io module"
    tools_params[tool_name] = {
        "filename": (str, Field(default="placeholder.loom", description="Path to the .loom file")),
        "sparse": (Optional[bool], Field(default=None, description="Whether to read data as sparse")),
        "cleanup": (Optional[bool], Field(default=None, description="Clean up invalid entries")),
        "X_name": (Optional[str], Field(default=None, description="Name to use for X matrix")),
        "obs_names": (Optional[str], Field(default=None, description="Column to use for observation names")),
        "var_names": (Optional[str], Field(default=None, description="Column to use for variable names"))
    }

    tool_name = "read_mtx"
    tools_descriptions[tool_name] = "Function to read Matrix Market files. Available in the anndata.io module"
    tools_params[tool_name] = {
        "filename": (str, Field(default="placeholder.mtx", description="Path to the .mtx file")),
        "dtype": (Optional[str], Field(default=None, description="Data type for the matrix"))
    }

    tool_name = "read_text"
    tools_descriptions[tool_name] = "Function to read text files. Available in the anndata.io module"
    tools_params[tool_name] = {
        "filename": (str, Field(default="placeholder.txt", description="Path to the text file")),
        "delimiter": (Optional[str], Field(default=None, description="Delimiter used in the file")),
        "first_column_names": (Optional[bool], Field(default=None, description="Whether the first column contains names"))
    }
    def __init__(self, tools_params: dict = tools_params, tools_descriptions: dict = tools_descriptions):
        super().__init__()
        self.tools_params = tools_params
        self.tools_descriptions = tools_descriptions


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
        tool_maker = Tools()
        tools = tool_maker.make_pydantic_tools()
        runnable = self.create_runnable(
            conversation=conversation, query_parameters=tools
        )
        anndata_io_call_obj = runnable.invoke(
            question,
        )
        return anndata_io_call_obj
