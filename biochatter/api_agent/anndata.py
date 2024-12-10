### Aim
## For any question about anndata:
## return the code to answer the question

# 1. Read in anndata object from any anndata api supported format -> built-in anndata api
# 2. Concatenate the anndata object -> built-in anndata api
# 3. Filter the anndata object -> NumPy or SciPy sparse matrix api
# 4. Write the anndata object to [xxx] format -> built-in anndata api

import uuid
from collections.abc import Callable
from typing import Optional

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field

from .abc import BaseQueryBuilder

if TYPE_CHECKING:
    from biochatter.llm_connect import Conversation

from typing import TYPE_CHECKING

AnnDataIO_QUERY_PROMPT = """

"""


class AnnDataIOParameters(BaseModel):
    """Pydantic model for AnnData input/output operations.

    This class is used to configure and perform various AnnData I/O tasks,
    such as reading and writing files in different formats.
    """

    # Reading AnnData native formats
    read_h5ad: Optional[str] = Field(
        None,
        description="Path to the .h5ad-formatted HDF5 file. Use this to read an AnnData\
            object in .h5ad format.",
    )
    read_zarr: Optional[str] = Field(
        None,
        description="Path to a hierarchical Zarr array store to read AnnData data.",
    )
    # Reading other formats
    read_csv: Optional[str] = Field(
        None,
        description="Path to a .csv file to read into AnnData.",
    )
    read_excel: Optional[str] = Field(
        None,
        description="Path to an .xlsx (Excel) file to read into AnnData.",
    )
    excel_sheet: Optional[str] = Field(
        None,
        description="Sheet name to read from the .xlsx file.",
    )
    read_hdf: Optional[list] = Field(
        None,
        description="A sorted list where the first element is the path to the \
        .h5 (HDF5) file to read into AnnData. The second element is the key to the \
            data set, the user will input this and specify it as a string.",
    )
    operation_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the operation.",
    )


class AnndataIOQueryBuilder(BaseQueryBuilder):
    """A class for building a BlastQuery object."""

    def create_runnable(
        self,
        query_parameters: "AnnDataIOParameters",
        conversation: "Conversation",
    ) -> Callable:
        """Create a runnable object for executing queries.

        Creates a runnable using the LangChain
        `create_structured_output_runnable` method.

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
    ) -> AnnDataIOParameters:
        """Generate a BlastQuery object.

        Generates the object based on the given question, prompt, and
        BioChatter conversation. Uses a Pydantic model to define the API fields.
        Creates a runnable that can be invoked on LLMs that are qualified to
        parameterise functions.

        Args:
        ----
            question (str): The question to be answered.

            conversation: The conversation object used for parameterising the
                BlastQuery.

        Returns:
        -------
            BlastQuery: the parameterised query object (Pydantic model)

        """
        runnable = self.create_runnable(
            query_parameters=AnnDataIOParameters,
            conversation=conversation,
        )
        blast_call_obj = runnable.invoke(
            {"input": f"Answer:\n{question} based on:\n {AnnDataIO_QUERY_PROMPT}"},
        )
        blast_call_obj.question_uuid = str(uuid.uuid4())
        return blast_call_obj
