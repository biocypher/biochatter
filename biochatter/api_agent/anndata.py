### Aim
## For any question about anndata:
## return the code to answer the question

# 1. Read in anndata object from any anndata api supported format -> built-in anndata api
# 2. Concatenate the anndata object -> built-in anndata api
# 3. Filter the anndata object -> NumPy or SciPy sparse matrix api
# 4. Write the anndata object to [xxx] format -> built-in anndata api

import uuid
from typing import Optional

from pydantic import BaseModel, Field

from .abc import BaseQueryBuilder


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

