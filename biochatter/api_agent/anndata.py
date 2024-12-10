# Aim
# For any question about anndata:
# return the code to answer the question

# 1. Read in anndata object from any anndata api supported format -> built-in anndata api
# 2. Concatenate the anndata object -> built-in anndata api
# 3. Filter the anndata object -> NumPy or SciPy sparse matrix api
# 4. Write the anndata object to [xxx] format -> built-in anndata api

ANNDATA_IO_QUERY_PROMPT = """
You are a world class algorithm for creating queries in structured formats.
Your task is to use the Python API of `anndata` to answer questions about the `anndata` io module.
You should prefix the function calls with `anndata.io.`, for instance, for reading a file, you should use
`anndata.io.read`.
"""
