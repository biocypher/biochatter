import os

from biochatter.api_agent import (
    ReadCSV,
    ReadExcel,
    ReadH5AD,
    ReadHDF,
    ReadLoom,
    ReadMTX,
    ReadText,
    ReadZarr,
)

os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

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
from langchain_core.output_parsers import PydanticToolsParser

from biochatter.api_agent.anndata import ANNDATA_IO_QUERY_PROMPT

llm = ChatOpenAI(model="gpt-4")

query = 'I want to read the h5ad file called "example_file.h5ad".'
print("running chain...")
llm_with_tools = llm.bind_tools(tools)

chain = llm_with_tools | PydanticToolsParser(tools=tools)
res = chain.invoke(query)

# Assuming `res[0]` is a Pydantic model instance
method = res[0].__class__.__name__  # Get the name of the Pydantic class
method_arguments = res[0].dict(
    exclude_none=True
)  # Convert the model attributes to a dictionary

# Create the final dictionary
result_dict = {method: method_arguments}

# Print the result
print(result_dict)
