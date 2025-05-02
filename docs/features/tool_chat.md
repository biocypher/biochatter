# In-Chat Tool Calling

## Overview

In addition to [Ad Hoc API calling](api.md), BioChatter provides a framework for
in-chat tool calling. This lets you integrate external tools directly into
conversations with the LLM, enabling functionality beyond the model's built-in
capabilities. Typical uses include:

- Performing complex calculations  
- Accessing external databases or APIs  
- Executing custom scripts or code  

Many commercial LLMs (e.g., OpenAI, Anthropic, Google, Mistral) support tool
calling natively.  Through [ollama](https://ollama.com/), BioChatter also
supports a variety of open-source models that are natively capable of tool
calling (e.g. [mistral 7b](https://ollama.com/library/mistral), 
[mistral-small3.1](https://ollama.com/library/mistral-small3.1),
[qwen2.5](https://ollama.com/library/qwen2.5),
[cogito](https://ollama.com/library/cogito)).

For models without native tool calling, BioChatter provides a fallback by
parameterizing the tool call in the prompt and then calling the tool via its own
API. [See below](#tool-calling-for-non-native-models) for more details.

## Defining a tool

BioChatter uses the
[LangChain](https://python.langchain.com/docs/concepts/tools/) framework to
define and manage tools. You can create a new tool by decorating a function with
the `@tool` decorator, for example:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
   """Multiply two numbers."""
   return a * b

@tool
def add(a: int, b: int) -> int:
   """Add two numbers."""
   return a + b
```

LangChain exposes the tool's description and signature to the model, allowing
the LLM to understand and use the tool within a chat session. For more
information on customizing tools, see the [LangChain
documentation](https://python.langchain.com/docs/how_to/custom_tools/).

## Passing a tool to the Chat

BioChatter’s `LangChainConversation` class implements in-chat tool calling.
Internally, it uses LangChain’s `init_chat_model` function (see
[documentation](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)),
allowing a consistent interface for loading various LLM providers and models.  

You can make tools available to the model in two ways:

#### 1. At the beginning of the conversation

```python
#import the conversation class
from biochatter.conversation import LangChainConversation

# Tools definition is recycled from the previous section

# Define the question
question = "What is 334*54? And what about 345+123?"

# Gather the tools into a list
tools = [multiply, add]

# Define the conversation
convo = LangChainConversation(
    model_provider="google_genai", 
    model_name="gemini-2.0-flash",
    prompts={},
    tools=tools #<------
)

# Set the API key (read from the environment variable, based on the model provider)
convo.set_api_key()

# Run the conversation
convo.query(question)
```

Tools passed in this way remain accessible for **the entire conversation**.

#### 2. For the current query

```python
#import the conversation class
from biochatter.conversation import LangChainConversation

# Tools definition is recycled from the previous section

# Define the question
question = "What is 334*54? And what about 345+123?"

# Define the conversation
convo = LangChainConversation(
    model_provider="google_genai", 
    model_name="gemini-2.0-flash",
    prompts={},
)

# Set the API key (read from the environment variable, based on the model provider)
convo.set_api_key()

# Run the conversation
convo.query(question, tools=tools) #<------
```

Tools passed in this way are available only for **the current query**.

## Tool calling modalities

When starting a conversation, you can specify one of two tool-calling modes:

- `"auto"`: If the model returns a tool call (or multiple tool calls), the
tool(s) is/are automatically executed, and the result is added to the
conversation history as a `ToolMessage`.  

- `"text"`: If the model returns a tool call, the arguments for the tool are
returned as text and stored in the conversation history as an `AIMessage`.

By default, the tool calling mode is `"auto"`.

```python
# Automatic tool calling
convo = LangChainConversation(
    model_provider="google_genai", 
    model_name="gemini-2.0-flash",
    prompts={},
    tool_calling_mode="auto" #<------
)

# Textual tool calling
convo = LangChainConversation(
    model_provider="google_genai", 
    model_name="gemini-2.0-flash",
    prompts={},
    tool_calling_mode="text" #<------
)
```

## Automatic tool call interpretation

BioChatter also allows you to interpret a tool’s output automatically by
specifying `explain_tool_result=True` when you query the model. This is
particularly helpful when:

- The tool call returns large or complex data that you want summarized.
- The tool provides context (e.g., RAG) that should inform the final answer.

For example, consider the following tool that performs enrichment analysis on a
list of genes:

```python
#import the needed libraries
from gseapy import enrichr

#define the new tool
@tool
def enrichr_query(gene_list: list[str]):
    """Run enrichment analysis on a list of genes.

    This tool allows to run enrichment analysis on a list of genes using the `gseapy` library.
    Using this tool, a model can get information about the biological processes enriched in a set of genes.
    
    Args:
        gene_list: list of genes to run enrichment analysis on

    Returns:
        DataFrame: DataFrame containing the enrichment results
    """
    # Run enrichment
    enr = enrichr(
        gene_list=gene_list,
        gene_sets='GO_Biological_Process_2021',
        organism='Human',
        outdir=None,  # no files will be written
        cutoff=0.05
    )

    # Save results as DataFrame
    df_results = enr.results

    return df_results
```

After defining the tool, you can enable automatic interpretation in the
conversation:

```python
#initialize the conversation
convo = LangChainConversation(
    model_provider="google_genai", 
    model_name="gemini-2.0-flash",
    prompts={},
)

#set the API key
convo.set_api_key()

#define the question
question = "What biological processes are regulated by TP53, BRCA1, BRCA2, PTEN, EGFR, MYC, CDK2, CDK4, CCND1, RB1?"

#run the conversation
convo.query(question, tools=[enrichr_query], explain_tool_result=True)

#print the answer
print(convo.messages[-1].content)
```

By default, the model attempts to interpret any tool output it receives when you
set `explain_tool_result=True`. You can further customize its interpretation by
passing dedicated prompts through the `query` method:

```python
general_instructions_tool_interpretation = "Your tool interpretation here..."
additional_instructions_tool_interpretation = "Your additional tool interpretation here..."

#run the conversation
convo.query(
    question,
    tools=[enrichr_query],
    explain_tool_result=True,
    general_instructions_tool_interpretation=general_instructions_tool_interpretation,
    additional_instructions_tool_interpretation=additional_instructions_tool_interpretation
)
```

If you wish to inspect the defaults for these prompts, you can print them from
the conversation object:

```python
print(convo.general_instructions_tool_interpretation)
print(convo.additional_instructions_tool_interpretation)
```

## MCP tools

BioChatter supports the [Model Context Protocol
(MCP)](https://modelcontextprotocol.io/introduction) for tool calling through
the `langchain_mcp_adapters` library. Below is a simple example of defining an
MCP server and integrating its tools into your chat.

First, define the MCP server using the `FastMCP` class:

```python title="math_server.py"
# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Next, run the MCP server and provide the tools to the model. This involves
asynchronous operations, so the code uses `async with` statements and `await`
keywords:

```python
# import the needed libraries
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from pathlib import Path
import sys

#patch event loop if running in a notebook
if 'ipykernel' in sys.modules:
    import nest_asyncio
    nest_asyncio.apply()

# define the path to the math_server.py file
server_path = Path('your_path_here')

# define the server parameters
server_params = StdioServerParameters(
    command="python",
    args=[server_path/"math_server.py"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        
        # Initialize the connection
        await session.initialize()

        # define the question
        question = "What is 2 times 2?"

        # Get tools
        tools = await load_mcp_tools(session)

        # define the conversation
        convo = LangChainConversation(
            model_provider="google_genai", 
            model_name="gemini-2.0-flash",
            prompts={},
            tools=tools,
            mcp=True
        )

        # set the API key
        convo.set_api_key()

        # invoke the model
        convo.query(question)
```

## Tool calling for non-native models

For models that do not natively support tool calling, BioChatter provides a
simple workaround by automatically generating a prompt that leverages tools
signatures and descriptions describing to the model how to use the tools.

The interface for calling the tool is the same as in the native case and works
also for MCP tools. Sometimes, it might be necessary to provide additional
information to the model in the prompt to help it correctly use the tools. This
can be done in two ways:

#### 1. by providing additional instructions in the class constructor

```python
#initialize the conversation
convo = LangChainConversation(
    model_provider="ollama", 
    model_name="gemma3:27b",
    prompts={},
    tools=#[your tool list here],
    additional_tools_instructions="...Here your additional instructions..."
)

#set the API key
convo.set_api_key()

#define the question
question = "...Here your question..."

#run the conversation
convo.query(question,)
```

#### 2. by providing additional instructions in the `query` method

```python
#initialize the conversation
convo = LangChainConversation(
    model_provider="ollama", 
    model_name="gemma3:27b",
    prompts={},
    tools=#[your tool list here]
)

#set the API key
convo.set_api_key()

#define the question
question = "...Here your question..."

#run the conversation
convo.query(question,additional_tools_instructions="...Here your additional instructions...")
```
Note that by specifying additional instructions in the `query` method you will
override any instructions provided in the class constructor. 

**Remark:** Given that this is a simple workaround, sometimes it might not be
sufficient to add additional instructions in the prompt to get the model to
correctly use the tools. In this case, we suggest to look at [Ad Hoc API
calling](api.md) as a more robust solution.