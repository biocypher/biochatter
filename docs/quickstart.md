# Quickstart

BioChatter is a versatile framework that can be used in various ways depending
on your needs and technical background. This guide will help you get started
based on your user profile and intended use case.

## Python Developer Profile

If you're a Python developer looking to integrate LLM capabilities into your
biomedical application:

### Basic Installation

```bash
pip install biochatter
```

### Core Usage Example

```python
from biochatter.llm_connect import GptConversation

# Initialize conversation
conversation = GptConversation(
    model_name="gpt-3.5-turbo",
    prompts={},
)
conversation.set_api_key(api_key="your-api-key")

# Query the model
response, token_usage, correction = conversation.query("Your biomedical question here")
```

This skeleton example is simply passing through the question to the LLM, which
is not recommended according to our [envisioned use](about/use-cases.md). We
recommend integrating at least one of the advanced features and dedicated
prompts for model instructions.

### Advanced Features

1. **Knowledge Graph Connectivity**: Connect to a [BioCypher](/biocypher) knowledge graph:
```python
from biochatter.prompts import BioCypherPromptEngine
from biochatter.llm_connect import GptConversation

# Create a conversation factory
def create_conversation():
    conversation = GptConversation(model_name="gpt-3.5-turbo", prompts={})
    conversation.set_api_key(api_key="your-api-key")
    return conversation

# Initialize the prompt engine with your BioCypher schema
prompt_engine = BioCypherPromptEngine(
    schema_config_or_info_path="path/to/schema_info.yaml",
    # or use schema_config_or_info_dict to pass the schema as a dictionary
    conversation_factory=create_conversation,
)

# Generate a Cypher query based on the question
cypher_query = prompt_engine.generate_query(
    question=question,
    query_language="Cypher"  # defaults to Cypher if not specified
)
```

The `BioCypherPromptEngine` handles:
- Entity selection based on your schema
- Relationship selection between entities
- Property selection for entities and relationships
- Query generation in your chosen query language

For a complete example of KG integration, check out our [Knowledge Graph
vignette](vignettes/kg.md).

2. **API Integration**: Connect to biological databases and APIs:
```python
from biochatter.api_agent.base.api_agent import APIAgent
from biochatter.api_agent.web.oncokb import OncoKBQueryBuilder, OncoKBFetcher, OncoKBInterpreter
from biochatter.llm_connect import GptConversation

# Create a conversation factory function
def create_conversation():
    conversation = GptConversation(
        model_name="gpt-3.5-turbo",  # or your preferred model
        prompts={},
        correct=False
    )
    conversation.set_api_key(api_key="your-api-key")
    return conversation

# Create API agent with OncoKB components
agent = APIAgent(
    conversation_factory=create_conversation,  # Function to create new conversations
    query_builder=OncoKBQueryBuilder(),       # Builds queries for OncoKB API
    fetcher=OncoKBFetcher(),                 # Handles API requests
    interpreter=OncoKBInterpreter()          # Interprets API responses
)

# Execute query - this will:
# 1. Build an appropriate OncoKB query
# 2. Fetch results from the OncoKB API
# 3. Interpret the results using the LLM
result = agent.execute("What is the oncogenic potential of BRAF V600E mutation?")
```

The API Agent architecture allows you to:
- Create structured queries for specific APIs
- Handle API requests and responses
- Interpret results using LLMs
- Support multiple API types (REST, Python, etc.)

For more examples of supported APIs and custom implementations, check our [API
documentation](api-docs/api-calling-base.md).

3. **Vector Database Integration**: For semantic search and RAG capabilities:
```python
from biochatter.vectorstore import DocumentReader, DocumentEmbedder
from langchain_openai import OpenAIEmbeddings

# Initialize document reader and embedder
reader = DocumentReader()

# Create embedder with Milvus as vector store
embedder = DocumentEmbedder(
    embedding_collection_name="your_embeddings",
    metadata_collection_name="your_metadata",
    connection_args={"host": "localhost", "port": "19530"}
)
embedder.connect()

# Load and embed a document
document = reader.load_document("path/to/your/document.pdf")  # Supports PDF and TXT
doc_id = embedder.save_document(document)

# Perform similarity search
results = embedder.similarity_search(
    query="Your search query here",
    k=3  # Number of results to return
)

# Clean up when needed
embedder.remove_document(doc_id)
```

This workflow allows:
- Document chunking with customizable parameters
- Metadata storage and retrieval
- Similarity search and retrieval

For more details on vector database integration, including advanced features and
configurations, check our [RAG documentation](features/rag.md).

## Streamlit GUI Developer Profile

For developers who want to create a user-friendly web interface quickly:

1. Clone the [BioChatter Light](https://github.com/biocypher/biochatter-light) repository
1. Install dependencies (Poetry recommended)
1. Set up debugging environment using `streamlit run app.py`
1. Modify the app components to introduce your desired functionality (refer to the [customisation vignette](vignettes/custom-bclight-advanced.md) for more details)

### Running via the docker image

We provide a Docker image for BioChatter Light, and we are always happy for
contributions. If you have an idea for a generally useful feature or panel,
please get in touch (e.g., open an issue). Once the feature has been added to
the BioChatter Light repository, it will be available via the official docker
image, potentially as an optional tab activated via environment variables in
the Docker setup (see the [vignette](vignettes/custom-bclight-advanced.md) for
details). This allows access to the feature in many environments without the
need for local installation, for instance using:

```bash
docker run -d -p 8501:8501 biocypher/biochatter-light --env-file .env
```
## REST API / Next.js Developer Profile

For developers building production-grade web applications:

### Components

1. **Backend (FastAPI)**:

    - Uses [BioChatter Server](https://github.com/biocypher/biochatter-server)
    for the REST API
   
    - Handles LLM interactions, database connections, and business logic

2. **Frontend (Next.js)**:

    - Uses [BioChatter Next](https://github.com/biocypher/biochatter-next)

    - Provides a modern, responsive UI

### Getting Started with Docker

```bash
# Clone the repository
git clone https://github.com/biocypher/biochatter-next
```

Configure the Next application, modifying the YAML configuration file. For an
example, check the
[example](https://github.com/biocypher/decider-genetics/blob/main/config/biochatter-next.yaml)
from our [Decider Genetics](vignettes/custom-decider-use-case.md) use case.

```bash
# Start the services
docker-compose up -d
```

## Open-Source Model Users

For users who prefer to use local, open-source LLMs:

### Using Ollama

```bash
pip install "biochatter[ollama]"
```

Running the Ollama software on port 11434:

```python
from biochatter.llm_connect import OllamaConversation

conversation = OllamaConversation(
    base_url="http://localhost:11434",
    prompts={},
    model_name='llama2',
    correct=False,
)
response, token_usage, correction = conversation.query("Your question here")
```

### Using Xinference

```bash
pip install "biochatter[xinference]"
```

Running the Xinference software on port 9997:

```python
from biochatter.llm_connect import XinferenceConversation

conversation = XinferenceConversation(
    base_url="http://localhost:9997",
    prompts={},
    correct=False,
)
response, token_usage, correction = conversation.query("Your question here")
```

