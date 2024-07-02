# Open-source and Local LLMs

There are two self-hosted/local LLM solutions that BioChatter currently supports
out-of-the-box.

- [Xorbits Inference](https://github.com/xorbitsai/inference)
- [Ollama](https://ollama.com/)

Below, we provide installation and usage instructions for both of them.

## Xorbits Inference (Xinference)

[Xorbits Inference](https://github.com/xorbitsai/inference) is an open-source
toolkit for running open-source models, particularly language models. To support
BioChatter applications in local and protected contexts, we provide API access
via BioChatter classes in a unified way. Briefly, this module allows to connect
to any open-source model supported by Xinference via the state-of-the-art and
easy-to-use OpenAI API. This allows local and remote access to essentially all
relevant open-source models, including [these builtin
models](https://inference.readthedocs.io/en/latest/models/builtin/index.html),
at very little setup cost.

### Usage

Usage is essentially the same as when calling the official OpenAI API, but uses
the `XinferenceConversation` class under the hood. Interaction with the class is
possible in the exact same way as with the [standard class](chat.md).

### Connecting to the model from BioChatter

All that remains once Xinference has started your model is to tell BioChatter
the API endpoint of your deployed model via the `base_url` parameter of the
`XinferenceConversation` class. For instance:

```python
from biochatter.llm_connect import XinferenceConversation

conversation = XinferenceConversation(
    base_url="http://localhost:9997",
    prompts={},
    correct=False,
)
response, token_usage, correction = conversation.query("Hello world!")
```

### Deploying locally via Docker

We have created a Docker workflow that allows the deployment of builtin
Xinference models,
[here](https://github.com/biocypher/xinference-docker-builtin). It will soon be
available via Dockerhub. There is another workflow that allows mounting
(potentially) any compatible model from HuggingFace,
[here](https://github.com/AndiMajore/xinference-docker-hf). Note that, due to
graphics driver limitations, this currently only works for Linux machines with
dedicated Nvidia graphics cards. If you have a different setup, please check
below for deploying Xinference without the Docker workflow.

### Deploying locally without Docker

#### Installation

To run Xinference locally on your computer or a workstation available on your
network, follow the [official
instructions](https://github.com/xorbitsai/inference) for your type of hardware.
Briefly, this includes installing the `xinference` and `ctransformers` Python
libraries into an environment of your choice, as well as a hardware-specific
installation of the `llama-ccp-python` library.

#### Deploying your model

After installation, you can run the model
([locally](https://github.com/xorbitsai/inference#local) using `xinference` or
in a [distributed](https://github.com/xorbitsai/inference#distributed) fashion.
After startup, you can visit the local server address in your browser (standard
is `http://localhost:9997`) and select and start your desired model. There is a
large selection of predefined models to choose from, as well as the possibility
to add your own favourite models to the framework. You will see your running
models in the `Running Models` tab, once they have started.

Alternatively, you can deploy (and query) your model via the Xinference Python
client:

```python
from xinference.client import Client

client = Client("http://localhost:9997")
model_uid = client.launch_model(model_name="chatglm2")  # download model from HuggingFace and deploy
model = client.get_model(model_uid)

chat_history = []
prompt = "What is the largest animal?"
model.chat(
    prompt,
    chat_history,
    generate_config={"max_tokens": 1024}
)
```

## Ollama

[Ollama](https://ollama.com/) is arguably the biggest open-source project for
local LLM hosting right now. In comparison to Xinference it lacks the complete
freedom of running any HuggingFace model in a simple fashion, but has the
benefit of higher stability for the supported models. The list of [supported
models](https://ollama.com/library) is updated diligently by the Ollama
community. BioChatter support was added by implementing the [LangChain
ChatOllama](https://python.langchain.com/v0.2/docs/integrations/chat/ollama/)
and [LangChain
OllamaEmbeddings](https://python.langchain.com/v0.2/docs/integrations/text_embedding/ollama/)
classes, connecting to Ollama APIs.

### Usage

Usage is essentially the same as when calling the official OpenAI API, but uses
the `OllamaConversation` class under the hood. Interaction with the class is
possible in the exact same way as with the [standard class](chat.md).

### Connecting to the model from BioChatter

Once Ollama has been set up (see below), you can directly use BioChatter to
connect to the API endpoint and start any
[available](https://ollama.com/library) model. It will be downloaded and
launched on-demand. You can now configure the `OllamaConversation` instance
setting the `base_url` and `model_name` parameters. For example:

```python
from biochatter.llm_connect import OllamaConversation

conversation = OllamaConversation(
    base_url="http://localhost:11434",
    prompts={},
    model_name='llama3',
    correct=False,
)
response, token_usage, correction = conversation.query("Hello world!")
```

### Deploying locally via Docker

To deploy Ollama with Docker is extremely easy and well documented. You can
follow the official [Ollama Docker blog
post](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image)
for that or check the [Ollama DockerHub
page](https://hub.docker.com/r/ollama/ollama) that will also help you with the
installation of the required `nvidia-container-toolkit` library if you want to
use GPUs from Docker containers.

### Deploying locally without Docker

#### Installation
You can download and run Ollama also directly on your computer. For this you can
just visit the [official website](https://ollama.com/download) that provides you
with an installer for any OS. More info on the setup and startup process can be
found in the [GitHub
README](https://github.com/ollama/ollama/blob/main/README.md).