# Open-source and Local LLMs

[Xorbits Inference](https://github.com/xorbitsai/inference) is an open-source
toolkit for running open-source models, particularly language models. To support
BioChatter applications in local and protected contexts, we provide API access
through the [LangChain OpenAI
Xinference](https://python.langchain.com/docs/integrations/llms/xinference)
module. Briefly, this module allows to connect to any open-source model
supported by Xinference via the state-of-the-art and easy-to-use OpenAI API.
This allows local and remote access to essentially all relevant open-source
models, including [these builtin
models](https://github.com/xorbitsai/inference#builtin-models), at very little
setup cost.

## Usage

Usage is essentially the same as when calling the official OpenAI API, but uses
the `XinferenceConversation` class under the hood. Interaction with the class is
possible in the exact same way as with the [standard class](chat.md).

## Connecting to the model from BioChatter

All that remains once Xinference has started your model is to tell BioChatter
the API endpoint of your deployed model via the `base_url` parameter of the
`XinferenceConversation` class. For instance:

```python
from biochatter.llm_connect import XinferenceConversation

conversation = XinferenceConversation(
         base_url="http://llm.biocypher.org",
         prompts={},
         correct=False,
     )
response, token_usage, correction = conversation.query("Hello world!")
```

## Deploying locally via Docker

We have created a Docker workflow that allows the deployment of builtin
Xinference models,
[here](https://github.com/biocypher/xinference-docker-builtin). It will soon be
available via Dockerhub. There is another workflow that allows mounting
(potentially) any compatible model from HuggingFace,
[here](https://github.com/AndiMajore/xinference-docker-hf). Note that, due to
graphics driver limitations, this currently only works for Linux machines with
dedicated Nvidia graphics cards. If you have a different setup, please check
below for deploying Xinference without the Docker workflow.

## Deploying locally without Docker

### Installation

To run Xinference locally on your computer or a workstation available on your
network, follow the [official
instructions](https://github.com/xorbitsai/inference) for your type of hardware.
Briefly, this includes installing the `xinference` and `ctransformers` Python
libraries into an environment of your choice, as well as a hardware-specific
installation of the `llama-ccp-python` library.

### Deploying your model

After installation, you can run the model
([locally](https://github.com/xorbitsai/inference#local) using `xinference` or
in a [distributed](https://github.com/xorbitsai/inference#distributed) fashion.
After startup, you can visit the local server address in your browser (standard
is `http://localhost:9997`) and select and start your desired model. There is a
large selection of predefined models to choose from, as well as the possibility
to add your own favourite models to the framework. You will see your running
models in the `Running Models` tab, once they have started.

Alternatively, you can deploy (and query) your model via the Xinference Python client:

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
