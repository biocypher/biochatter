# biochatter

This repository contains the `biochatter` Python package, a generic backend
library for the connection of biomedical applications to conversational AI.
Described in [this preprint](https://arxiv.org/abs/2305.06488) and used in
[ChatGSE](https://chat.biocypher.org), which is being developed at
https://github.com/biocypher/ChatGSE. More to come, so stay tuned!

## Installation

To use the package, install it from PyPI, for instance using pip (`pip install
biochatter`) or Poetry (`poetry add biochatter`).

### Extras

The package has some optional dependencies that can be installed using the
following extras (e.g. `pip install biochatter[streamlit]`):

- `streamlit`: support for streamlit UI functions (used in ChatGSE)
- `podcast`: support for podcast text-to-speech

## Usage

As an interim documentation until we have a proper one, check out the
[Wiki](https://github.com/biocypher/biochatter/wiki) for some usage examples.
Many common functionalities covered by BioChatter can be seen in use in the
[ChatGSE](https://github.com/biocypher/ChatGSE) code base.

# More information about LLMs

Check out [this repository](https://github.com/csbl-br/awesome-compbio-chatgpt)
for more info on computational biology usage of large language models.

# Dev Container

Due to some incompatibilities of `pymilvus` with Apple Silicon, we have created
a dev container for this project. To use it, you need to have Docker installed
on your machine. Then, you can run the devcontainer setup as recommended by
VSCode
[here](https://code.visualstudio.com/docs/remote/containers#_quick-start-open-an-existing-folder-in-a-container)
or using Docker directly.

The dev container expects an environment file (there are options, but the basic
one is `.devcontainer/local.env`) with the following variables:

```
OPENAI_API_KEY=(sk-...)
DOCKER_COMPOSE=true
DEVCONTAINER=true
```

To test vector database functionality, you also need to start a Milvus
standalone server. You can do this by running `docker-compose up` as described
[here](https://milvus.io/docs/install_standalone-docker.md) on the host machine
(not from inside the devcontainer).
