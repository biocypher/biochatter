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

# Dev Container

Due to some incompatibilities of `pymilvus` with Apple Silicon, we have created
a dev container for this project. To use it, you need to have Docker installed
on your machine. Then, you can run the devcontainer setup as recommended by
VSCode
[here](https://code.visualstudio.com/docs/remote/containers#_quick-start-open-an-existing-folder-in-a-container)
or using Docker directly.
