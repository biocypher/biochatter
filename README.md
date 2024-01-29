# BioChatter

|     |     |     |     |
| --- | --- | --- | --- |
| __License__ | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | __Python__ | [![Python](https://img.shields.io/pypi/pyversions/biochatter)](https://www.python.org) |
| __Package__ | [![PyPI version](https://img.shields.io/pypi/v/biochatter)](https://pypi.org/project/biochatter/) [![Downloads](https://static.pepy.tech/badge/biochatter)](https://pepy.tech/project/biochatter) | __Build status__ | [![CI](https://github.com/biocypher/biochatter/actions/workflows/ci.yaml/badge.svg)](https://github.com/biocypher/biochatter/actions/workflows/ci.yaml) [![Docs](https://github.com/biocypher/biochatter/actions/workflows/docs.yaml/badge.svg)](https://github.com/biocypher/biochatter/actions/workflows/docs.yaml) |
| __Tests__ | Coverage coming soon. | __Docker__ | [![Latest image](https://img.shields.io/docker/v/biocypher/chatgse)](https://hub.docker.com/repository/docker/biocypher/chatgse/general) [![Image size](https://img.shields.io/docker/image-size/biocypher/chatgse/latest)](https://hub.docker.com/repository/docker/biocypher/chatgse/general) |
| __Development__ | [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/) | __Contributions__ | [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CONTRIBUTING.md) |

## Description

Generative AI models have shown tremendous usefulness in increasing
accessibility and automation of a wide range of tasks. Yet, their application to
the biomedical domain is still limited, in part due to the lack of a common
framework for deploying, testing, and evaluating the diverse models and
auxiliary technologies that are needed.  This repository contains the
`biochatter` Python package, a generic backend library for the connection of
biomedical applications to conversational AI.  Described in [this
preprint](https://arxiv.org/abs/2305.06488) and used in
[ChatGSE](https://chat.biocypher.org), which is being developed at
https://github.com/biocypher/ChatGSE. More to come, so stay tuned!

BioChatter is part of the [BioCypher](https://github.com/biocypher) ecosystem, 
connecting natively to BioCypher knowledge graphs. The BioChatter paper is
being written [here](https://github.com/biocypher/biochatter-paper).

## Installation

To use the package, install it from PyPI, for instance using pip (`pip install
biochatter`) or Poetry (`poetry add biochatter`).

### Extras

The package has some optional dependencies that can be installed using the
following extras (e.g. `pip install biochatter[xinference]`):

- `xinference`: support for querying open-source LLMs through Xorbits Inference

- `podcast`: support for podcast text-to-speech (for the free Google TTS; the
paid OpenAI TTS can be used without this extra)

- `streamlit`: support for streamlit UI functions (used in ChatGSE)

## Usage

Check out the [documentation](https://biochatter.org/) for
examples, use cases, and more information. Many common functionalities covered
by BioChatter can be seen in use in the
[ChatGSE](https://github.com/biocypher/ChatGSE) code base.
[![Built with Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)

# More information about LLMs

Check out [this repository](https://github.com/csbl-br/awesome-compbio-chatgpt)
for more info on computational biology usage of large language models.

## Developer notes

If you're on Apple Silicon, you may encounter issues with the `grpcio`
dependency (`grpc` library, which is used in `pymilvus`). If so, try to install
the binary from source after removing the installed package from the virtual
environment from
[here](https://stackoverflow.com/questions/72620996/apple-m1-symbol-not-found-cfrelease-while-running-python-app):

```bash
pip uninstall grpcio
export GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation"
pip install grpcio==1.53.0 --no-binary :all:
```