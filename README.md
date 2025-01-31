# BioChatter

|     |     |     |     |
| --- | --- | --- | --- |
| __License__ | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | __Python__ | [![Python](https://img.shields.io/pypi/pyversions/biochatter)](https://www.python.org) |
| __Package__ | [![PyPI version](https://img.shields.io/pypi/v/biochatter)](https://pypi.org/project/biochatter/) [![Downloads](https://static.pepy.tech/badge/biochatter)](https://pepy.tech/project/biochatter) [![DOI](https://zenodo.org/badge/650181006.svg)](https://zenodo.org/doi/10.5281/zenodo.10777945) | __Build status__ | [![CI](https://github.com/biocypher/biochatter/actions/workflows/ci.yaml/badge.svg)](https://github.com/biocypher/biochatter/actions/workflows/ci.yaml) [![Docs](https://github.com/biocypher/biochatter/actions/workflows/docs.yaml/badge.svg)](https://github.com/biocypher/biochatter/actions/workflows/docs.yaml) |
| __Tests__ | [![Coverage](https://raw.githubusercontent.com/biocypher/biochatter/coverage/coverage.svg)](https://github.com/biocypher/biochatter/actions/workflows/ci.yaml) | __Docker__ | [![Latest image](https://img.shields.io/docker/v/biocypher/chatgse)](https://hub.docker.com/repository/docker/biocypher/chatgse/general) [![Image size](https://img.shields.io/docker/image-size/biocypher/chatgse/latest)](https://hub.docker.com/repository/docker/biocypher/chatgse/general) |
| __Development__ | [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) | __Contributions__ | [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CONTRIBUTING.md) |

## Description

🤖 BioChatter is a community-driven Python library that connects biomedical
applications to conversational AI, making it easy to leverage generative AI
models in the biomedical domain.

### 🌟 Key Features
- Generic backend for biomedical AI applications
- Seamless integration with multiple LLM providers
- Native connection to BioCypher knowledge graphs
- Extensive testing and evaluation framework
- Living benchmark of specific biomedical applications

### 🚀 Demo Applications and Utilities

- [BioChatter Light](https://light.biochatter.org) - Simple Python frontend
([repo](https://github.com/biocypher/biochatter-light))

- [BioChatter Next](https://next.biochatter.org) - Advanced Next.js frontend
([repo](https://github.com/biocypher/biochatter-next))

- [BioChatter Server](https://github.com/biocypher/biochatter-server) - RESTful
API server

📖 Learn more in our [paper](https://www.nature.com/articles/s41587-024-02534-3).

## Installation

To use the package, install it from PyPI, for instance using pip (`pip install
biochatter`) or Poetry (`poetry add biochatter`).

### Extras

The package has some optional dependencies that can be installed using the
following extras (e.g. `pip install biochatter[xinference]`):

- `xinference`: support for querying open-source LLMs through Xorbits Inference

- `podcast`: support for podcast text-to-speech (for the free Google TTS; the
paid OpenAI TTS can be used without this extra)

- `streamlit`: support for streamlit UI functions (used in BioChatter Light)

## Usage

Check out the [documentation](https://biochatter.org/) for examples, use cases,
and more information. Many common functionalities covered by BioChatter can be
seen in use in the [BioChatter
Light](https://github.com/biocypher/biochatter-light) code base.  [![Built with
Material for
MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)

## 🤝 Getting involved

We are very happy about contributions from the community, large and small!
If you would like to contribute to BioCypher development, please refer to
our [contribution guidelines](CONTRIBUTING.md) and the [developer
docs](DEVELOPER.md). :)

If you want to ask informal questions, talk about dev things, or just chat,
please join our community at https://biocypher.zulipchat.com!

> **Imposter syndrome disclaimer:** We want your help. No, really. There may be a little voice inside your head that is telling you that you're not ready, that you aren't skilled enough to contribute. We assure you that the little voice in your head is wrong. Most importantly, there are many valuable ways to contribute besides writing code.
>
> This disclaimer was adapted from the [Pooch](https://github.com/fatiando/pooch) project.

## More information about LLMs

Check out [this repository](https://github.com/csbl-br/awesome-compbio-chatgpt)
for more info on computational biology usage of large language models.

## Citation

If you use BioChatter in your work, please cite our
[paper](https://www.nature.com/articles/s41587-024-02534-3).
