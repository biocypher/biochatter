---
title: BioChatter - Conversational AI for Biomedical Applications
description: A framework for deploying, testing, and evaluating conversational AI models in the biomedical domain.
---
# Home

Generative AI models have shown tremendous usefulness in increasing
accessibility and automation of a wide range of tasks. Yet, their application to
the biomedical domain is still limited, in part due to the lack of a common
framework for deploying, testing, and evaluating the diverse models and
auxiliary technologies that are needed. `biochatter` is a Python package
implementing a generic backend library for the connection of biomedical
applications to conversational AI. We describe the framework in [this
preprint](https://arxiv.org/abs/2305.06488); for a more hands-on experience,
check out our two web app implementations:

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } &nbsp; __BioChatter Light__

    ---

    Agile framework in pure Python built with [Streamlit](https://streamlit.io),
    for fast prototyping and iteration.

    [:octicons-arrow-right-24: Go To BioChatter Light](https://light.biochatter.org)

-   :fontawesome-solid-wand-magic-sparkles:{ .lg .middle } &nbsp; __BioChatter Next__

    ---

    Advanced client-server architecture based on
    [Flask](https://flask.palletsprojects.com) and
    [Next.js](https://nextjs.org).

    [:octicons-arrow-right-24: Go To BioChatter Next](https://next.biochatter.org)

</div>

BioChatter is part of the [BioCypher](https://github.com/biocypher) ecosystem,
connecting natively to BioCypher knowledge graphs. The BioChatter paper is being
written [here](https://github.com/biocypher/biochatter-paper) and the current
version can be read [here](https://biocypher.github.io/biochatter-paper).

![BioChatter Overview](images/biochatter_overview.png)

!!! note "Hot Topics"

    BioChatter natively extends [BioCypher](https://biocypher.org) knowledge
    graphs. Check there for more information.

    We have also recently published a perspective on connecting knowledge and
    machine learning to enable causal reasoning in biomedicine, with a
    particular focus on the currently emerging "foundation models." You can read
    it [here](https://arxiv.org/abs/2401.09558).

## Installation

To use the package, install it from PyPI, for instance using pip (`pip install
biochatter`) or Poetry (`poetry add biochatter`).

### Extras

The package has some optional dependencies that can be installed using the
following extras (e.g. `pip install biochatter[xinference]`):

- `xinference`: support for querying open-source LLMs through Xorbits Inference

- `ollama`: support for querying open-source LLMs through Ollama

- `podcast`: support for podcast text-to-speech (for the free Google TTS; the
paid OpenAI TTS can be used without this extra)

- `streamlit`: support for streamlit UI functions (used in ChatGSE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/pypi/pyversions/biochatter)](https://www.python.org) [![PyPI version](https://img.shields.io/pypi/v/biochatter)](https://pypi.org/project/biochatter/) [![Downloads](https://static.pepy.tech/badge/biochatter)](https://pepy.tech/project/biochatter) [![CI](https://github.com/biocypher/biochatter/actions/workflows/ci.yaml/badge.svg)](https://github.com/biocypher/biochatter/actions/workflows/ci.yaml) [![Latest image](https://img.shields.io/docker/v/biocypher/chatgse)](https://hub.docker.com/repository/docker/biocypher/chatgse/general) [![Image size](https://img.shields.io/docker/image-size/biocypher/chatgse/latest)](https://hub.docker.com/repository/docker/biocypher/chatgse/general) [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/biocypher/biochatter/blob/main/CONTRIBUTING.md)
