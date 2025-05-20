# Installation

To use the package, install it from PyPI, for instance using pip (`pip install
biochatter`) or Poetry (`poetry add biochatter`).

## Extras

The package has some optional dependencies that can be installed using the
following extras (e.g. `pip install biochatter[xinference]`):

- `xinference`: support for querying open-source LLMs through Xorbits Inference

- `ollama`: support for querying open-source LLMs through Ollama

- `podcast`: support for podcast text-to-speech (for the free Google TTS; the
paid OpenAI TTS can be used without this extra)

- `streamlit`: support for streamlit UI functions (used in BioChatter Light)