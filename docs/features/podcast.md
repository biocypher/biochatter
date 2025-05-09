# Podcast my Paper

We provide a module to perform document processing and text-to-speech to enable
listening to any document in podcast style.  The functionality can be accessed
through the podcast API or by running the script
`scripts/podcast_single_document.py`.

!!! note
    This feature is currently not under active development. In particular, due
    to the more recent involvement of large corporations in similar technologies
    (e.g., Google's NotebookLM), we are currently not prioritising this feature.

## API access

The podcast API is available through the `podcast` module.
An end-to-end workflow looks like this (modified from the test module):

```python
from biochatter.podcast import Podcaster
from biochatter.vectorstore import DocumentReader

# Load document
reader = DocumentReader()
document = reader.load_document("test/dcn.pdf")

# Initialise podcaster
podcaster = Podcaster(document)

# Generate podcast (LLM task)
podcaster.generate_podcast(characters_per_paragraph=5000)

# Employ text-to-speech to generate audio file (optional)
podcaster.podcast_to_file("test/test.mp3", model="tts-1-hd", voice="alloy")
```

If you do not want audio output, you can simply access the generated text though
the function `podcaster.podcast_to_text()`.

This example uses the paid OpenAI text-to-speech API to generate the audio file.
The default of the `podcast_to_file` function is to use the free Google
text-to-speech API.  When using OpenAI, due to the input length limit of 4096
characters, the podcast is split into multiple parts indicated by integer
suffixes.

## Command line access

To generate a podcast from a single document more quickly, you can use the
`scripts/podcast_single_document.py` script.  It accepts two arguments, the path
to the document and the path to the desired output file.  If the output file
ends in `.mp3`, the OpenAI text-to-speech API will be used to generate an audio
file.  Otherwise, the script will generate a text file and skip the
text-to-speech step.  If using the OpenAI text-to-speech API, multiple files
will be generated with integer suffixes.  If you installed BioChatter with
poetry, you can run the script like this (from the root directory of the
repository):

```bash
poetry run python scripts/podcast_single_document.py test/dcn.pdf test/test.mp3
```
