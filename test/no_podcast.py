from biochatter.podcast import Podcaster
from biochatter.vectorstore import DocumentReader
import os
import pytest


@pytest.mark.skip(
    reason=(
        "More of an integration test without asserts. "
        "Currently used for debugging, to be later replaced."
    )
)
def test_podcast():
    # runs long, requires OpenAI API key
    reader = DocumentReader()
    document = reader.load_document("test/dcn.pdf")
    podcaster = Podcaster(document)
    podcaster.generate_podcast(characters_per_paragraph=5000)
    podcaster.podcast_to_file("test/test.mp3")


def test_podcast_to_text():
    # create a Podcaster object with a sample document
    reader = DocumentReader()
    document = reader.load_document("test/dcn.pdf")
    podcaster = Podcaster(document)

    # set fixed text for podcast intro and summarised sections
    podcaster.podcast_info = "This is a podcast intro."
    podcaster.processed_sections = [
        "This is the first section.",
        "This is the second section.",
    ]

    # test the podcast_to_text method
    expected_text = (
        "You are listening to: "
        + podcaster.podcast_info
        + "\n\n"
        + "\n\n".join(podcaster.processed_sections)
        + "\n\n"
    )
    assert podcaster.podcast_to_text() == expected_text


@pytest.mark.skip(reason=("Temporary skip, gtts is slow."))
def test_podcast_to_file_gtts(tmpdir):
    # create a Podcaster object with fixed text
    reader = DocumentReader()
    document = reader.load_document("test/dcn.pdf")
    podcaster = Podcaster(document)
    podcaster.podcast_info = "This is a podcast intro."
    podcaster.processed_sections = [
        "This is the first section.",
        "This is the second section.",
    ]

    # generate a temporary file path
    path = os.path.join(tmpdir, "test.mp3")

    # test the podcast_to_file method
    podcaster.podcast_to_file(path)
    assert os.path.isfile(path)
