from biochatter.podcast import Podcaster
from biochatter.vectorstore import DocumentReader


def test_podcast():
    # runs long, requires OpenAI API key
    reader = DocumentReader()
    document = reader.load_document("test/dcn.pdf")
    podcaster = Podcaster(document)
    podcaster.generate_podcast(characters_per_paragraph=5000)
    podcaster.podcast_to_file("test/test.mp3")
