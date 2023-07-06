from biochatter.podcast import Podcaster

from biochatter.vectorstore import DocumentReader


def test_podcast():
    reader = DocumentReader()
    document = reader.load_document("test/dcn.pdf")
    podcaster = Podcaster(document)
    podcaster.podcast(characters_per_paragraph=5000)
    assert False
