from biochatter.podcast import Podcaster

from biochatter.vectorstore import DocumentReader


def test_podcast():
    reader = DocumentReader()
    document = reader.load_document("test/bc_summary.txt")
    podcaster = Podcaster(document)
    podcaster.podcast()
    assert False
