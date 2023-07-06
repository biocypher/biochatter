from biochatter.podcast import Podcaster

from biochatter.vectorstore import DocumentEmbedder


def test_podcast():
    de = DocumentEmbedder()
    de._load_document("test/bc_summary.txt")
    podcaster = Podcaster(de.document)
    podcaster.podcast()
    assert False
