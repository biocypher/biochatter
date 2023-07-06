from langchain.schema import Document


class Podcaster:
    def __init__(self, document: Document) -> None:
        """
        Orchestrates the podcasting of a document.
        """
        self.document = document

    def podcast(self) -> None:
        """
        Podcasts the document.
        """
        print("Podcasting document...")
        print(self.document)
        print("Podcast done.")
