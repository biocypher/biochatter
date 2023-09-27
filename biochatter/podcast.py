from typing import List
from langchain.schema import Document
from .llm_connect import GptConversation
from gtts import gTTS
import nltk
import os

FIRST_PROMPT = (
    "You are tasked with summarising a scientific manuscript for consumption as"
    "a podcast. As a first step, extract title and authors of the manuscript"
    "from the following text. Return them in the format 'Title: <title>,"
    "Authors: <authors>'."
)

PROMPT = (
    "You are tasked with summarising a scientific manuscript for consumption as"
    "a podcast. You will receive a collection of sentences from the"
    "manuscript, from which you will remove any information not relevant to the"
    "content, such as references, figure legends, tables, author information, "
    "journal metadata, and so on. You will then be asked to summarise the"
    "section of the manuscript, making the wording more suitable for listening."
    "Remove all content in brackets that is of technical nature, such as"
    "p-values, statistical tests, and so on. If the given text contains only "
    "literature references, return 'No content'."
)


class Podcaster:
    def __init__(self, document: Document) -> None:
        """
        Orchestrates the podcasting of a document.
        """
        self.document = document

    def generate_podcast(self, characters_per_paragraph: int) -> None:
        """
        Podcasts the document.

        TODO:
        - chain of density prompting for variable summary length
        - do not summarise but just make readable
        """
        full_text = self.document[0].page_content

        # split text by sentence
        sentences = self._split_text(full_text)

        # could embed sentences and cluster on cosine similarity to identify
        # paragraphs here

        # preprocess text
        for i, sentence in enumerate(sentences):
            # special cases i.e. and e.g. - if sentence ends with one of these,
            # append next sentence
            special_cases = ["i.e.", "e.g."]
            if sentence.endswith(tuple(special_cases)):
                sentences[i] = sentence + " " + sentences[i + 1]
                del sentences[i + 1]

        # concatenate first 5 sentences for title and author extraction
        first_5 = "\n".join(sentences[:5])
        self.podcast_intro = self._title_and_authors(first_5)

        # LLM to determine section breaks?

        # go through sections and summarise each
        self.summarised_sections = self._summarise_sections(
            sentences,
            characters_per_paragraph,
        )

        # summarise the summaries

    def _split_text(self, text: str) -> List[str]:
        """
        Splits consecutive text into sentences.
        """
        nltk.download("punkt")
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        return tokenizer.tokenize(text)

    def _title_and_authors(self, text: str) -> str:
        """
        Extracts title and authors from document.

        Args:
            text (str): text to extract title and authors from

        Returns:
            str: title and authors
        """
        # first sentence - extract title, authors
        c_first = GptConversation(
            model_name="gpt-3.5-turbo",
            prompts={},
            correct=False,
        )
        c_first.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="podcast")
        c_first.append_system_message(FIRST_PROMPT)
        msg, token_usage, correction = c_first.query(text)
        # split at authors ('Authors:' or '\nAuthors:')
        title = msg.split("Title:")[1].split("Authors:")[0].strip()
        authors = msg.split("Authors:")[1].strip()
        return f"{title}, by {authors}, podcasted by biochatter."

    def _summarise_section(self, text: str) -> str:
        """
        Summarises a section of the document.

        Args:
            text (str): text to summarise

        Returns:
            str: summarised text
        """
        # summarise section
        c = GptConversation(
            model_name="gpt-3.5-turbo",
            prompts={},
            correct=False,
        )
        c.set_api_key(api_key=os.getenv("OPENAI_API_KEY"), user="podcast")
        c.append_system_message(PROMPT)
        msg, token_usage, correction = c.query(text)
        return msg

    def _summarise_sections(
        self, sentences: list, characters_per_paragraph: int
    ) -> list:
        """

        Summarises sections of the document. Concatenates sentences until
        characters_per_paragraph is reached, removing each sentence from the
        list as it is added to the section to be summarised.

        Args:
            sentences (list): list of sentences to summarise
            characters_per_paragraph (int): number of characters per paragraph

        Returns:
            list: list of summarised sections
        """
        summarised_sections = []
        section = ""
        while sentences:
            sentence = sentences.pop(0)
            tmp = section + sentence
            if len(tmp) < characters_per_paragraph and sentences:
                section += sentence
            else:
                if sentences:
                    sentences.insert(0, sentence)
                summarised_section = self._summarise_section(section)
                summarised_sections.append(summarised_section)
                section = ""

        return summarised_sections

    def podcast_to_file(self, path: str) -> None:
        """
        Uses text-to-speech to generate audio for the summarised paper podcast.

        Args:
            path (str): path to save audio file to
        """

        full_text = self.podcast_to_text()

        audio = gTTS(text=full_text)
        audio.save(path)

    def podcast_to_text(self):
        """
        Returns the summarised paper podcast as text.
        """
        full_text = self.podcast_intro + "\n\n"
        for section in self.summarised_sections:
            full_text += section + "\n\n"
        return full_text
