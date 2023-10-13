from typing import List, Optional, Any

import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import _create_retry_decorator
from pydantic import BaseModel
from langchain.embeddings.base import Embeddings


def embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        kwargs["api_type"] = "open_ai"
        kwargs["model"] = kwargs["engine"]
        kwargs["engine"] = None
        return embeddings.client.create(**kwargs)

    return _embed_with_retry(**kwargs)


class GenericOpenAIEmbeddings(OpenAIEmbeddings, BaseModel, Embeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deployment = kwargs["model"]

        # please refer to
        # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb

    def _get_len_safe_embeddings(
            self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for i in range(len(texts))]
        try:
            import tiktoken

            tokens = []
            indices = []
            # encoding = tiktoken.model.encoding_for_model(self.model)
            for i, text in enumerate(texts):
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
                for disallowed in self.disallowed_special:
                    text = text.replace(disallowed, "")
                token = text.split(" ")
                # token = encoding.encode(
                #     text,
                #     allowed_special=self.allowed_special,
                #     disallowed_special=self.disallowed_special,
                # )
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens += [token[j: j + self.embedding_ctx_length]]
                    indices += [i]

            batched_embeddings = []
            _chunk_size = chunk_size or self.chunk_size
            for i in range(0, len(tokens), _chunk_size):
                print([" ".join(t) for t in tokens[i: i + _chunk_size]])
                response = embed_with_retry(
                    self,
                    input=[" ".join(t) for t in tokens[i: i + _chunk_size]],
                    engine=self.deployment,
                )
                batched_embeddings += [r["embedding"] for r in response["data"]]

            results: List[List[List[float]]] = [[] for i in range(len(texts))]
            lens: List[List[int]] = [[] for i in range(len(texts))]
            for i in range(len(indices)):
                results[indices[i]].append(batched_embeddings[i])
                lens[indices[i]].append(len(batched_embeddings[i]))

            for i in range(len(texts)):
                average = np.average(results[i], axis=0, weights=lens[i])
                embeddings[i] = (average / np.linalg.norm(average)).tolist()

            return embeddings

        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        # handle large input text
        if self.embedding_ctx_length > 0:
            return self._get_len_safe_embeddings([text], engine=engine)[0]
        else:
            # replace newlines, which can negatively affect performance.
            text = text.replace("\n", " ")
            return embed_with_retry(self, input=[text], engine=engine)["data"][0][
                "embedding"
            ]

    def embed_documents(
            self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        # handle batches of large input text
        if self.embedding_ctx_length > 0:
            return self._get_len_safe_embeddings(texts, engine=self.deployment)
        else:
            results = []
            _chunk_size = chunk_size or self.chunk_size
            for i in range(0, len(texts), _chunk_size):
                response = embed_with_retry(
                    self,
                    input=texts[i: i + _chunk_size],
                    engine=self.deployment,
                )
                results += [r["embedding"] for r in response["data"]]
            return results

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = self._embedding_func(text, engine=self.deployment)
        return embedding
