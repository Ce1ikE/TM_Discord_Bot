from sentence_transformers import SentenceTransformer

from typing import Iterable, Literal

# the embedder is mostly a wrapper around the SentenceTransformer library
# to provide a uniform interface for embedding generation
# the only thing we need to do here is take in a list of texts and return a list of embeddings (high dimensional vector representations)
class Embedder:
    """Thin wrapper around SentenceTransformer for text embedding generation."""

    def __init__(
        self,
        model_name: Literal[
            "sentence-transformers/all-MiniLM-L6-v2",
            "ibm-granite/granite-embedding-278m-multilingual",
            "ibm-granite/granite-embedding-107m-multilingual"
        ] = "ibm-granite/granite-embedding-278m-multilingual",
    ):
        self.model_name = model_name 
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: Iterable[str]):
        """Embed a collection of texts and return vectors."""
        embeddings = []
        for text in texts:
            embeddings.append(self.model.encode(
                    text,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )
            )

        return embeddings
