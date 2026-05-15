"""Embedding provider abstraction with SentenceTransformers and TF-IDF backends."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            2-D numpy array of shape (len(texts), dimension).
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns:
            1-D numpy array of shape (dimension,).
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimensionality."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding using the sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts, batch_size=self._batch_size, show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class TFIDFEmbedder(BaseEmbedder):
    """Lightweight TF-IDF fallback when sentence-transformers is unavailable."""

    def __init__(self, max_features: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._is_fitted = False
        self._dimension = max_features

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._is_fitted = True
        self._dimension = len(self._vectorizer.vocabulary_)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._is_fitted:
            self.fit(texts)
        sparse = self._vectorizer.transform(texts)
        dense = sparse.toarray().astype(np.float32)
        # L2 normalise
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    def embed_query(self, query: str) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("TFIDFEmbedder must be fit before querying.")
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedder(provider: str = "sentence_transformers", **kwargs) -> BaseEmbedder:
    """Factory function for creating an embedder.

    Args:
        provider: One of 'sentence_transformers' or 'tfidf'.
    """
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_name=kwargs.get("model_name", "all-MiniLM-L6-v2"),
            batch_size=kwargs.get("batch_size", 32),
        )
    elif provider == "tfidf":
        max_features = int(kwargs.get("dimension", kwargs.get("max_features", 384)))
        return TFIDFEmbedder(max_features=max_features)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
