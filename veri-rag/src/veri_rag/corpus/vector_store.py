"""Vector store abstraction with FAISS and in-memory backends."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from veri_rag.config.schema import Chunk, RetrievedChunk


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the store."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        """Search for the most similar chunks to a query embedding."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the index to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load the index from disk."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Look up a chunk by its ID (if supported)."""
        return None


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store using inner-product search on L2-normalised vectors."""

    def __init__(self, dimension: int = 384):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        self._faiss = faiss
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)  # cosine via normalised vectors
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        assert embeddings.shape[0] == len(chunks), "Mismatch between chunks and embeddings"
        assert embeddings.shape[1] == self._dimension, (
            f"Embedding dim {embeddings.shape[1]} != store dim {self._dimension}"
        )
        self._index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        if self._index.ntotal == 0:
            return []
        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: list[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    rank=rank + 1,
                    similarity_score=float(score),
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / "index.faiss"))
        # Save chunk metadata alongside
        chunks_data = [c.model_dump(mode="json") for c in self._chunks]
        with open(path / "chunks.json", "w") as f:
            json.dump(chunks_data, f, indent=2, default=str)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._index = self._faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.json", "r") as f:
            chunks_data = json.load(f)
        self._chunks = [Chunk(**d) for d in chunks_data]

    def count(self) -> int:
        return self._index.ntotal

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for c in self._chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def get_all_chunks(self) -> list[Chunk]:
        return list(self._chunks)


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory cosine-similarity store using numpy. No external deps."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._embeddings: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if self._embeddings is None:
            self._embeddings = embeddings.astype(np.float32)
        else:
            self._embeddings = np.vstack(
                [self._embeddings, embeddings.astype(np.float32)]
            )
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        query = query_embedding.reshape(1, -1).astype(np.float32)
        # Cosine similarity (vectors assumed L2-normalised)
        scores = (self._embeddings @ query.T).flatten()
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(top_indices):
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    rank=rank + 1,
                    similarity_score=float(scores[idx]),
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._embeddings is not None:
            np.save(str(path / "embeddings.npy"), self._embeddings)
        chunks_data = [c.model_dump(mode="json") for c in self._chunks]
        with open(path / "chunks.json", "w") as f:
            json.dump(chunks_data, f, indent=2, default=str)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        emb_path = path / "embeddings.npy"
        if emb_path.exists():
            self._embeddings = np.load(str(emb_path))
        with open(path / "chunks.json", "r") as f:
            chunks_data = json.load(f)
        self._chunks = [Chunk(**d) for d in chunks_data]

    def count(self) -> int:
        return len(self._chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for c in self._chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def get_all_chunks(self) -> list[Chunk]:
        return list(self._chunks)


def create_vector_store(
    provider: str = "faiss", dimension: int = 384
) -> BaseVectorStore:
    """Factory function for creating a vector store."""
    if provider == "faiss":
        return FAISSVectorStore(dimension=dimension)
    elif provider == "memory":
        return InMemoryVectorStore(dimension=dimension)
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")
