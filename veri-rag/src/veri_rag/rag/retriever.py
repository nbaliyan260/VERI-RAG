"""Retriever — wraps vector store search with scoring and filtering."""

from __future__ import annotations

import numpy as np

from veri_rag.config.schema import RetrievedChunk
from veri_rag.corpus.embedder import BaseEmbedder
from veri_rag.corpus.vector_store import BaseVectorStore


class Retriever:
    """Retrieve top-k chunks from a vector store given a query."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        self._store = vector_store
        self._embedder = embedder
        self._top_k = top_k
        self._score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        exclude_chunk_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: The user query string.
            top_k: Override the default top-k. If None, uses the instance default.
            exclude_chunk_ids: Set of chunk IDs to exclude from results.

        Returns:
            List of RetrievedChunk sorted by descending similarity.
        """
        k = top_k or self._top_k
        query_embedding = self._embedder.embed_query(query)

        # Retrieve more than needed if we're filtering
        fetch_k = k + len(exclude_chunk_ids or [])
        results = self._store.search(query_embedding, top_k=fetch_k)

        # Apply filters
        filtered: list[RetrievedChunk] = []
        for rc in results:
            if rc.similarity_score < self._score_threshold:
                continue
            if exclude_chunk_ids and rc.chunk.chunk_id in exclude_chunk_ids:
                continue
            filtered.append(rc)
            if len(filtered) >= k:
                break

        # Re-rank by position
        for i, rc in enumerate(filtered):
            rc.rank = i + 1

        return filtered

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get the embedding for a query (useful for risk scoring)."""
        return self._embedder.embed_query(query)
