"""Baseline RAG pipeline — the end-to-end query-answer flow."""

from __future__ import annotations

import time
from typing import Any

from veri_rag.config.schema import RAGAnswer, RetrievedChunk
from veri_rag.config.settings import Settings
from veri_rag.corpus.embedder import BaseEmbedder, create_embedder
from veri_rag.corpus.vector_store import BaseVectorStore, create_vector_store
from veri_rag.rag.generator import BaseLLM, create_llm
from veri_rag.rag.prompts import BASELINE_RAG_PROMPT, SAFE_PROMPT, build_prompt
from veri_rag.rag.retriever import Retriever


class BaselineRAG:
    """Standard RAG pipeline: retrieve → build prompt → generate answer."""

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        prompt_template: str = BASELINE_RAG_PROMPT,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.top_k = top_k

    def ask(
        self,
        query: str,
        query_id: str = "",
        top_k: int | None = None,
        exclude_chunk_ids: set[str] | None = None,
        chunks_override: list[RetrievedChunk] | None = None,
        prompt_template: str | None = None,
    ) -> RAGAnswer:
        """Run the full RAG pipeline for a single query.

        Args:
            query: User query string.
            query_id: Optional query identifier for tracking.
            top_k: Override the default top-k.
            exclude_chunk_ids: Chunks to exclude from retrieval.
            chunks_override: If provided, skip retrieval and use these chunks.
            prompt_template: Override the default prompt template.

        Returns:
            RAGAnswer with the generated answer and retrieved chunks.
        """
        start_time = time.time()

        # Step 1: Retrieve
        if chunks_override is not None:
            retrieved = chunks_override
        else:
            retrieved = self.retriever.retrieve(
                query, top_k=top_k or self.top_k, exclude_chunk_ids=exclude_chunk_ids
            )

        # Step 2: Build prompt
        template = prompt_template or self.prompt_template
        prompt = build_prompt(query, retrieved, template=template)

        # Step 3: Generate
        answer = self.llm.generate(prompt)

        latency_ms = (time.time() - start_time) * 1000

        return RAGAnswer(
            query_id=query_id or "unnamed",
            query=query,
            answer=answer,
            retrieved_chunks=retrieved,
            model_name=self.llm.model_name,
            prompt=prompt,
            latency_ms=latency_ms,
            token_count=self.llm.token_count,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        vector_store: BaseVectorStore | None = None,
        embedder: BaseEmbedder | None = None,
        llm: BaseLLM | None = None,
    ) -> "BaselineRAG":
        """Create a BaselineRAG instance from configuration settings."""
        if embedder is None:
            embedder = create_embedder(
                provider=settings.embedding.provider,
                model_name=settings.embedding.model_name,
                batch_size=settings.embedding.batch_size,
            )

        if vector_store is None:
            vector_store = create_vector_store(
                provider=settings.vector_store.provider,
                dimension=settings.embedding.dimension,
            )

        if llm is None:
            llm = create_llm(
                provider=settings.llm.provider,
                model_name=settings.llm.model_name,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )

        retriever = Retriever(
            vector_store=vector_store,
            embedder=embedder,
            top_k=settings.retrieval.top_k,
            score_threshold=settings.retrieval.score_threshold,
        )

        return cls(
            retriever=retriever,
            llm=llm,
            top_k=settings.retrieval.top_k,
        )
