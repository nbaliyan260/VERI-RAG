"""Ingest pipeline: load documents → chunk → embed → index."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from veri_rag.config.schema import Chunk
from veri_rag.config.settings import Settings
from veri_rag.corpus.chunker import chunk_documents
from veri_rag.corpus.embedder import BaseEmbedder, TFIDFEmbedder, create_embedder
from veri_rag.corpus.loader import load_documents
from veri_rag.corpus.vector_store import BaseVectorStore, create_vector_store
from veri_rag.config.settings import get_project_root


class IndexArtifacts:
    """Loaded index state for retrieval."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        chunks: list[Chunk],
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks = chunks


def ingest_corpus(settings: Settings, data_dir: str | Path | None = None) -> IndexArtifacts:
    """Build or load the vector index from corpus settings."""
    root = get_project_root()
    data_dir = Path(data_dir or settings.corpus.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    processed = root / settings.corpus.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    index_path = root / settings.vector_store.index_path

    documents = load_documents(data_dir, settings.corpus.file_types)
    if not documents:
        raise ValueError(f"No documents found in {data_dir}")

    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        min_chunk_size=settings.chunking.min_chunk_size,
    )

    provider = settings.embedding.provider
    if provider == "sentence_transformers":
        provider = "sentence_transformers"

    embedder = create_embedder(
        provider=provider,
        model_name=settings.embedding.model_name,
        batch_size=settings.embedding.batch_size,
    )

    texts = [c.text for c in chunks]
    if isinstance(embedder, TFIDFEmbedder):
        embedder.fit(texts)
        with open(processed / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(embedder._vectorizer, f)

    embeddings = embedder.embed(texts)
    settings.embedding.dimension = embedder.dimension

    store = create_vector_store(
        provider=settings.vector_store.provider,
        dimension=embedder.dimension,
    )
    store.add(chunks, embeddings)
    store.save(index_path)

    manifest = {
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "embedding_provider": settings.embedding.provider,
        "embedding_model": settings.embedding.model_name,
    }
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return IndexArtifacts(vector_store=store, embedder=embedder, chunks=chunks)


def load_index(settings: Settings) -> IndexArtifacts:
    """Load a previously built index from disk."""
    root = get_project_root()
    index_path = root / settings.vector_store.index_path
    processed = root / settings.corpus.processed_dir

    provider = settings.embedding.provider
    embedder = create_embedder(
        provider=provider,
        model_name=settings.embedding.model_name,
        batch_size=settings.embedding.batch_size,
    )

    if isinstance(embedder, TFIDFEmbedder):
        vec_path = processed / "tfidf_vectorizer.pkl"
        if vec_path.exists():
            with open(vec_path, "rb") as f:
                from sklearn.feature_extraction.text import TfidfVectorizer

                embedder._vectorizer = pickle.load(f)
                embedder._is_fitted = True
                embedder._dimension = len(embedder._vectorizer.vocabulary_)

    store = create_vector_store(
        provider=settings.vector_store.provider,
        dimension=embedder.dimension,
    )
    store.load(index_path)

    chunks: list[Chunk] = []
    if hasattr(store, "get_all_chunks"):
        chunks = store.get_all_chunks()  # type: ignore[attr-defined]

    return IndexArtifacts(vector_store=store, embedder=embedder, chunks=chunks)
