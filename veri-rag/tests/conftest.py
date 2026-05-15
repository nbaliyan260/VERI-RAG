"""Pytest fixtures for VERI-RAG."""

from __future__ import annotations

import pytest

from veri_rag.config.settings import Settings, load_settings
from veri_rag.config.settings import get_project_root
from veri_rag.corpus.ingest import ingest_corpus
from veri_rag.corpus.synthetic import create_synthetic_corpus
from veri_rag.pipeline import VERIRAGPipeline


@pytest.fixture(scope="session")
def project_root():
    return get_project_root()


@pytest.fixture(scope="session")
def test_settings(project_root) -> Settings:
    return load_settings(project_root / "configs" / "mvp.yaml")


@pytest.fixture(scope="session")
def pipeline(project_root, test_settings) -> VERIRAGPipeline:
    create_synthetic_corpus(project_root / "data" / "synthetic_enterprise")
    index = ingest_corpus(test_settings)
    return VERIRAGPipeline(test_settings, index)
