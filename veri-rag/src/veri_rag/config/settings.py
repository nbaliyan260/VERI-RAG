"""Configuration loader — reads YAML configs and exposes typed settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    strategy: str = "fixed"
    chunk_size: int = 300
    chunk_overlap: int = 50
    min_chunk_size: int = 50


class EmbeddingConfig(BaseModel):
    provider: str = "sentence_transformers"
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    dimension: int = 384


class VectorStoreConfig(BaseModel):
    provider: str = "faiss"
    index_path: str = "data/processed/faiss_index"
    similarity_metric: str = "cosine"


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.0


class LLMConfig(BaseModel):
    provider: str = "mock"
    model_name: str = "mock-llm-v1"
    temperature: float = 0.0
    max_tokens: int = 512


class RiskWeights(BaseModel):
    instruction: float = 0.25
    sensitive: float = 0.25
    retrieval_anomaly: float = 0.15
    conflict: float = 0.15
    coordination: float = 0.10
    stance_shift: float = 0.10


class RiskThresholds(BaseModel):
    suspicious: float = 0.60
    high_risk: float = 0.80


class RiskScoringConfig(BaseModel):
    weights: RiskWeights = Field(default_factory=RiskWeights)
    thresholds: RiskThresholds = Field(default_factory=RiskThresholds)


class InfluenceWeights(BaseModel):
    semantic_answer_change: float = 0.35
    unsafe_score_drop: float = 0.30
    leakage_drop: float = 0.20
    citation_dependency: float = 0.15


class InfluenceThresholds(BaseModel):
    harmful: float = 0.50
    critical: float = 0.65


class InfluenceConfig(BaseModel):
    method: str = "riaa"
    weights: InfluenceWeights = Field(default_factory=InfluenceWeights)
    thresholds: InfluenceThresholds = Field(default_factory=InfluenceThresholds)
    risk_filter_threshold: float = 0.40
    enable_pairwise: bool = True
    max_pairwise_tests: int = 10
    pairwise_threshold: float = 0.15
    calibrator_path: str = "data/processed/harmfulness_calibrator.pkl"


class CertifiedSmoothingConfig(BaseModel):
    enabled: bool = True
    num_samples: int = 8
    subset_fraction: float = 0.75
    confidence_alpha: float = 0.05
    stability_threshold: float = 0.6


class RepairConfig(BaseModel):
    max_replacement_chunks: int = 3
    source_diversity_min: int = 2
    preserve_utility: bool = True


class CertificateWeights(BaseModel):
    grounding: float = 0.25
    stability: float = 0.20
    no_leakage: float = 0.20
    low_attacker_dependency: float = 0.15
    source_diversity: float = 0.10
    instruction_isolation: float = 0.10


class VerificationConfig(BaseModel):
    certificate_weights: CertificateWeights = Field(default_factory=CertificateWeights)
    pass_threshold: float = 0.60
    certified_smoothing: CertifiedSmoothingConfig = Field(
        default_factory=CertifiedSmoothingConfig
    )


class OutputsConfig(BaseModel):
    base_dir: str = "outputs"
    answers_dir: str = "outputs/answers"
    certificates_dir: str = "outputs/certificates"
    provenance_dir: str = "outputs/provenance_graphs"
    results_dir: str = "outputs/experiment_results"
    reports_dir: str = "outputs/reports"
    hpc_runs_dir: str = "outputs/hpc_runs"
    cache_dir: str = "cache"


class CorpusConfig(BaseModel):
    data_dir: str = "data/synthetic_enterprise"
    processed_dir: str = "data/processed"
    file_types: list[str] = Field(default_factory=lambda: [".txt", ".md", ".jsonl"])


class Settings(BaseModel):
    """Top-level settings parsed from a YAML config file."""
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    risk_scoring: RiskScoringConfig = Field(default_factory=RiskScoringConfig)
    influence: InfluenceConfig = Field(default_factory=InfluenceConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)


def load_settings(config_path: str | Path) -> Settings:
    """Load settings from a YAML file, merging with defaults."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Remove 'project' key which is metadata, not settings
    raw.pop("project", None)

    return Settings(**raw)


def get_project_root() -> Path:
    """Return the project root directory (where pyproject.toml lives)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def ensure_output_dirs(settings: Settings) -> None:
    """Create all output directories if they don't exist."""
    root = get_project_root()
    for field_name in OutputsConfig.model_fields:
        dir_path = root / getattr(settings.outputs, field_name)
        dir_path.mkdir(parents=True, exist_ok=True)
