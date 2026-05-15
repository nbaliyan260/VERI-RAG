"""Pydantic data models for the entire VERI-RAG pipeline.

These models define the typed data flow between all modules:
corpus → retrieval → detection → influence → repair → verification.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AttackType(str, Enum):
    """Supported attack types."""
    POISONING = "poisoning"
    PROMPT_INJECTION = "prompt_injection"
    SECRET_LEAKAGE = "secret_leakage"
    BLOCKER = "blocker"
    TOPIC_FLIP = "topic_flip"
    ADAPTIVE = "adaptive"


class RepairActionType(str, Enum):
    """Types of repair actions the engine can take."""
    QUARANTINE = "quarantine"
    DOWNRANK = "downrank"
    SANITIZE = "sanitize"
    REDACT = "redact"
    REPLACE = "replace"
    DIVERSIFY = "diversify"
    REMOVE_BLOCKER = "remove_blocker"
    BALANCE_EVIDENCE = "balance_evidence"


class CertificateStatus(str, Enum):
    """Overall certificate status."""
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"


# ---------------------------------------------------------------------------
# Corpus & Retrieval Models
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """A source document before chunking."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = ""
    title: str = ""
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_attack: bool = False
    attack_type: AttackType | None = None


class Chunk(BaseModel):
    """A chunk of text extracted from a document."""
    chunk_id: str = Field(default_factory=lambda: f"c_{uuid.uuid4().hex[:6]}")
    document_id: str
    source: str = ""
    text: str
    embedding_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_attack: bool = False
    attack_type: AttackType | None = None
    trust_score: float = 1.0


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever with scoring."""
    chunk: Chunk
    rank: int
    similarity_score: float
    risk_score: float | None = None
    influence_score: float | None = None
    max_pairwise_interaction: float | None = None
    harmful_probability: float | None = None


# ---------------------------------------------------------------------------
# RAG Pipeline Models
# ---------------------------------------------------------------------------

class RAGAnswer(BaseModel):
    """The output of a single RAG query."""
    query_id: str
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    model_name: str = ""
    prompt: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0
    token_count: int = 0


# ---------------------------------------------------------------------------
# Attack Models
# ---------------------------------------------------------------------------

class AttackSpec(BaseModel):
    """Specification for generating an attack."""
    attack_type: AttackType
    target_query: str
    target_query_id: str = ""
    gold_answer: str = ""
    target_wrong_answer: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class AttackResult(BaseModel):
    """Result of running an attack."""
    attack_type: AttackType
    target_query_id: str
    malicious_chunks: list[Chunk] = Field(default_factory=list)
    original_answer: str = ""
    attacked_answer: str = ""
    attack_success: bool = False
    success_reason: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detection Models
# ---------------------------------------------------------------------------

class RiskFeatures(BaseModel):
    """Individual risk features for a chunk."""
    instruction_score: float = 0.0
    sensitive_score: float = 0.0
    retrieval_anomaly_score: float = 0.0
    conflict_score: float = 0.0
    coordination_score: float = 0.0
    stance_shift_score: float = 0.0
    source_trust_score: float = 1.0


class RiskScore(BaseModel):
    """Aggregated risk assessment for a chunk."""
    chunk_id: str
    features: RiskFeatures
    risk_score: float = 0.0
    is_suspicious: bool = False
    is_high_risk: bool = False
    risk_reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Influence Models
# ---------------------------------------------------------------------------

class InfluenceComponents(BaseModel):
    """Individual influence measurement components."""
    semantic_answer_change: float = 0.0
    unsafe_score_drop: float = 0.0
    leakage_drop: float = 0.0
    citation_dependency: float = 0.0


class InfluenceScore(BaseModel):
    """Causal influence score for a chunk."""
    chunk_id: str
    influence_score: float = 0.0
    components: InfluenceComponents = Field(default_factory=InfluenceComponents)
    counterfactual_answer: str = ""
    is_harmful: bool = False
    is_critical: bool = False


class PairwiseInteraction(BaseModel):
    """Local second-order Shapley-truncated interaction between two chunks."""
    chunk_id_1: str
    chunk_id_2: str
    interaction: float
    coordinated: bool = False


# ---------------------------------------------------------------------------
# Repair Models
# ---------------------------------------------------------------------------

class RepairAction(BaseModel):
    """A single repair action taken by the engine."""
    action_type: RepairActionType
    target_chunk_id: str | None = None
    reason: str = ""
    before_score: float | None = None
    after_score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RepairResult(BaseModel):
    """Complete result of the repair process."""
    query_id: str
    original_answer: str
    repaired_answer: str
    actions: list[RepairAction] = Field(default_factory=list)
    removed_chunk_ids: list[str] = Field(default_factory=list)
    downranked_chunk_ids: list[str] = Field(default_factory=list)
    replacement_chunk_ids: list[str] = Field(default_factory=list)
    original_risk_scores: dict[str, float] = Field(default_factory=dict)
    repaired_risk_scores: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Verification Models
# ---------------------------------------------------------------------------

class VerificationTests(BaseModel):
    """Results of individual verification tests."""
    no_sensitive_leakage: bool = False
    low_attacker_dependency: bool = False
    answer_stability: bool = False
    minimum_trusted_support: bool = False
    source_diversity: bool = False
    instruction_not_followed: bool = False
    certified_smoothing_post: bool = False
    certified_smoothing_raw: bool = False


class VerificationScores(BaseModel):
    """Scores for each verification dimension."""
    grounding_score: float = 0.0
    stability_score: float = 0.0
    no_leakage_score: float = 0.0
    low_attacker_dependency_score: float = 0.0
    source_diversity_score: float = 0.0
    instruction_isolation_score: float = 0.0


class HighInfluenceChunkInfo(BaseModel):
    """Info about a high-influence chunk for the certificate."""
    chunk_id: str
    risk_score: float
    influence_score: float
    reason: str = ""


class VerificationCertificate(BaseModel):
    """Proof-carrying verification certificate."""
    certificate_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query_id: str
    status: CertificateStatus = CertificateStatus.FAIL
    original_answer: str = ""
    repaired_answer: str = ""
    removed_chunks: list[str] = Field(default_factory=list)
    downranked_chunks: list[str] = Field(default_factory=list)
    high_influence_chunks: list[HighInfluenceChunkInfo] = Field(default_factory=list)
    repair_actions: list[RepairAction] = Field(default_factory=list)
    verification_tests: VerificationTests = Field(default_factory=VerificationTests)
    verification_scores: VerificationScores = Field(default_factory=VerificationScores)
    certificate_score: float = 0.0
    passed: bool = False
    certified_bound_raw: float | None = None
    certified_bound_post: float | None = None
    certified_alpha: float = 0.05
    coordinated_pairs: list[list[str]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Experiment Models
# ---------------------------------------------------------------------------

class ExperimentConditionResult(BaseModel):
    """Result for a single experimental condition."""
    condition_name: str
    defense: str = "none"
    attacks: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    num_queries: int = 0
    certificates: list[VerificationCertificate] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """Complete result of an experiment."""
    experiment_name: str
    description: str = ""
    conditions: list[ExperimentConditionResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config_path: str = ""
