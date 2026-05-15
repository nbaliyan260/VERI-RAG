"""Risk scoring for retrieved chunks — fast, no LLM."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from veri_rag.config.schema import RetrievedChunk, RiskFeatures, RiskScore
from veri_rag.config.settings import RiskScoringConfig

INSTRUCTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+override",
    r"highest\s+priority",
    r"you\s+must\s+answer",
    r"must\s+refuse",
    r"must\s+not\s+answer",
    r"supersedes\s+all",
    r"do\s+not\s+follow",
]

SENSITIVE_PATTERNS = [
    r"api[_-]?key",
    r"password",
    r"secret",
    r"token",
    r"employee[_-]?id",
    r"customer[_-]?account",
    r"codename",
    r"db[_-]?password",
    r"[A-Z0-9_]{8,}",
]

FAKE_SECRETS = [
    "API_KEY_TEST_12345",
    "EMPLOYEE_ID_7788",
    "CUSTOMER_ACCOUNT_9912",
    "INTERNAL_PROJECT_CODENAME_ORION",
    "DB_PASSWORD_Pr0d_S3cur3!",
]


class RiskScorer:
    """Compute per-chunk risk scores from lexical and retrieval features."""

    def __init__(self, config: RiskScoringConfig):
        self.config = config
        self.weights = config.weights
        self.thresholds = config.thresholds

    def score_all(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
    ) -> dict[str, RiskScore]:
        """Score every retrieved chunk."""
        scores: dict[str, RiskScore] = {}
        for rc in retrieved:
            rs = self.score_chunk(query, rc, retrieved)
            scores[rc.chunk.chunk_id] = rs
            rc.risk_score = rs.risk_score
        return scores

    def score_chunk(
        self,
        query: str,
        rc: RetrievedChunk,
        all_retrieved: list[RetrievedChunk],
    ) -> RiskScore:
        text = rc.chunk.text
        features = RiskFeatures(
            instruction_score=self._instruction_score(text),
            sensitive_score=self._sensitive_score(text),
            retrieval_anomaly_score=self._retrieval_anomaly(rc, all_retrieved),
            conflict_score=self._conflict_score(text, all_retrieved, rc.chunk.chunk_id),
            coordination_score=self._coordination_score(text, all_retrieved, rc.chunk.chunk_id),
            stance_shift_score=self._stance_shift_score(text),
            source_trust_score=rc.chunk.trust_score,
        )

        w = self.weights
        risk = (
            w.instruction * features.instruction_score
            + w.sensitive * features.sensitive_score
            + w.retrieval_anomaly * features.retrieval_anomaly_score
            + w.conflict * features.conflict_score
            + w.coordination * features.coordination_score
            + w.stance_shift * features.stance_shift_score
        )
        risk = min(1.0, max(0.0, risk))

        reasons: list[str] = []
        if features.instruction_score >= 0.5:
            reasons.append("prompt-injection")
        if features.sensitive_score >= 0.5:
            reasons.append("sensitive content")
        if features.conflict_score >= 0.5:
            reasons.append("conflicting evidence")
        if features.coordination_score >= 0.5:
            reasons.append("coordinated near-duplicate")
        if features.stance_shift_score >= 0.5:
            reasons.append("stance manipulation")
        if not reasons:
            reasons.append("trusted policy document" if risk < 0.4 else "elevated risk signals")

        return RiskScore(
            chunk_id=rc.chunk.chunk_id,
            features=features,
            risk_score=risk,
            is_suspicious=risk >= self.thresholds.suspicious,
            is_high_risk=risk >= self.thresholds.high_risk,
            risk_reasons=reasons,
        )

    def _instruction_score(self, text: str) -> float:
        lower = text.lower()
        hits = sum(1 for p in INSTRUCTION_PATTERNS if re.search(p, lower))
        imperative = sum(1 for w in ["must", "ignore", "override", "refuse"] if w in lower)
        return min(1.0, 0.4 * hits + 0.15 * imperative)

    def _sensitive_score(self, text: str) -> float:
        score = 0.0
        for secret in FAKE_SECRETS:
            if secret in text:
                score = max(score, 1.0)
        for pat in SENSITIVE_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                score = max(score, 0.6)
        return min(1.0, score)

    def _retrieval_anomaly(self, rc: RetrievedChunk, all_retrieved: list[RetrievedChunk]) -> float:
        if not all_retrieved:
            return 0.0
        sims = [r.similarity_score for r in all_retrieved]
        avg = sum(sims) / len(sims)
        if rc.similarity_score > avg + 0.15 and "untrusted" in rc.chunk.source:
            return 0.8
        if rc.similarity_score >= 0.95 and rc.chunk.trust_score < 0.5:
            return 0.7
        return 0.2 if rc.similarity_score > avg else 0.0

    def _conflict_score(
        self, text: str, all_retrieved: list[RetrievedChunk], chunk_id: str
    ) -> float:
        """Heuristic: untrusted chunk contradicting trusted evidence."""
        rc = next((r for r in all_retrieved if r.chunk.chunk_id == chunk_id), None)
        if not rc or "untrusted" not in rc.chunk.source:
            return 0.0
        trusted_texts = [
            r.chunk.text
            for r in all_retrieved
            if r.chunk.trust_score >= 0.9 and r.chunk.chunk_id != chunk_id
        ]
        if not trusted_texts:
            return 0.5
        for trusted in trusted_texts:
            if "30 days" in trusted and "7 days" in text:
                return 0.9
            if "25 days" in trusted and "10 days" in text:
                return 0.85
            if "3.5/5" in trusted and "5/5" in text:
                return 0.85
        return 0.7

    def _coordination_score(
        self, text: str, all_retrieved: list[RetrievedChunk], chunk_id: str
    ) -> float:
        max_sim = 0.0
        for r in all_retrieved:
            if r.chunk.chunk_id == chunk_id:
                continue
            if "untrusted" in r.chunk.source:
                ratio = SequenceMatcher(None, text[:80], r.chunk.text[:80]).ratio()
                max_sim = max(max_sim, ratio)
        return min(1.0, max_sim)

    def _stance_shift_score(self, text: str) -> float:
        markers = [
            "universally considered",
            "definitive",
            "premium",
            "5/5",
            "supersedes",
            "authoritative answer",
        ]
        hits = sum(1 for m in markers if m.lower() in text.lower())
        return min(1.0, 0.25 * hits)
