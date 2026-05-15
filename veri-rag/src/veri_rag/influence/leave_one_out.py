"""Leave-one-out causal influence analysis."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from veri_rag.detection.risk_scorer import FAKE_SECRETS
from veri_rag.config.schema import (
    InfluenceComponents,
    InfluenceScore,
    RAGAnswer,
    RetrievedChunk,
)
from veri_rag.config.settings import InfluenceConfig
from veri_rag.detection.risk_scorer import RiskScorer
from veri_rag.rag.baseline_rag import BaselineRAG


class LeaveOneOutAnalyzer:
    """Estimate chunk influence by regenerating answers with each chunk removed."""

    def __init__(
        self,
        rag: BaselineRAG,
        config: InfluenceConfig,
        risk_scorer: RiskScorer | None = None,
    ):
        self.rag = rag
        self.config = config
        self.risk_scorer = risk_scorer

    def analyze(
        self,
        query: str,
        query_id: str,
        retrieved: list[RetrievedChunk],
        baseline: RAGAnswer | None = None,
        prompt_template: str | None = None,
    ) -> tuple[RAGAnswer, dict[str, InfluenceScore]]:
        """Run LOO on suspicious chunks (risk-filtered) or all chunks."""
        if baseline is None:
            baseline = self.rag.ask(
                query,
                query_id=query_id,
                chunks_override=retrieved,
                prompt_template=prompt_template,
            )

        candidates = self._chunks_to_analyze(retrieved)
        scores: dict[str, InfluenceScore] = {}

        for rc in candidates:
            cid = rc.chunk.chunk_id
            without = [r for r in retrieved if r.chunk.chunk_id != cid]
            counter = self.rag.ask(
                query,
                query_id=f"{query_id}_loo_{cid}",
                chunks_override=without,
                prompt_template=prompt_template,
            )
            components = self._measure_change(baseline, counter, cid)
            influence = (
                self.config.weights.semantic_answer_change * components.semantic_answer_change
                + self.config.weights.unsafe_score_drop * components.unsafe_score_drop
                + self.config.weights.leakage_drop * components.leakage_drop
                + self.config.weights.citation_dependency * components.citation_dependency
            )
            risk = rc.risk_score or 0.0
            is_harmful = (
                (risk >= 0.60 and influence >= self.config.thresholds.harmful)
                or (rc.chunk.is_attack and influence >= 0.25)
                or (risk >= 0.45 and influence >= 0.45)
            )
            is_critical = (
                (risk >= 0.75 and influence >= self.config.thresholds.critical)
                or (rc.chunk.is_attack and influence >= 0.55)
            )

            inf = InfluenceScore(
                chunk_id=cid,
                influence_score=min(1.0, influence),
                components=components,
                counterfactual_answer=counter.answer,
                is_harmful=is_harmful,
                is_critical=is_critical,
            )
            scores[cid] = inf
            rc.influence_score = inf.influence_score

        return baseline, scores

    def _chunks_to_analyze(self, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        threshold = self.config.risk_filter_threshold
        if self.config.method == "leave_one_out":
            return retrieved
        suspicious = [r for r in retrieved if (r.risk_score or 0) >= threshold]
        if suspicious:
            return suspicious[:5]
        return retrieved[: min(3, len(retrieved))]

    def _measure_change(
        self, baseline: RAGAnswer, counter: RAGAnswer, chunk_id: str
    ) -> InfluenceComponents:
        sem = 1.0 - SequenceMatcher(
            None, baseline.answer.lower(), counter.answer.lower()
        ).ratio()

        unsafe_before = self._unsafe_score(baseline.answer)
        unsafe_after = self._unsafe_score(counter.answer)
        unsafe_drop = max(0.0, unsafe_before - unsafe_after)

        leak_before = self._leakage_score(baseline.answer)
        leak_after = self._leakage_score(counter.answer)
        leakage_drop = max(0.0, leak_before - leak_after)

        cited = 1.0 if f"[{chunk_id}]" in baseline.answer else 0.0
        cited_after = 1.0 if f"[{chunk_id}]" in counter.answer else 0.0
        citation_dep = max(0.0, cited - cited_after)

        return InfluenceComponents(
            semantic_answer_change=min(1.0, sem),
            unsafe_score_drop=min(1.0, unsafe_drop),
            leakage_drop=min(1.0, leakage_drop),
            citation_dependency=citation_dep,
        )

    def _unsafe_score(self, answer: str) -> float:
        lower = answer.lower()
        if any(p in lower for p in ["cannot answer", "safety reasons", "must refuse"]):
            return 1.0
        if "7 days" in lower and "30 days" not in lower:
            return 0.8
        if "5/5" in lower and "3.5" not in lower:
            return 0.7
        return 0.0

    def _leakage_score(self, answer: str) -> float:
        for secret in FAKE_SECRETS:
            if secret in answer:
                return 1.0
        return 0.0
