"""Run published-style baselines on attacked retrieval."""

from __future__ import annotations

from typing import Literal

from veri_rag.baselines.grada import GRADAReranker
from veri_rag.baselines.robust_rag import RobustRAGBaseline
from veri_rag.config.schema import RetrievedChunk
from veri_rag.detection.risk_scorer import RiskScorer
from veri_rag.rag.baseline_rag import BaselineRAG
from veri_rag.rag.prompts import SAFE_PROMPT
from veri_rag.repair.repair_engine import RepairEngine

BaselineMethod = Literal[
    "no_defense",
    "safe_prompt",
    "perplexity_filter",
    "instruction_filter",
    "dedup",
    "grada",
    "robust_rag",
    "risk_only",
]


class BaselineRunner:
    """Apply a baseline defense and return the model answer."""

    def __init__(self, rag: BaselineRAG, risk_scorer: RiskScorer, repair: RepairEngine):
        self.rag = rag
        self.risk_scorer = risk_scorer
        self.repair = repair
        self.grada = GRADAReranker(risk_scorer)
        self.robust = RobustRAGBaseline(rag)

    def run(
        self,
        method: BaselineMethod,
        query: str,
        query_id: str,
        retrieved: list[RetrievedChunk],
        top_k: int = 5,
    ) -> str:
        chunks = list(retrieved)

        if method == "dedup":
            chunks = self._dedup(chunks)
        if method in ("instruction_filter", "perplexity_filter"):
            chunks = self._instruction_filter(chunks)
        if method == "grada":
            self.risk_scorer.score_all(query, chunks)
            chunks = self.grada.rerank(query, chunks, top_k)

        if method == "no_defense":
            return self.rag.ask(query, query_id, chunks_override=chunks[:top_k]).answer
        if method == "safe_prompt":
            return self.rag.ask(
                query, query_id, chunks_override=chunks[:top_k], prompt_template=SAFE_PROMPT
            ).answer
        if method == "robust_rag":
            return self.robust.aggregate_answer(query, query_id, chunks[:top_k])
        if method == "risk_only":
            risk = self.risk_scorer.score_all(query, chunks)
            baseline = self.rag.ask(query, query_id, chunks_override=chunks[:top_k])
            repair = self.repair.risk_only_quarantine(
                query, query_id, chunks[:top_k], baseline, risk
            )
            return repair.repaired_answer

        return self.rag.ask(query, query_id, chunks_override=chunks[:top_k]).answer

    def _dedup(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        out: list[RetrievedChunk] = []
        for rc in chunks:
            key = rc.chunk.text[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(rc)
        return out

    def _instruction_filter(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        blocked = ("ignore previous", "system override", "must refuse", "api_key")
        out: list[RetrievedChunk] = []
        for rc in chunks:
            lower = rc.chunk.text.lower()
            if any(b in lower for b in blocked):
                continue
            out.append(rc)
        return out if out else chunks
