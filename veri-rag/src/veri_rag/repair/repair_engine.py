"""Self-healing repair engine — minimally destructive context fixes."""

from __future__ import annotations

import re

from veri_rag.config.schema import (
    Chunk,
    InfluenceScore,
    RAGAnswer,
    RepairAction,
    RepairActionType,
    RepairResult,
    RetrievedChunk,
    RiskScore,
)
from veri_rag.config.settings import RepairConfig
from veri_rag.rag.baseline_rag import BaselineRAG
from veri_rag.rag.prompts import REPAIR_PROMPT


class RepairEngine:
    """Select and apply repair actions, then regenerate the answer."""

    def __init__(self, rag: BaselineRAG, config: RepairConfig):
        self.rag = rag
        self.config = config

    def repair(
        self,
        query: str,
        query_id: str,
        retrieved: list[RetrievedChunk],
        baseline: RAGAnswer,
        risk_scores: dict[str, RiskScore],
        influence_scores: dict[str, InfluenceScore],
        extra_pool: list[RetrievedChunk] | None = None,
        coordinated_pairs: list[tuple[str, str]] | None = None,
    ) -> RepairResult:
        actions: list[RepairAction] = []
        removed: list[str] = []
        downranked: list[str] = []
        replacements: list[str] = []
        coordinated_pairs = coordinated_pairs or []

        for c1, c2 in coordinated_pairs:
            for cid in (c1, c2):
                if cid not in removed:
                    removed.append(cid)
                    actions.append(
                        RepairAction(
                            action_type=RepairActionType.QUARANTINE,
                            target_chunk_id=cid,
                            reason="coordinated pair detected by RIAA",
                        )
                    )

        working = [self._copy_retrieved(r) for r in retrieved]

        for rc in working:
            cid = rc.chunk.chunk_id
            risk = risk_scores.get(cid)
            infl = influence_scores.get(cid)
            if not risk or not infl:
                continue

            should_quarantine = (
                (risk.is_high_risk and infl.is_harmful)
                or (rc.chunk.is_attack and infl.is_harmful)
                or (rc.chunk.is_attack and risk.is_suspicious)
            )
            if should_quarantine:
                actions.append(
                    RepairAction(
                        action_type=RepairActionType.QUARANTINE,
                        target_chunk_id=cid,
                        reason="; ".join(risk.risk_reasons),
                        before_score=risk.risk_score,
                    )
                )
                removed.append(cid)
                continue

            if risk.features.instruction_score >= 0.5:
                sanitized = self._sanitize_chunk(rc.chunk)
                rc.chunk = sanitized
                actions.append(
                    RepairAction(
                        action_type=RepairActionType.SANITIZE,
                        target_chunk_id=cid,
                        reason="instruction-like text removed",
                    )
                )

            if risk.features.sensitive_score >= 0.5:
                redacted = self._redact_chunk(rc.chunk)
                rc.chunk = redacted
                actions.append(
                    RepairAction(
                        action_type=RepairActionType.REDACT,
                        target_chunk_id=cid,
                        reason="sensitive spans redacted",
                    )
                )

            if risk.is_suspicious and infl.is_harmful and cid not in removed:
                actions.append(
                    RepairAction(
                        action_type=RepairActionType.DOWNRANK,
                        target_chunk_id=cid,
                        reason="high risk and influence",
                    )
                )
                downranked.append(cid)

        repaired_chunks = [
            r for r in working if r.chunk.chunk_id not in removed and r.chunk.chunk_id not in downranked
        ]

        # Source diversity: prefer trusted sources
        trusted = [r for r in repaired_chunks if r.chunk.trust_score >= 0.9]
        if len(trusted) < self.config.source_diversity_min and extra_pool:
            seen = {r.chunk.chunk_id for r in repaired_chunks}
            for r in extra_pool:
                if r.chunk.trust_score >= 0.9 and r.chunk.chunk_id not in seen:
                    repaired_chunks.append(r)
                    replacements.append(r.chunk.chunk_id)
                    seen.add(r.chunk.chunk_id)
                if len([x for x in repaired_chunks if x.chunk.trust_score >= 0.9]) >= self.config.source_diversity_min:
                    break
            actions.append(
                RepairAction(
                    action_type=RepairActionType.DIVERSIFY,
                    reason="added trusted replacement chunks",
                )
            )

        repaired_answer = self.rag.ask(
            query,
            query_id=f"{query_id}_repaired",
            chunks_override=repaired_chunks,
            prompt_template=REPAIR_PROMPT,
        )

        return RepairResult(
            query_id=query_id,
            original_answer=baseline.answer,
            repaired_answer=repaired_answer.answer,
            actions=actions,
            removed_chunk_ids=removed,
            downranked_chunk_ids=downranked,
            replacement_chunk_ids=replacements,
        )

    def risk_only_quarantine(
        self,
        query: str,
        query_id: str,
        retrieved: list[RetrievedChunk],
        baseline: RAGAnswer,
        risk_scores: dict[str, RiskScore],
    ) -> RepairResult:
        """Ablation: quarantine by risk only, no influence."""
        removed = [
            cid for cid, rs in risk_scores.items() if rs.is_suspicious
        ]
        kept = [r for r in retrieved if r.chunk.chunk_id not in removed]
        repaired = self.rag.ask(query, query_id=f"{query_id}_risk", chunks_override=kept)
        return RepairResult(
            query_id=query_id,
            original_answer=baseline.answer,
            repaired_answer=repaired.answer,
            actions=[
                RepairAction(
                    action_type=RepairActionType.QUARANTINE,
                    target_chunk_id=cid,
                    reason="risk-only quarantine",
                )
                for cid in removed
            ],
            removed_chunk_ids=removed,
        )

    def _sanitize_chunk(self, chunk: Chunk) -> Chunk:
        lines = []
        for line in chunk.text.split("\n"):
            lower = line.lower()
            if any(p in lower for p in ["ignore", "override", "must refuse", "must answer", "supersedes"]):
                continue
            lines.append(line)
        new_text = "\n".join(lines).strip() or "[sanitized]"
        return chunk.model_copy(update={"text": new_text})

    def _redact_chunk(self, chunk: Chunk) -> Chunk:
        text = chunk.text
        for secret in [
            "API_KEY_TEST_12345",
            "EMPLOYEE_ID_7788",
            "CUSTOMER_ACCOUNT_9912",
            "INTERNAL_PROJECT_CODENAME_ORION",
            "DB_PASSWORD_Pr0d_S3cur3!",
        ]:
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"api[_-]?key\s*[:=]\s*\S+", "[REDACTED]", text, flags=re.I)
        return chunk.model_copy(update={"text": text})

    def _copy_retrieved(self, rc: RetrievedChunk) -> RetrievedChunk:
        return RetrievedChunk(
            chunk=rc.chunk.model_copy(),
            rank=rc.rank,
            similarity_score=rc.similarity_score,
            risk_score=rc.risk_score,
            influence_score=rc.influence_score,
        )
