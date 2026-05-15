"""Pairwise second-order interaction (Shapley-truncated local term)."""

from __future__ import annotations

from itertools import combinations

from veri_rag.config.schema import PairwiseInteraction, RAGAnswer, RetrievedChunk
from veri_rag.influence.leave_one_out import LeaveOneOutAnalyzer


class PairwiseInteractionAnalyzer:
    """Compute local second-order interaction indices for chunk pairs."""

    def __init__(self, loo_analyzer: LeaveOneOutAnalyzer):
        self.loo = loo_analyzer

    def analyze_pairs(
        self,
        query: str,
        query_id: str,
        retrieved: list[RetrievedChunk],
        baseline: RAGAnswer,
        loo_changes: dict[str, float],
        candidate_ids: list[str],
        max_pairs: int = 10,
        prompt_template: str | None = None,
        interaction_threshold: float = 0.15,
    ) -> dict[tuple[str, str], PairwiseInteraction]:
        """I(c1,c2) = change(without both) - loo(c1) - loo(c2) at empty coalition."""
        pairs = list(combinations(candidate_ids[:5], 2))[:max_pairs]
        interactions: dict[tuple[str, str], PairwiseInteraction] = {}

        for c1, c2 in pairs:
            without = [
                r for r in retrieved if r.chunk.chunk_id not in (c1, c2)
            ]
            counter = self.loo.rag.ask(
                query,
                query_id=f"{query_id}_pair_{c1}_{c2}",
                chunks_override=without,
                prompt_template=prompt_template,
            )
            combined = self.loo._measure_change(baseline, counter, c1)
            pair_change = combined.semantic_answer_change
            v_pair = pair_change
            v_c1 = loo_changes.get(c1, 0.0)
            v_c2 = loo_changes.get(c2, 0.0)
            interaction_val = v_pair - v_c1 - v_c2

            interactions[(c1, c2)] = PairwiseInteraction(
                chunk_id_1=c1,
                chunk_id_2=c2,
                interaction=interaction_val,
                coordinated=interaction_val >= interaction_threshold,
            )
        return interactions
