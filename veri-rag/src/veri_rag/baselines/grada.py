"""GRADA-style graph reranking defense (lightweight approximation)."""

from __future__ import annotations

import networkx as nx

from veri_rag.config.schema import RetrievedChunk
from veri_rag.detection.risk_scorer import RiskScorer


class GRADAReranker:
    """Down-rank chunks that form suspicious similarity cliques."""

    def __init__(self, risk_scorer: RiskScorer):
        self.risk_scorer = risk_scorer

    def rerank(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if len(retrieved) <= 1:
            return retrieved

        g = nx.Graph()
        for rc in retrieved:
            g.add_node(rc.chunk.chunk_id, risk=rc.risk_score or 0.0)

        for i, a in enumerate(retrieved):
            for b in retrieved[i + 1 :]:
                sim = self._text_sim(a.chunk.text, b.chunk.text)
                if sim > 0.5 and "untrusted" in a.chunk.source + b.chunk.source:
                    g.add_edge(a.chunk.chunk_id, b.chunk.chunk_id, weight=sim)

        suspicious_clique = set()
        for component in nx.connected_components(g):
            if len(component) >= 2:
                risks = [g.nodes[n].get("risk", 0) for n in component]
                if max(risks) >= 0.4:
                    suspicious_clique |= component

        scored = []
        for rc in retrieved:
            penalty = 0.3 if rc.chunk.chunk_id in suspicious_clique else 0.0
            if rc.risk_score is None:
                rs = self.risk_scorer.score_chunk(query, rc, retrieved)
                rc.risk_score = rs.risk_score
            score = rc.similarity_score - penalty - (rc.risk_score or 0) * 0.2
            scored.append((score, rc))

        scored.sort(key=lambda x: -x[0])
        out: list[RetrievedChunk] = []
        for rank, (_, rc) in enumerate(scored[:top_k], 1):
            rc.rank = rank
            out.append(rc)
        return out

    def _text_sim(self, a: str, b: str) -> float:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a[:200], b[:200]).ratio()
