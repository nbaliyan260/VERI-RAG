"""RobustRAG-style isolate-then-aggregate baseline."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from veri_rag.config.schema import RetrievedChunk
from veri_rag.rag.baseline_rag import BaselineRAG
from veri_rag.rag.prompts import SAFE_PROMPT


class RobustRAGBaseline:
    """Generate per-chunk answers and aggregate by semantic voting."""

    def __init__(self, rag: BaselineRAG):
        self.rag = rag

    def aggregate_answer(
        self,
        query: str,
        query_id: str,
        retrieved: list[RetrievedChunk],
    ) -> str:
        if not retrieved:
            return "The evidence is insufficient to answer this question."

        sub_answers: list[str] = []
        for rc in retrieved:
            ans = self.rag.ask(
                query,
                query_id=f"{query_id}_iso_{rc.chunk.chunk_id}",
                chunks_override=[rc],
                prompt_template=SAFE_PROMPT,
            )
            sub_answers.append(ans.answer)

        clusters: list[list[str]] = []
        for ans in sub_answers:
            placed = False
            for cluster in clusters:
                if SequenceMatcher(None, ans.lower(), cluster[0].lower()).ratio() >= 0.8:
                    cluster.append(ans)
                    placed = True
                    break
            if not placed:
                clusters.append([ans])

        largest = max(clusters, key=len)
        return Counter(largest).most_common(1)[0][0]
