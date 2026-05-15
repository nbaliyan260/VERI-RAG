"""Certified randomized smoothing for RAG (Clopper–Pearson bound)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from difflib import SequenceMatcher

from veri_rag.config.schema import RAGAnswer, RetrievedChunk
from veri_rag.config.settings import CertifiedSmoothingConfig
from veri_rag.rag.baseline_rag import BaselineRAG


@dataclass
class SmoothingResult:
    """Certified smoothing outcome."""

    dominant_answer: str
    empirical_frequency: float
    lower_confidence_bound: float
    certified_instability: float
    passed: bool
    cluster_counts: dict[str, int]


class CertifiedSmoothing:
    """Randomized subset smoothing with binomial lower confidence bound."""

    def __init__(self, rag: BaselineRAG, config: CertifiedSmoothingConfig):
        self.rag = rag
        self.config = config

    def certify(
        self,
        query: str,
        query_id: str,
        chunks: list[RetrievedChunk],
        prompt_template: str | None = None,
        seed: int = 42,
    ) -> SmoothingResult:
        if not chunks:
            return SmoothingResult("", 0.0, 0.0, 1.0, False, {})

        rng = random.Random(seed)
        subset_size = max(1, int(len(chunks) * self.config.subset_fraction))
        answers: list[str] = []

        for i in range(self.config.num_samples):
            sample = rng.sample(chunks, min(subset_size, len(chunks)))
            ans = self.rag.ask(
                query,
                query_id=f"{query_id}_smooth_{i}",
                chunks_override=sample,
                prompt_template=prompt_template,
            )
            answers.append(ans.answer)

        clusters = self._cluster_answers(answers)
        dominant = max(clusters, key=clusters.get)
        p_hat = clusters[dominant] / len(answers)
        n = len(answers)
        successes = clusters[dominant]
        p_lower = self._clopper_pearson_lower(successes, n, self.config.confidence_alpha)
        instability = 1.0 - p_lower
        passed = p_lower >= self.config.stability_threshold

        return SmoothingResult(
            dominant_answer=dominant,
            empirical_frequency=p_hat,
            lower_confidence_bound=p_lower,
            certified_instability=instability,
            passed=passed,
            cluster_counts=clusters,
        )

    def _cluster_answers(self, answers: list[str], threshold: float = 0.85) -> dict[str, int]:
        clusters: dict[str, int] = {}
        for ans in answers:
            matched = False
            for key in list(clusters.keys()):
                if SequenceMatcher(None, ans.lower(), key.lower()).ratio() >= threshold:
                    clusters[key] += 1
                    matched = True
                    break
            if not matched:
                clusters[ans] = 1
        return clusters

    def _clopper_pearson_lower(self, successes: int, n: int, alpha: float) -> float:
        """One-sided lower bound for binomial proportion."""
        if n == 0:
            return 0.0
        try:
            from scipy.stats import beta

            return float(beta.ppf(alpha, successes, n - successes + 1))
        except ImportError:
            # Conservative Wilson-like fallback without scipy
            p = successes / n
            z = 1.64  # ~95% one-sided
            denom = 1 + z**2 / n
            centre = p + z**2 / (2 * n)
            margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
            return max(0.0, (centre - margin) / denom)
