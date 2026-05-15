"""RIAA: Risk-filtered Interaction-Aware Attribution."""

from __future__ import annotations

from dataclasses import dataclass, field

from veri_rag.config.schema import (
    InfluenceScore,
    PairwiseInteraction,
    RAGAnswer,
    RetrievedChunk,
    RiskScore,
)
from veri_rag.config.settings import InfluenceConfig
from veri_rag.detection.risk_scorer import RiskScorer
from veri_rag.influence.calibrator import HarmfulnessCalibrator
from veri_rag.influence.leave_one_out import LeaveOneOutAnalyzer
from veri_rag.influence.pairwise import PairwiseInteractionAnalyzer
from veri_rag.rag.baseline_rag import BaselineRAG


@dataclass
class RIAAResult:
    """Full RIAA attribution output."""

    baseline: RAGAnswer
    loo_scores: dict[str, InfluenceScore] = field(default_factory=dict)
    pairwise: dict[tuple[str, str], PairwiseInteraction] = field(default_factory=dict)
    harmful_probability: dict[str, float] = field(default_factory=dict)
    coordinated_pairs: list[tuple[str, str]] = field(default_factory=list)


class RIAAAnalyzer:
    """Risk-filtered LOO + pairwise Shapley-truncated interactions."""

    def __init__(
        self,
        rag: BaselineRAG,
        config: InfluenceConfig,
        risk_scorer: RiskScorer,
        calibrator: HarmfulnessCalibrator | None = None,
    ):
        self.config = config
        self.risk_scorer = risk_scorer
        self.loo = LeaveOneOutAnalyzer(rag, config, risk_scorer)
        self.pairwise = PairwiseInteractionAnalyzer(self.loo)
        self.calibrator = calibrator

    def analyze(
        self,
        query: str,
        query_id: str,
        retrieved: list[RetrievedChunk],
        risk_scores: dict[str, RiskScore],
        baseline: RAGAnswer | None = None,
        prompt_template: str | None = None,
        enable_pairwise: bool = True,
    ) -> RIAAResult:
        baseline, loo_scores = self.loo.analyze(
            query, query_id, retrieved, baseline, prompt_template
        )

        loo_changes = {
            cid: inf.components.semantic_answer_change for cid, inf in loo_scores.items()
        }
        suspicious_ids = list(loo_scores.keys())

        pairwise: dict[tuple[str, str], PairwiseInteraction] = {}
        if enable_pairwise and self.config.enable_pairwise and len(suspicious_ids) >= 2:
            pairwise = self.pairwise.analyze_pairs(
                query,
                query_id,
                retrieved,
                baseline,
                loo_changes,
                suspicious_ids,
                max_pairs=self.config.max_pairwise_tests,
                prompt_template=prompt_template,
                interaction_threshold=self.config.pairwise_threshold,
            )

        max_pairwise_by_chunk: dict[str, float] = {cid: 0.0 for cid in suspicious_ids}
        coordinated: list[tuple[str, str]] = []
        for (c1, c2), pi in pairwise.items():
            max_pairwise_by_chunk[c1] = max(max_pairwise_by_chunk[c1], abs(pi.interaction))
            max_pairwise_by_chunk[c2] = max(max_pairwise_by_chunk[c2], abs(pi.interaction))
            if pi.coordinated:
                coordinated.append((c1, c2))
                for cid in (c1, c2):
                    if cid in loo_scores:
                        loo_scores[cid].is_harmful = True

        for rc in retrieved:
            cid = rc.chunk.chunk_id
            if cid in max_pairwise_by_chunk:
                rc.max_pairwise_interaction = max_pairwise_by_chunk[cid]

        harmful_p: dict[str, float] = {}
        for cid, inf in loo_scores.items():
            risk = risk_scores.get(cid)
            mp = max_pairwise_by_chunk.get(cid, 0.0)
            if self.calibrator and self.calibrator._fitted:
                features = self.calibrator.feature_vector(cid, risk, inf, mp)
                harmful_p[cid] = self.calibrator.predict_proba(features)
            else:
                harmful_p[cid] = 1.0 if inf.is_harmful else 0.0
            for rc in retrieved:
                if rc.chunk.chunk_id == cid:
                    rc.harmful_probability = harmful_p[cid]

        return RIAAResult(
            baseline=baseline,
            loo_scores=loo_scores,
            pairwise=pairwise,
            harmful_probability=harmful_p,
            coordinated_pairs=coordinated,
        )
