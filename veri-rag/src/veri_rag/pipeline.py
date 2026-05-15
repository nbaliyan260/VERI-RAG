"""VERI-RAG end-to-end pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from veri_rag.attacks.runner import AttackRunner
from veri_rag.baselines.runner import BaselineRunner
from veri_rag.config.schema import AttackType, RAGAnswer, RetrievedChunk
from veri_rag.config.settings import Settings, ensure_output_dirs, get_project_root
from veri_rag.corpus.ingest import IndexArtifacts, ingest_corpus, load_index
from veri_rag.detection.risk_scorer import RiskScorer
from veri_rag.eval.metrics import MetricsCalculator
from veri_rag.influence.calibrator import HarmfulnessCalibrator
from veri_rag.influence.riaa import RIAAAnalyzer
from veri_rag.provenance.graph_builder import ProvenanceGraphBuilder
from veri_rag.rag.baseline_rag import BaselineRAG
from veri_rag.rag.prompts import REPAIR_PROMPT, SAFE_PROMPT
from veri_rag.rag.retriever import Retriever
from veri_rag.repair.repair_engine import RepairEngine
from veri_rag.verify.smoothing import CertifiedSmoothing
from veri_rag.verify.verifier import Verifier

DefenseMode = Literal[
    "none",
    "safe_prompt",
    "risk_quarantine",
    "grada",
    "robust_rag",
    "veri_rag",
]


class VERIRAGPipeline:
    """Full VERI-RAG system bound to config and index artifacts."""

    def __init__(self, settings: Settings, index: IndexArtifacts | None = None):
        self.settings = settings
        ensure_output_dirs(settings)
        self.index = index or load_index(settings)
        self.retriever = Retriever(
            vector_store=self.index.vector_store,
            embedder=self.index.embedder,
            top_k=settings.retrieval.top_k,
        )
        from veri_rag.rag.generator import create_llm

        self.rag = BaselineRAG(
            retriever=self.retriever,
            llm=create_llm(
                provider=settings.llm.provider,
                model_name=settings.llm.model_name,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            ),
            top_k=settings.retrieval.top_k,
        )
        self.risk_scorer = RiskScorer(settings.risk_scoring)
        root = get_project_root()
        cal_path = root / settings.influence.calibrator_path
        calibrator = HarmfulnessCalibrator(cal_path if cal_path.exists() else None)
        self.riaa = RIAAAnalyzer(self.rag, settings.influence, self.risk_scorer, calibrator)
        self.repair_engine = RepairEngine(self.rag, settings.repair)
        self.verifier = Verifier(settings.verification)
        self.smoothing = CertifiedSmoothing(self.rag, settings.verification.certified_smoothing)
        self.baseline_runner = BaselineRunner(
            self.rag, self.risk_scorer, self.repair_engine
        )
        self.attack_runner = AttackRunner()
        self.metrics = MetricsCalculator()
        self.provenance = ProvenanceGraphBuilder()

    @classmethod
    def from_config(cls, config_path: str | Path, build_index: bool = False) -> "VERIRAGPipeline":
        from veri_rag.config.settings import load_settings

        settings = load_settings(config_path)
        index = ingest_corpus(settings) if build_index else load_index(settings)
        return cls(settings, index)

    def retrieve_clean(self, query: str) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, top_k=self.settings.retrieval.top_k)

    def run_with_attack(
        self,
        query_id: str,
        query: str,
        attack_type: AttackType,
        gold_answer: str = "",
        defense: DefenseMode = "veri_rag",
        enable_certified: bool | None = None,
        enable_pairwise: bool = True,
        seed: int = 42,
    ) -> dict:
        """Run pipeline for one query under attack and optional defense."""
        clean = self.retrieve_clean(query)
        spec = self.attack_runner.build_spec(attack_type, query_id, query, gold_answer)
        malicious = self.attack_runner.generate_chunks(spec)
        retrieved = self.attack_runner.inject_into_retrieval(
            clean, malicious, self.settings.retrieval.top_k
        )
        attack_ids = {c.chunk_id for c in malicious}

        if defense == "grada":
            self.risk_scorer.score_all(query, retrieved)
            retrieved = self.baseline_runner.grada.rerank(
                query, retrieved, self.settings.retrieval.top_k
            )

        prompt = SAFE_PROMPT if defense == "safe_prompt" else None

        if defense == "robust_rag":
            final = self.baseline_runner.run(
                "robust_rag", query, query_id, retrieved, self.settings.retrieval.top_k
            )
            return self._result_row(
                query_id, attack_type, defense, "", final, gold_answer, spec, attack_ids
            )

        baseline = self.rag.ask(
            query, query_id=query_id, chunks_override=retrieved, prompt_template=prompt
        )

        result: dict = {
            "query_id": query_id,
            "attack": attack_type.value,
            "defense": defense,
            "seed": seed,
            "baseline_answer": baseline.answer,
            "attack_success_baseline": self.metrics.attack_success(
                attack_type, baseline.answer, gold_answer, spec.target_wrong_answer
            ),
        }

        if defense in ("none", "safe_prompt"):
            result["final_answer"] = baseline.answer
            result["attack_success"] = result["attack_success_baseline"]
            return result

        risk_scores = self.risk_scorer.score_all(query, retrieved)

        if defense == "risk_quarantine":
            repair = self.repair_engine.risk_only_quarantine(
                query, query_id, retrieved, baseline, risk_scores
            )
            result["final_answer"] = repair.repaired_answer
            result["attack_success"] = self.metrics.attack_success(
                attack_type, repair.repaired_answer, gold_answer, spec.target_wrong_answer
            )
            result["repair_success"] = self.metrics.repair_success(
                attack_type, repair.repaired_answer, gold_answer
            )
            return result

        # Full VERI-RAG with RIAA
        riaa_result = self.riaa.analyze(
            query,
            query_id,
            retrieved,
            risk_scores,
            baseline=baseline,
            prompt_template=prompt,
            enable_pairwise=enable_pairwise,
        )
        extra = [r for r in clean if r.chunk.trust_score >= 0.9]
        repair = self.repair_engine.repair(
            query,
            query_id,
            retrieved,
            riaa_result.baseline,
            risk_scores,
            riaa_result.loo_scores,
            extra_pool=extra,
            coordinated_pairs=riaa_result.coordinated_pairs,
        )
        repaired_chunks = [
            r
            for r in retrieved
            if r.chunk.chunk_id not in repair.removed_chunk_ids
            and r.chunk.chunk_id not in repair.downranked_chunk_ids
        ]
        for rid in repair.replacement_chunk_ids:
            match = next((r for r in extra if r.chunk.chunk_id == rid), None)
            if match and match not in repaired_chunks:
                repaired_chunks.append(match)

        do_cert = enable_certified
        if do_cert is None:
            do_cert = self.settings.verification.certified_smoothing.enabled

        smooth_raw = smooth_post = None
        if do_cert:
            smooth_raw = self.smoothing.certify(
                query, f"{query_id}_raw", retrieved, prompt, seed=seed
            )
            smooth_post = self.smoothing.certify(
                query,
                f"{query_id}_post",
                repaired_chunks,
                REPAIR_PROMPT,
                seed=seed + 1,
            )

        cert = self.verifier.verify(
            query_id,
            repair,
            risk_scores,
            riaa_result.loo_scores,
            repaired_chunks,
            gold_answer,
            smoothing_raw=smooth_raw,
            smoothing_post=smooth_post,
            coordinated_pairs=riaa_result.coordinated_pairs,
        )

        graph = self.provenance.build(
            query_id, query, retrieved, risk_scores, repair, cert
        )
        root = get_project_root()
        prov_dir = root / self.settings.outputs.provenance_dir
        self.provenance.export(graph, prov_dir, query_id)

        cert_path = root / self.settings.outputs.certificates_dir / f"{query_id}.json"
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        cert_path.write_text(cert.model_dump_json(indent=2), encoding="utf-8")

        ranked = self.metrics.rank_combined(risk_scores, riaa_result.loo_scores)
        result.update(
            {
                "final_answer": repair.repaired_answer,
                "attack_success": self.metrics.attack_success(
                    attack_type, repair.repaired_answer, gold_answer, spec.target_wrong_answer
                ),
                "repair_success": self.metrics.repair_success(
                    attack_type, repair.repaired_answer, gold_answer
                ),
                "certificate_score": cert.certificate_score,
                "certificate_passed": cert.passed,
                "certified_bound_raw": cert.certified_bound_raw,
                "certified_bound_post": cert.certified_bound_post,
                "coordinated_pairs": cert.coordinated_pairs,
                "removed_chunks": repair.removed_chunk_ids,
                "precision_at_1": self.metrics.precision_at_k(ranked, attack_ids, 1),
                "precision_at_3": self.metrics.precision_at_k(ranked, attack_ids, 3),
                "recall_at_3": self.metrics.recall_at_k(ranked, attack_ids, 3),
                "mrr": self.metrics.mrr(ranked, attack_ids),
            }
        )
        return result

    def _result_row(
        self,
        query_id: str,
        attack_type: AttackType,
        defense: str,
        baseline_answer: str,
        final_answer: str,
        gold_answer: str,
        spec,
        attack_ids: set[str],
    ) -> dict:
        return {
            "query_id": query_id,
            "attack": attack_type.value,
            "defense": defense,
            "baseline_answer": baseline_answer,
            "final_answer": final_answer,
            "attack_success": self.metrics.attack_success(
                attack_type, final_answer, gold_answer, spec.target_wrong_answer
            ),
        }

    def train_calibrator(self, samples_path: str | Path | None = None) -> Path:
        """Train harmfulness calibrator from labeled retrieval samples."""
        from veri_rag.influence.calibrator import HarmfulnessCalibrator

        root = get_project_root()
        out = root / self.settings.influence.calibrator_path
        samples: list[dict] = []

        queries = []
        qpath = root / "data" / "synthetic_enterprise" / "queries" / "enterprise_qa.jsonl"
        if qpath.exists():
            from veri_rag.corpus.synthetic import load_query_set

            queries = load_query_set(qpath)

        for q in queries[:5]:
            for attack in [AttackType.POISONING, AttackType.PROMPT_INJECTION]:
                clean = self.retrieve_clean(q["query"])
                spec = self.attack_runner.build_spec(
                    attack, q["query_id"], q["query"], q.get("gold_answer", "")
                )
                malicious = self.attack_runner.generate_chunks(spec)
                retrieved = self.attack_runner.inject_into_retrieval(
                    clean, malicious, self.settings.retrieval.top_k
                )
                risk = self.risk_scorer.score_all(q["query"], retrieved)
                base = self.rag.ask(q["query"], query_id=q["query_id"], chunks_override=retrieved)
                _, loo = self.riaa.loo.analyze(q["query"], q["query_id"], retrieved, base)
                for rc in retrieved:
                    cid = rc.chunk.chunk_id
                    inf = loo.get(cid)
                    rs = risk.get(cid)
                    mp = 0.0
                    feat = self.riaa.calibrator.feature_vector(cid, rs, inf, mp) if self.riaa.calibrator else []
                    if self.riaa.calibrator:
                        samples.append(
                            {
                                "features": feat,
                                "is_attack": rc.chunk.is_attack,
                            }
                        )

        if len(samples) < 4:
            return out
        HarmfulnessCalibrator.train_from_labels(samples, out)
        self.riaa.calibrator = HarmfulnessCalibrator(out)
        return out

    def save_answer(self, query_id: str, answer: RAGAnswer) -> Path:
        root = get_project_root()
        path = root / self.settings.outputs.answers_dir / f"{query_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(answer.model_dump_json(indent=2), encoding="utf-8")
        return path
