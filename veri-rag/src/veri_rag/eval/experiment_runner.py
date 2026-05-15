"""Experiment runner for MVP evaluation suite with optional HPC sharding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from veri_rag.attacks.runner import load_attack_config_from_experiments
from veri_rag.config.schema import AttackType
from veri_rag.config.settings import get_project_root
from veri_rag.corpus.synthetic import load_query_set
from veri_rag.eval.metrics import MetricsCalculator
from veri_rag.eval.report_writer import write_csv, write_markdown_report
from veri_rag.eval.shard import (
    load_cached_result,
    result_cache_path,
    save_cached_result,
    shard_tasks,
)
from veri_rag.pipeline import VERIRAGPipeline


class ExperimentRunner:
    """Run configured experiments and write CSV/Markdown outputs."""

    def __init__(self, pipeline: VERIRAGPipeline, experiments_config: Path):
        self.pipeline = pipeline
        self.experiments_config = experiments_config
        self.metrics = MetricsCalculator()
        with open(experiments_config, encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}
        self.attack_runner_config = load_attack_config_from_experiments(experiments_config)
        from veri_rag.attacks.runner import AttackRunner

        self.pipeline.attack_runner = AttackRunner(self.attack_runner_config)

    def load_queries(self) -> list[dict[str, str]]:
        qs = self.config.get("query_sets", {}).get("enterprise_qa")
        if qs:
            return qs
        root = get_project_root()
        for rel in (
            "data/synthetic_enterprise/queries/enterprise_qa.jsonl",
            "data/benchmarks/poisonedrag/queries.jsonl",
        ):
            qpath = root / rel
            if qpath.exists():
                return load_query_set(qpath)
        return []

    def build_task_list(self) -> list[dict[str, Any]]:
        queries = self.load_queries()
        attacks = [
            AttackType.POISONING,
            AttackType.PROMPT_INJECTION,
            AttackType.SECRET_LEAKAGE,
            AttackType.BLOCKER,
            AttackType.TOPIC_FLIP,
            AttackType.ADAPTIVE,
        ]
        defenses = ["none", "safe_prompt", "risk_quarantine", "grada", "robust_rag", "veri_rag"]
        tasks: list[dict[str, Any]] = []
        for q in queries:
            for attack in attacks:
                if attack == AttackType.BLOCKER and q["query_id"] not in ["q001", "q002", "q004"]:
                    continue
                if attack == AttackType.TOPIC_FLIP and q["query_id"] != "q005":
                    continue
                for defense in defenses:
                    tasks.append(
                        {
                            "query_id": q["query_id"],
                            "query": q["query"],
                            "gold_answer": q.get("gold_answer", ""),
                            "attack": attack.value,
                            "defense": defense,
                            "seed": 0,
                        }
                    )
        return tasks

    def run_task(self, task: dict[str, Any]) -> dict[str, Any]:
        row = self.pipeline.run_with_attack(
            task["query_id"],
            task["query"],
            AttackType(task["attack"]),
            task.get("gold_answer", ""),
            defense=task["defense"],  # type: ignore[arg-type]
            seed=task.get("seed", 0),
        )
        row["defense"] = task["defense"]
        return row

    def run_repair_effectiveness(
        self,
        run_dir: Path | None = None,
        shard_id: int | None = None,
        num_shards: int = 1,
        resume: bool = True,
    ) -> tuple[list[dict], dict[str, dict[str, float]]]:
        root = get_project_root()
        run_dir = run_dir or (root / self.pipeline.settings.outputs.results_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        tasks = self.build_task_list()
        if shard_id is not None:
            tasks = shard_tasks(tasks, shard_id, num_shards)

        rows: list[dict[str, Any]] = []
        for task in tasks:
            cache = result_cache_path(
                run_dir,
                task["query_id"],
                task["attack"],
                task["defense"],
                task.get("seed", 0),
            )
            if resume and cache.exists():
                rows.append(load_cached_result(cache) or {})
                continue
            row = self.run_task(task)
            save_cached_result(cache, row)
            rows.append(row)

            jsonl_shard = run_dir / "shards" / f"shard_{shard_id or 0}.jsonl"
            jsonl_shard.parent.mkdir(parents=True, exist_ok=True)
            with open(jsonl_shard, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")

        summary: dict[str, dict[str, float]] = {}
        for defense in {t["defense"] for t in self.build_task_list()}:
            subset = [r for r in rows if r.get("defense") == defense]
            if not subset:
                continue
            summary[defense] = {
                "attack_success_rate": sum(1 for r in subset if r.get("attack_success")) / len(subset),
                "repair_success_rate": sum(1 for r in subset if r.get("repair_success")) / len(subset),
                "mean_certificate_score": sum(r.get("certificate_score", 0) for r in subset) / len(subset),
                "mean_certified_bound_post": sum(
                    r.get("certified_bound_post") or 0 for r in subset
                ) / len(subset),
                "precision_at_1": sum(r.get("precision_at_1", 0) for r in subset) / len(subset),
            }
        return rows, summary

    def run_all(
        self,
        shard_id: int | None = None,
        num_shards: int = 1,
        run_id: str = "default",
    ) -> Path:
        root = get_project_root()
        run_dir = root / self.pipeline.settings.outputs.hpc_runs_dir / run_id
        rows, summary = self.run_repair_effectiveness(
            run_dir=run_dir,
            shard_id=shard_id,
            num_shards=num_shards,
        )

        if shard_id is not None:
            return run_dir / "shards" / f"shard_{shard_id}.jsonl"

        results_dir = root / self.pipeline.settings.outputs.results_dir
        csv_path = results_dir / "results.csv"
        write_csv(rows, csv_path)
        report_path = root / self.pipeline.settings.outputs.reports_dir / "report.md"
        write_markdown_report(summary, report_path, "repair_effectiveness")
        jsonl_path = results_dir / "results.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        return csv_path
