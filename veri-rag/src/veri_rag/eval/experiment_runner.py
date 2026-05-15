"""Experiment runner for MVP evaluation suite with optional HPC sharding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from veri_rag.attacks.runner import load_attack_config_from_experiments
from veri_rag.config.schema import AttackType
from veri_rag.config.settings import Settings, get_project_root
from veri_rag.corpus.benchmarks.poisonedrag import PoisonedRAGLoader
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

    def __init__(
        self,
        pipeline: VERIRAGPipeline,
        experiments_config: Path | None = None,
    ):
        self.pipeline = pipeline
        self.settings = pipeline.settings
        self.metrics = MetricsCalculator()
        self.config: dict[str, Any] = {}
        if experiments_config and experiments_config.exists():
            with open(experiments_config, encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        from veri_rag.attacks.runner import DEFAULT_ATTACK_CONFIG, AttackRunner

        if experiments_config and experiments_config.exists():
            self.attack_runner_config = load_attack_config_from_experiments(experiments_config)
        else:
            self.attack_runner_config = DEFAULT_ATTACK_CONFIG
        self.pipeline.attack_runner = AttackRunner(self.attack_runner_config)

    def load_queries(self) -> list[dict[str, str]]:
        exp = self.settings.experiment
        if exp.benchmark == "poisonedrag":
            queries = PoisonedRAGLoader().load_queries(max_queries=exp.max_queries)
            return queries
        qs = self.config.get("query_sets", {}).get("enterprise_qa")
        if qs:
            queries = qs
        else:
            queries = []
            root = get_project_root()
            for rel in (
                "data/synthetic_enterprise/queries/enterprise_qa.jsonl",
                "data/benchmarks/poisonedrag/queries.jsonl",
            ):
                qpath = root / rel
                if qpath.exists():
                    queries = load_query_set(qpath)
                    break
        if exp.max_queries is not None:
            queries = queries[: exp.max_queries]
        return queries

    def build_task_list(self) -> list[dict[str, Any]]:
        queries = self.load_queries()
        exp = self.settings.experiment
        attack_names = exp.attacks
        defenses = exp.defenses
        is_poisonedrag = exp.benchmark == "poisonedrag"

        tasks: list[dict[str, Any]] = []
        for q in queries:
            for attack_name in attack_names:
                attack = AttackType(attack_name.replace("-", "_"))
                if not is_poisonedrag:
                    if attack == AttackType.BLOCKER and q["query_id"] not in [
                        "q001",
                        "q002",
                        "q004",
                    ]:
                        continue
                    if attack == AttackType.TOPIC_FLIP and q["query_id"] != "q005":
                        continue
                if is_poisonedrag and attack not in (
                    AttackType.POISONING,
                    AttackType.PROMPT_INJECTION,
                    AttackType.ADAPTIVE,
                    AttackType.SECRET_LEAKAGE,
                ):
                    continue
                for defense in defenses:
                    tasks.append(
                        {
                            "query_id": q["query_id"],
                            "query": q["query"],
                            "gold_answer": q.get("gold_answer", ""),
                            "target_wrong_answer": q.get("target_wrong_answer", ""),
                            "attack": attack.value,
                            "defense": defense,
                            "seed": 0,
                            "benchmark": exp.benchmark,
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
        row["benchmark"] = task.get("benchmark", self.settings.experiment.benchmark)
        if task.get("target_wrong_answer"):
            row["attack_success"] = self.metrics.attack_success(
                AttackType(task["attack"]),
                row.get("final_answer", row.get("baseline_answer", "")),
                task.get("gold_answer", ""),
                task["target_wrong_answer"],
            )
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

            if shard_id is not None:
                jsonl_shard = run_dir / "shards" / f"shard_{shard_id}.jsonl"
                jsonl_shard.parent.mkdir(parents=True, exist_ok=True)
                with open(jsonl_shard, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")

        all_tasks = self.build_task_list()
        summary: dict[str, dict[str, float]] = {}
        for defense in {t["defense"] for t in all_tasks}:
            subset = [r for r in rows if r.get("defense") == defense]
            if not subset:
                continue
            n = len(subset)
            summary[defense] = {
                "attack_success_rate": sum(1 for r in subset if r.get("attack_success")) / n,
                "repair_success_rate": sum(1 for r in subset if r.get("repair_success")) / n,
                "mean_certificate_score": sum(r.get("certificate_score", 0) for r in subset) / n,
                "mean_certified_bound_post": sum(
                    r.get("certified_bound_post") or 0 for r in subset
                )
                / n,
                "precision_at_1": sum(r.get("precision_at_1", 0) for r in subset) / n,
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
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "results.csv"
        write_csv(rows, csv_path)
        report_path = root / self.pipeline.settings.outputs.reports_dir / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(
            summary, report_path, self.settings.experiment.benchmark
        )
        jsonl_path = results_dir / "results.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        return csv_path
