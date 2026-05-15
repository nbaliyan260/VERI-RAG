#!/usr/bin/env python3
"""Merge HPC shard JSONL outputs into a single CSV and Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def merge_shards(run_dir: Path) -> tuple[list[dict], Path]:
    rows: list[dict] = []
    shards_dir = run_dir / "shards"
    if shards_dir.exists():
        for path in sorted(shards_dir.glob("shard_*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
    cache_dir = run_dir / "cache"
    if not rows and cache_dir.exists():
        for path in cache_dir.glob("*.json"):
            rows.append(json.loads(path.read_text(encoding="utf-8")))

    out_csv = run_dir / "merged_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows, out_csv


def write_report(rows: list[dict], run_dir: Path) -> Path:
    report = run_dir / "final_report.md"
    by_defense: dict[str, list[dict]] = {}
    for r in rows:
        by_defense.setdefault(r.get("defense", "unknown"), []).append(r)

    lines = ["# VERI-RAG HPC Merged Report", ""]
    for defense, subset in sorted(by_defense.items()):
        asr = sum(1 for r in subset if r.get("attack_success")) / max(len(subset), 1)
        lines.append(f"## {defense}")
        lines.append(f"- attack_success_rate: {asr:.4f}")
        lines.append(f"- n: {len(subset)}")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="outputs/hpc_runs/<run_id>")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows, csv_path = merge_shards(run_dir)
    report = write_report(rows, run_dir)
    print(f"Merged {len(rows)} rows -> {csv_path}")
    print(f"Report -> {report}")


if __name__ == "__main__":
    main()
