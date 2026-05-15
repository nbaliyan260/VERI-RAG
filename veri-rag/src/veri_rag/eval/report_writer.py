"""Markdown and CSV report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_markdown_report(
    summary: dict[str, dict[str, float]],
    path: Path,
    experiment_name: str = "veri-rag",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# VERI-RAG Experiment Report: {experiment_name}",
        "",
        "## Summary",
        "",
    ]
    for section, metrics in summary.items():
        lines.append(f"### {section}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Lower `attack_success_rate` indicates stronger defense.\n"
        "- Higher `repair_success_rate` and `certificate_score` indicate effective self-healing.\n"
        "- Higher localization precision/recall indicates RIAA/risk scoring quality."
    )
    path.write_text("\n".join(lines), encoding="utf-8")
