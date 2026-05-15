"""Download and convert PoisonedRAG attack data from the official GitHub repo."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from veri_rag.config.settings import get_project_root

POISONEDRAG_REPO = "https://github.com/sleeepeer/PoisonedRAG"
RAW_BASE = "https://raw.githubusercontent.com/sleeepeer/PoisonedRAG/main"
ADV_FILES = {
    "nq": f"{RAW_BASE}/results/adv_targeted_results/nq.json",
    "hotpotqa": f"{RAW_BASE}/results/adv_targeted_results/hotpotqa.json",
    "msmarco": f"{RAW_BASE}/results/adv_targeted_results/msmarco.json",
}


def _fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def convert_adv_json_to_veri_rag(
    data: dict[str, Any],
    dataset: str,
    max_queries: int | None = 50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert PoisonedRAG adv_targeted_results JSON to query and attack rows."""
    queries: list[dict[str, Any]] = []
    attacks: list[dict[str, Any]] = []

    for i, (key, row) in enumerate(data.items()):
        if max_queries is not None and i >= max_queries:
            break
        qid = f"pr_{dataset}_{row.get('id', key)}"
        question = row.get("question", "")
        gold = row.get("correct answer") or row.get("correct_answer", "")
        wrong = row.get("incorrect answer") or row.get("incorrect_answer", "")
        adv_texts = row.get("adv_texts", [])

        clean_passage = (
            f"Reference answer for the question '{question}': "
            f"The verified correct answer is {gold}."
        )

        queries.append(
            {
                "query_id": qid,
                "query": question,
                "gold_answer": gold,
                "target_wrong_answer": wrong,
                "clean_passage": clean_passage,
                "dataset": dataset,
            }
        )

        for j, adv in enumerate(adv_texts[:5]):
            attacks.append(
                {
                    "query_id": qid,
                    "attack_index": j,
                    "adv_passage": adv,
                    "target_answer": wrong,
                }
            )

    return queries, attacks


def download_poisonedrag_dataset(
    dataset: str = "nq",
    output_dir: str | Path | None = None,
    max_queries: int | None = 50,
) -> Path:
    """Download adv_targeted_results JSON and convert to VERI-RAG format.

    Source: https://github.com/sleeepeer/PoisonedRAG
    """
    if dataset not in ADV_FILES:
        raise ValueError(f"Unknown dataset {dataset}. Choose from {list(ADV_FILES)}")

    root = get_project_root()
    output_dir = Path(output_dir or root / "data" / "benchmarks" / "poisonedrag")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw" / f"{dataset}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    url = ADV_FILES[dataset]
    data = _fetch_json(url)
    raw_path.write_text(json.dumps(data), encoding="utf-8")

    queries, attacks = convert_adv_json_to_veri_rag(data, dataset, max_queries=max_queries)

    for row in queries:
        clean_file = output_dir / "clean_corpus" / f"{row['query_id']}.txt"
        clean_file.parent.mkdir(parents=True, exist_ok=True)
        clean_file.write_text(row["clean_passage"], encoding="utf-8")

    queries_path = output_dir / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    attacks_path = output_dir / f"attacks_{dataset}.jsonl"
    with open(attacks_path, "w", encoding="utf-8") as f:
        for a in attacks:
            f.write(json.dumps(a) + "\n")

    meta = {
        "source": POISONEDRAG_REPO,
        "dataset": dataset,
        "num_queries": len(queries),
        "num_adv_texts": len(attacks),
        "raw_file": str(raw_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_dir


def clone_poisonedrag_repo(
    output_dir: str | Path | None = None,
    shallow: bool = True,
) -> Path:
    """Optional: shallow clone full PoisonedRAG repo for offline use."""
    import subprocess

    root = get_project_root()
    dest = Path(output_dir or root / "data" / "benchmarks" / "poisonedrag" / "upstream_repo")
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", POISONEDRAG_REPO + ".git", str(dest)]
    subprocess.run(cmd, check=True)
    return dest
