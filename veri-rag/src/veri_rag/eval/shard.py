"""Sharded experiment execution with resume support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from veri_rag.config.settings import get_project_root


def result_cache_path(
    run_dir: Path,
    query_id: str,
    attack: str,
    defense: str,
    seed: int = 0,
) -> Path:
    key = f"{query_id}_{attack}_{defense}_{seed}"
    return run_dir / "cache" / f"{key}.json"


def load_cached_result(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cached_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, default=str, indent=2), encoding="utf-8")


def shard_tasks(tasks: list[dict], shard_id: int, num_shards: int) -> list[dict]:
    """Deterministic shard by hash of task key."""
    if num_shards <= 1:
        return tasks
    out: list[dict] = []
    for task in tasks:
        key = f"{task.get('query_id')}_{task.get('attack')}_{task.get('defense')}_{task.get('seed', 0)}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        if h % num_shards == shard_id:
            out.append(task)
    return out
