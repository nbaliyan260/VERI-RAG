#!/usr/bin/env bash
# End-to-end paper-scale pipeline (laptop-friendly subset).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

CONFIG="${1:-configs/mvp.yaml}"
LLM_PROFILE="${2:-auto}"
MAX_PR="${3:-20}"

echo "==> Phase 1: Synthetic corpus + ingest"
veri-rag create-synthetic-corpus
veri-rag ingest --config "$CONFIG"

echo "==> Phase 2: Generate attack corpora"
for atk in poisoning prompt-injection secret-leakage blocker topic-flip adaptive; do
  veri-rag generate-attacks --attack "$atk" --config "$CONFIG" || true
done

echo "==> Phase 3: Train RIAA calibrator"
veri-rag train-calibrator --config "$CONFIG"

echo "==> Phase 4: PoisonedRAG download (NQ subset) + ingest"
veri-rag download-benchmark --name poisonedrag --dataset nq --max-queries "$MAX_PR"
veri-rag ingest --config configs/poisonedrag.yaml

echo "==> Phase 5: Enterprise defense comparison"
veri-rag run-experiment --config "$CONFIG"

echo "==> Phase 6: PoisonedRAG defense comparison"
veri-rag run-experiment --config configs/poisonedrag.yaml

echo "==> Phase 7: HPC shard demo (2 shards) + merge"
veri-rag run-experiment-shard --config configs/hpc_template.yaml --run-id paper_demo --shard-id 0 --num-shards 2
veri-rag run-experiment-shard --config configs/hpc_template.yaml --run-id paper_demo --shard-id 1 --num-shards 2
veri-rag merge-hpc-results --run-dir outputs/hpc_runs/paper_demo

echo "==> Phase 8: LLM profile experiments (falls back to mock if API unavailable)"
veri-rag run-paper-llm --profile "$LLM_PROFILE" --max-queries 4 --fallback-mock

echo "Done. See outputs/ and outputs/poisonedrag/"
