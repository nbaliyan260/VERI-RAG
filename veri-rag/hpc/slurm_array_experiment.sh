#!/bin/bash
#SBATCH --job-name=veri-rag
#SBATCH --output=logs/veri-rag-%A_%a.out
#SBATCH --error=logs/veri-rag-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-9

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
source .venv/bin/activate

RUN_ID="${RUN_ID:-hpc_default}"
NUM_SHARDS="${NUM_SHARDS:-10}"

veri-rag run-experiment-shard \
  --config configs/hpc_template.yaml \
  --run-id "$RUN_ID" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS"
