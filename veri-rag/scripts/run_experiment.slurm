#!/bin/bash
# =============================================================================
# VERI-RAG Experiment SLURM Job Script
# =============================================================================
# Submit with:  sbatch scripts/run_experiment.slurm
# =============================================================================

#SBATCH --job-name=veri-rag-exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm_%j.log
#SBATCH --error=outputs/slurm_%j.err

# Initialize environment
source /apps/local/anaconda3/conda_init.sh
conda activate veri-rag

cd ~/veri-rag

echo "=== VERI-RAG Experiment Starting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time:   $(date)"
nvidia-smi 2>/dev/null || echo "Running in CPU mode"

# Step 1: Create synthetic corpus (if not already done)
python -m veri_rag.cli create-synthetic-corpus
echo "[✓] Synthetic corpus ready"

# Step 2: Ingest into vector store
python -m veri_rag.cli ingest --config configs/mvp.yaml
echo "[✓] Corpus ingested"

# Step 3: Run full experiment suite
python -m veri_rag.cli run-experiment --config configs/experiments.yaml
echo "[✓] Experiments complete"

echo ""
echo "=== Results ==="
echo "CSV:          outputs/experiment_results/"
echo "Certificates: outputs/certificates/"
echo "Reports:      outputs/reports/"
echo "Finished at:  $(date)"
