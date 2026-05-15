#!/bin/bash
# =============================================================================
# VERI-RAG HPC Setup Script for MBZUAI Student Lab
# =============================================================================
# Run this AFTER you SSH in and get a compute allocation:
#   ssh nazish.baliyan@login-student-lab.mbzu.ae
#   salloc -N1 -n12 --mem=24G
#   tmux
#   bash scripts/hpc_setup.sh
# =============================================================================

set -e

echo "=== VERI-RAG HPC Environment Setup ==="

# Step 1: Initialize conda
echo "[1/5] Initializing conda..."
source /apps/local/anaconda3/conda_init.sh

# Step 2: Create environment
echo "[2/5] Creating veri-rag conda environment..."
conda create -n veri-rag python=3.11 -y
conda activate veri-rag

# Step 3: Install PyTorch (for sentence-transformers GPU support)
echo "[3/5] Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install VERI-RAG dependencies
echo "[4/5] Installing VERI-RAG dependencies..."
cd ~/veri-rag
pip install -r requirements.txt
pip install -e .

# Step 5: Verify installation
echo "[5/5] Verifying installation..."
python -c "import veri_rag; print(f'VERI-RAG v{veri_rag.__version__} installed successfully')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
nvidia-smi 2>/dev/null && echo "GPU detected" || echo "No GPU (CPU mode)"

echo ""
echo "=== Setup complete! ==="
echo "To activate later:  source /apps/local/anaconda3/conda_init.sh && conda activate veri-rag"
echo "To run VERI-RAG:    cd ~/veri-rag && python -m veri_rag.cli --help"
