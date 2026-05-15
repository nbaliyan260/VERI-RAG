# VERI-RAG (Python package)

Runnable prototype for **Evidence-Carrying Self-Healing Defense for Secure RAG**.

Full specification, architecture, and results tables: **[../README.md](../README.md)**.

## Quick commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[llm]"
./scripts/run_paper_pipeline.sh
pytest -q
```

Results: `outputs/experiment_results/`, `outputs/poisonedrag/`, `outputs/paper_openai/`, `outputs/hpc_runs/paper_demo/`.
