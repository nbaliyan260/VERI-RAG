# VERI-RAG (Python package)

Evidence-carrying self-healing defense for secure RAG — **runnable research prototype**.

**Full documentation, architecture, experiment tables, and project log:** [../README.md](../README.md)

## Install & run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[llm]"
cp .env.example .env    # ANTHROPIC_API_KEY for Claude; OPENAI_API_KEY optional

./scripts/veri-rag.sh run-paper-llm --profile auto --max-queries 4
./scripts/run_paper_pipeline.sh
pytest -q               # 20 tests
```

## Results (committed on GitHub)

| Run | Path |
|-----|------|
| Enterprise | `outputs/experiment_results/results.csv` |
| PoisonedRAG | `outputs/poisonedrag/experiment_results/results.csv` |
| Claude LLM | `outputs/paper_claude/results/results.csv` |
