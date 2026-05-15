"""Typer CLI for VERI-RAG."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from veri_rag.config.schema import AttackType
from veri_rag.config.settings import (
    apply_llm_profile,
    ensure_output_dirs,
    get_project_root,
    load_settings,
)

try:
    from dotenv import load_dotenv

    load_dotenv(get_project_root() / ".env")
except ImportError:
    pass
from veri_rag.corpus.ingest import ingest_corpus
from veri_rag.corpus.synthetic import create_synthetic_corpus, load_query_set
from veri_rag.eval.experiment_runner import ExperimentRunner
from veri_rag.pipeline import VERIRAGPipeline

app = typer.Typer(
    name="veri-rag",
    help="VERI-RAG: Evidence-Carrying Self-Healing Defense for Secure RAG",
    add_completion=False,
)
console = Console()


def _resolve_config(config: str) -> Path:
    root = get_project_root()
    path = Path(config)
    if not path.is_absolute():
        path = root / path
    return path


def _load_pipeline(config: str, build: bool = False) -> VERIRAGPipeline:
    cfg = _resolve_config(config)
    settings = load_settings(cfg)
    ensure_output_dirs(settings)
    if build:
        index = ingest_corpus(settings)
        return VERIRAGPipeline(settings, index)
    return VERIRAGPipeline.from_config(cfg, build_index=False)


@app.command("create-synthetic-corpus")
def create_synthetic(
    output_dir: str = typer.Option("data/synthetic_enterprise"),
) -> None:
    """Generate synthetic enterprise corpus and query set."""
    root = get_project_root()
    create_synthetic_corpus(root / output_dir)
    console.print(f"[green]Created synthetic corpus at {root / output_dir}[/green]")


@app.command("download-benchmark")
def download_benchmark(
    name: str = typer.Option(..., "--name", help="poisonedrag"),
    dataset: str = typer.Option("nq", "--dataset", help="nq | hotpotqa | msmarco"),
    max_queries: int = typer.Option(20, "--max-queries", help="Limit queries for laptop runs"),
    sample_only: bool = typer.Option(
        False, "--sample-only", help="Use built-in sample instead of GitHub download"
    ),
    clone_repo: bool = typer.Option(
        False, "--clone-repo", help="Shallow-clone full PoisonedRAG repo"
    ),
) -> None:
    """Download PoisonedRAG attacks from GitHub and export VERI-RAG format."""
    if name != "poisonedrag":
        console.print(f"[red]Unknown benchmark: {name}[/red]")
        raise typer.Exit(1)
    from veri_rag.corpus.benchmarks.poisonedrag import PoisonedRAGLoader

    loader = PoisonedRAGLoader()
    if sample_only:
        loader.ensure_sample_data()
        out = loader.export_for_ingest()
    else:
        console.print(
            f"[dim]Downloading from "
            f"https://github.com/sleeepeer/PoisonedRAG ({dataset}, max={max_queries})...[/dim]"
        )
        out = loader.download(dataset=dataset, max_queries=max_queries, clone_repo=clone_repo)
    console.print(f"[green]PoisonedRAG data at {loader.data_dir}[/green]")
    console.print(f"[dim]Clean corpus: {out}[/dim]")


@app.command("ingest")
def ingest(config: str = typer.Option("configs/mvp.yaml")) -> None:
    """Ingest documents, chunk, embed, and build the vector index."""
    settings = load_settings(_resolve_config(config))
    artifacts = ingest_corpus(settings)
    console.print(
        f"[green]Indexed {artifacts.vector_store.count()} chunks "
        f"(dim={artifacts.embedder.dimension})[/green]"
    )


@app.command("ask")
def ask(
    question: str = typer.Argument(...),
    config: str = typer.Option("configs/mvp.yaml"),
    query_id: str = typer.Option(""),
) -> None:
    """Run baseline RAG for a single question."""
    pipe = _load_pipeline(config)
    answer = pipe.rag.ask(question, query_id=query_id or "adhoc")
    console.print(f"\n[bold]Answer:[/bold] {answer.answer}\n")
    console.print(f"[dim]Chunks: {', '.join(r.chunk.chunk_id for r in answer.retrieved_chunks)}[/dim]")


@app.command("generate-attacks")
def generate_attacks(
    attack: str = typer.Option(..., "--attack"),
    config: str = typer.Option("configs/mvp.yaml"),
    output_dir: str = typer.Option("data/attacked_corpus"),
) -> None:
    """Generate and save malicious chunks for all queries."""
    attack_type = AttackType(attack.replace("-", "_"))
    root = get_project_root()
    exp_path = _resolve_config(config).parent / "experiments.yaml"
    from veri_rag.attacks.runner import AttackRunner, load_attack_config_from_experiments

    runner = AttackRunner(
        load_attack_config_from_experiments(exp_path) if exp_path.exists() else None
    )
    qpath = root / "data" / "synthetic_enterprise" / "queries" / "enterprise_qa.jsonl"
    queries = load_query_set(qpath) if qpath.exists() else []
    if not queries:
        raise typer.Exit(1)
    out = runner.save_attack_corpus(attack_type, queries, root / output_dir)
    console.print(f"[green]Wrote {out}[/green]")


@app.command("scan-risk")
def scan_risk(
    query_id: str = typer.Option(..., "--query-id"),
    config: str = typer.Option("configs/mvp.yaml"),
    attack: str = typer.Option("poisoning"),
) -> None:
    """Risk-scan retrieved chunks for a query."""
    pipe = _load_pipeline(config)
    q = _get_query(pipe, query_id)
    retrieved = _attacked_retrieval(pipe, q, AttackType(attack.replace("-", "_")))
    scores = pipe.risk_scorer.score_all(q["query"], retrieved)
    table = Table(title=f"Risk — {query_id}")
    table.add_column("Chunk")
    table.add_column("Risk")
    table.add_column("Reason")
    for cid, rs in sorted(scores.items(), key=lambda x: -x[1].risk_score):
        table.add_row(cid, f"{rs.risk_score:.2f}", "; ".join(rs.risk_reasons))
    console.print(table)


@app.command("analyze-influence")
def analyze_influence(
    query_id: str = typer.Option(..., "--query-id"),
    config: str = typer.Option("configs/mvp.yaml"),
    attack: str = typer.Option("poisoning"),
    mode: str = typer.Option("riaa", "--mode"),
    pairwise: bool = typer.Option(True, "--pairwise/--no-pairwise"),
) -> None:
    """RIAA / LOO influence analysis for a query."""
    pipe = _load_pipeline(config)
    q = _get_query(pipe, query_id)
    at = AttackType(attack.replace("-", "_"))
    retrieved = _attacked_retrieval(pipe, q, at)
    risk = pipe.risk_scorer.score_all(q["query"], retrieved)
    baseline = pipe.rag.ask(q["query"], query_id=query_id, chunks_override=retrieved)

    if mode == "riaa":
        result = pipe.riaa.analyze(
            q["query"], query_id, retrieved, risk, baseline, enable_pairwise=pairwise
        )
        scores = result.loo_scores
        console.print(f"[bold]Coordinated pairs:[/bold] {result.coordinated_pairs}")
    else:
        _, scores = pipe.riaa.loo.analyze(q["query"], query_id, retrieved, baseline)

    table = Table(title=f"Influence ({mode}) — {query_id}")
    table.add_column("Chunk")
    table.add_column("Influence")
    table.add_column("Harmful")
    for cid, inf in sorted(scores.items(), key=lambda x: -x[1].influence_score):
        table.add_row(cid, f"{inf.influence_score:.2f}", str(inf.is_harmful))
    console.print(table)


@app.command("train-calibrator")
def train_calibrator(config: str = typer.Option("configs/mvp.yaml")) -> None:
    """Train RIAA harmfulness calibrator on synthetic attacks."""
    pipe = _load_pipeline(config)
    path = pipe.train_calibrator()
    console.print(f"[green]Calibrator saved to {path}[/green]")


@app.command("run-baseline")
def run_baseline(
    query_id: str = typer.Option(..., "--query-id"),
    method: str = typer.Option(..., "--method"),
    config: str = typer.Option("configs/mvp.yaml"),
    attack: str = typer.Option("poisoning"),
) -> None:
    """Run a published-style baseline defense."""
    pipe = _load_pipeline(config)
    q = _get_query(pipe, query_id)
    retrieved = _attacked_retrieval(pipe, q, AttackType(attack.replace("-", "_")))
    ans = pipe.baseline_runner.run(
        method,  # type: ignore[arg-type]
        q["query"],
        query_id,
        retrieved,
        pipe.settings.retrieval.top_k,
    )
    console.print(f"[bold]{method}:[/bold] {ans}")


@app.command("repair")
def repair_cmd(
    query_id: str = typer.Option(..., "--query-id"),
    config: str = typer.Option("configs/mvp.yaml"),
    attack: str = typer.Option("poisoning"),
) -> None:
    """Run full VERI-RAG repair for one query."""
    pipe = _load_pipeline(config)
    q = _get_query(pipe, query_id)
    result = pipe.run_with_attack(
        query_id, q["query"], AttackType(attack.replace("-", "_")), q.get("gold_answer", "")
    )
    console.print(f"[bold]Original:[/bold] {result['baseline_answer'][:200]}")
    console.print(f"[bold]Repaired:[/bold] {result['final_answer'][:200]}")
    console.print(f"[dim]Removed: {result.get('removed_chunks', [])}[/dim]")
    console.print(f"[dim]Coordinated: {result.get('coordinated_pairs', [])}[/dim]")


@app.command("verify")
def verify_cmd(
    query_id: str = typer.Option(..., "--query-id"),
    config: str = typer.Option("configs/mvp.yaml"),
    attack: str = typer.Option("poisoning"),
    certified_smoothing: bool = typer.Option(True, "--certified-smoothing/--no-certified-smoothing"),
) -> None:
    """Run repair + verification certificate (with optional certified smoothing)."""
    pipe = _load_pipeline(config)
    q = _get_query(pipe, query_id)
    result = pipe.run_with_attack(
        query_id,
        q["query"],
        AttackType(attack.replace("-", "_")),
        q.get("gold_answer", ""),
        enable_certified=certified_smoothing,
    )
    root = get_project_root()
    cert_path = root / "outputs" / "certificates" / f"{query_id}.json"
    console.print(f"[bold]Certificate score:[/bold] {result.get('certificate_score')}")
    console.print(f"[bold]ε_raw:[/bold] {result.get('certified_bound_raw')}")
    console.print(f"[bold]ε_post:[/bold] {result.get('certified_bound_post')}")
    if cert_path.exists():
        console.print(f"[green]Saved: {cert_path}[/green]")


@app.command("run-experiment")
def run_experiment(config: str = typer.Option("configs/mvp.yaml")) -> None:
    """Run full experiment matrix (local)."""
    pipe = _load_pipeline(config)
    runner = ExperimentRunner(pipe, _resolve_config(config).parent / "experiments.yaml")
    csv_path = runner.run_all()
    console.print(f"[green]Results: {csv_path}[/green]")


@app.command("run-experiment-shard")
def run_experiment_shard(
    config: str = typer.Option("configs/hpc_template.yaml"),
    run_id: str = typer.Option("hpc_default", "--run-id"),
    shard_id: int = typer.Option(0, "--shard-id"),
    num_shards: int = typer.Option(10, "--num-shards"),
) -> None:
    """Run one shard of the experiment matrix (HPC)."""
    pipe = _load_pipeline(config)
    runner = ExperimentRunner(pipe, _resolve_config(config).parent / "experiments.yaml")
    out = runner.run_all(shard_id=shard_id, num_shards=num_shards, run_id=run_id)
    console.print(f"[green]Shard {shard_id} -> {out}[/green]")


@app.command("merge-hpc-results")
def merge_hpc(
    run_dir: str = typer.Option(..., "--run-dir", help="outputs/hpc_runs/<run_id>"),
) -> None:
    """Merge HPC shard outputs into CSV and report."""
    root = get_project_root()
    path = Path(run_dir)
    if not path.is_absolute():
        path = root / path
    script = root / "hpc" / "merge_results.py"
    subprocess.run(["python", str(script), "--run-dir", str(path)], check=True)


@app.command("run-paper-pipeline")
def run_paper_pipeline(
    config: str = typer.Option("configs/mvp.yaml"),
    max_poisonedrag: int = typer.Option(15, "--max-poisonedrag"),
    skip_experiments: bool = typer.Option(False, "--skip-experiments"),
) -> None:
    """Run full paper-scale setup: corpus, calibrator, PoisonedRAG, experiments."""
    root = get_project_root()
    create_synthetic_corpus(root / "data/synthetic_enterprise")
    ingest(config=config)

    for atk in ["poisoning", "prompt_injection", "secret_leakage", "blocker", "topic_flip", "adaptive"]:
        try:
            generate_attacks(attack=atk, config=config)
        except Exception as exc:
            console.print(f"[yellow]Skip attack {atk}: {exc}[/yellow]")

    train_calibrator(config=config)

    download_benchmark(
        name="poisonedrag",
        dataset="nq",
        max_queries=max_poisonedrag,
        sample_only=False,
        clone_repo=False,
    )
    ingest(config="configs/poisonedrag.yaml")

    if not skip_experiments:
        run_experiment(config=config)
        run_experiment(config="configs/poisonedrag.yaml")
        run_experiment_shard(
            config="configs/hpc_template.yaml",
            run_id="paper_demo",
            shard_id=0,
            num_shards=2,
        )
        run_experiment_shard(
            config="configs/hpc_template.yaml",
            run_id="paper_demo",
            shard_id=1,
            num_shards=2,
        )
        merge_hpc(run_dir="outputs/hpc_runs/paper_demo")
    console.print("[green]Paper pipeline complete.[/green]")


@app.command("run-paper-llm")
def run_paper_llm(
    profile: str = typer.Option("mock", "--profile", help="mock | openai | ollama_llama"),
    max_queries: int = typer.Option(4, "--max-queries"),
    fallback_mock: bool = typer.Option(
        True,
        "--fallback-mock/--no-fallback-mock",
        help="If OpenAI/Ollama unavailable, run with mock LLM",
    ),
) -> None:
    """Run a small experiment matrix (mock by default; OpenAI/Ollama optional)."""
    from veri_rag.rag.llm_health import resolve_profile_with_fallback

    cfg_path = _resolve_config("configs/paper_openai.yaml")
    settings = load_settings(cfg_path)
    effective, warning = resolve_profile_with_fallback(
        profile,
        fallback_mock=fallback_mock,
        model_name=settings.llm.model_name,
    )
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")
        console.print(f"[dim]Continuing with profile: {effective}[/dim]")
    settings = apply_llm_profile(settings, effective)
    settings.experiment.max_queries = max_queries
    ensure_output_dirs(settings)
    from veri_rag.corpus.ingest import ingest_corpus

    index = ingest_corpus(settings)
    pipe = VERIRAGPipeline(settings, index)
    pipe.reload_llm()
    exp_path = cfg_path.parent / "experiments.yaml"
    runner = ExperimentRunner(pipe, exp_path if exp_path.exists() else None)
    csv_path = runner.run_all(run_id=f"llm_{effective}")
    console.print(f"[green]LLM experiment ({effective}): {csv_path}[/green]")


@app.command("run-attack-eval")
def run_attack_eval(config: str = typer.Option("configs/mvp.yaml")) -> None:
    """Quick attack evaluation on q001."""
    pipe = _load_pipeline(config)
    row = pipe.run_with_attack(
        "q001", "What is the refund period?", AttackType.POISONING, "30 days"
    )
    console.print(json.dumps(row, indent=2, default=str))


def _get_query(pipe: VERIRAGPipeline, query_id: str) -> dict[str, str]:
    root = get_project_root()
    for rel in (
        "data/synthetic_enterprise/queries/enterprise_qa.jsonl",
        "data/benchmarks/poisonedrag/queries.jsonl",
    ):
        qpath = root / rel
        if qpath.exists():
            for q in load_query_set(qpath):
                if q["query_id"] == query_id:
                    return q
    raise typer.BadParameter(f"Unknown query_id: {query_id}")


def _attacked_retrieval(pipe: VERIRAGPipeline, q: dict, attack_type: AttackType):
    clean = pipe.retrieve_clean(q["query"])
    spec = pipe.attack_runner.build_spec(
        attack_type, q["query_id"], q["query"], q.get("gold_answer", "")
    )
    malicious = pipe.attack_runner.generate_chunks(spec)
    return pipe.attack_runner.inject_into_retrieval(
        clean, malicious, pipe.settings.retrieval.top_k
    )


if __name__ == "__main__":
    app()
