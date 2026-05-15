from pathlib import Path

from veri_rag.eval.experiment_runner import ExperimentRunner


def test_experiment_runner_single_task(pipeline, project_root):
    exp_cfg = project_root / "configs" / "experiments.yaml"
    runner = ExperimentRunner(pipeline, exp_cfg)
    tasks = runner.build_task_list()
    assert tasks
    row = runner.run_task(tasks[0])
    assert "query_id" in row
    assert "defense" in row


def test_experiment_shard(pipeline, project_root):
    exp_cfg = project_root / "configs" / "experiments.yaml"
    runner = ExperimentRunner(pipeline, exp_cfg)
    run_dir = project_root / "outputs" / "hpc_runs" / "test_shard"
    rows, _ = runner.run_repair_effectiveness(
        run_dir=run_dir, shard_id=0, num_shards=4, resume=False
    )
    assert isinstance(rows, list)
