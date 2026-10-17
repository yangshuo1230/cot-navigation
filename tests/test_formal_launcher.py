"""Plan/argument/resume checks only. No training, GPU work, smoke or W&B."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("formal_runner", ROOT / "scripts/train_first_round.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def test_formal_plan_has_one_shared_knowledge_parent_and_active_protocols():
    args = runner.parse_args(["--run-name", "unit_plan", "--dry-run"])
    output, jobs = runner.build_jobs(args)
    assert args.protocols == ["direct", "path", "dfs_state"]
    assert len(jobs) == 7
    parents = [job for job in jobs if job.name.endswith("stage_a")]
    assert len(parents) == 1 and parents[0].steps == 1000
    children = [job for job in jobs if job.depends_on]
    assert len(children) == 3 and all(job.depends_on == parents[0].name for job in children)
    assert all(job.steps == 2000 for job in children)
    plan = runner.plan_record(jobs)
    for entry in plan.values():
        cfg = entry["config"]
        assert not cfg["runtime"]["smoke"] and cfg["train"]["persist_outputs"]
        assert cfg["protocol"]["filler_tokens"] is None
        assert cfg["train"]["eval_interval"] == 1000
    assert not output.exists()


def test_fixed_three_graphs_share_data_but_have_distinct_knowledge_parents():
    args = runner.parse_args(["--setting", "fixed", "--graph-seeds", "0", "1", "2", "--run-name", "plan_three"])
    _, jobs = runner.build_jobs(args)
    assert len(jobs) == 12
    parents = {job.name for job in jobs if job.depends_on is None}
    assert len(parents) == 3 and all(job.depends_on in parents for job in jobs if job.depends_on)


def test_resume_can_extend_tasks_but_cannot_change_knowledge_or_data():
    args = runner.parse_args(["--run-name", "unit_resume"])
    _, jobs = runner.build_jobs(args)
    old = runner.plan_record(jobs)
    legacy = copy.deepcopy(old)
    for entry in legacy.values():
        for section, keys in {
            "protocol": ("negative_random_path", "random_search_order"),
            "train": ("shuffle_context_edges", "query_hops", "curriculum_stage_steps",
                      "walk_loss_mode", "walk_loss_weight", "walk_balance_decay"),
            "tracking": ("variant",),
        }.items():
            for key in keys:
                entry["config"][section].pop(key)
    runner.check_resume_plan(legacy, old)
    extended = copy.deepcopy(old)
    for entry in extended.values():
        if entry["config"]["train"]["stage"] != "A":
            entry["config"]["train"]["max_steps"] = 5000
    runner.check_resume_plan(old, extended)
    bad = copy.deepcopy(old)
    bad["fixed/seed0/stage_a"]["config"]["train"]["max_steps"] = 2000
    with pytest.raises(ValueError, match="knowledge-stage"):
        runner.check_resume_plan(old, bad)
    bad = copy.deepcopy(old)
    bad["context/direct"]["config"]["data"]["data_seed"] += 1
    with pytest.raises(ValueError, match="differs"):
        runner.check_resume_plan(old, bad)


def test_formal_runner_rejects_unsafe_names_and_ephemeral_overrides():
    for argv in (["--run-name", "../bad"], ["--resume"], ["--set", "runtime.smoke=true"]):
        with pytest.raises(SystemExit):
            runner.parse_args(argv)


def test_checkpoint_and_parent_arguments_are_distinct():
    args = runner.parse_args(["--setting", "fixed", "--protocols", "dfs_state", "--run-name", "argv_check"])
    _, jobs = runner.build_jobs(args)
    child = jobs[1]
    assert "--init-from" in child.command(parent=Path("knowledge.pt"))
    command = child.command(checkpoint=Path("resume.pt"), parent=Path("knowledge.pt"))
    assert "--resume" in command and "--init-from" not in command


def test_background_creates_log_parent_and_detaches_without_training(tmp_path, monkeypatch):
    argv = ["--run-name", "background_test", "--background", "--context-steps", "10000"]
    args = runner.parse_args(argv)
    output = tmp_path / "absent" / "launches" / args.run_name
    calls = []
    def fake_spawn(command, **kwargs):
        calls.append((command, kwargs))
        assert output.parent.is_dir()
        assert kwargs["stdout"].name == str(output.parent / "background_test.launch.log")
        return type("Process", (), {"pid": 12345})()
    monkeypatch.setattr(runner.subprocess, "Popen", fake_spawn)
    assert runner.start_background(args, output, argv) == 0
    assert not output.exists()  # Child owns the actual run directory/lock.
    command, kwargs = calls[0]
    assert "--background" not in command
    assert command[command.index("--context-steps") + 1] == "10000"
    assert kwargs["start_new_session"] and kwargs["stdin"] == runner.subprocess.DEVNULL
    assert kwargs["stderr"] == runner.subprocess.STDOUT


def test_background_dry_run_and_invalid_gpu_never_spawn_or_create_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "idle_gpus", lambda: [{"index": i, "idle": True} for i in range(8)])
    def forbidden(*args, **kwargs):
        raise AssertionError("no subprocess should be started")
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    output = tmp_path / "missing" / "launches"
    common = ["--run-name", "preflight", "--background", "--output-root", str(output)]
    assert runner.main(common + ["--dry-run", "--gpus", "0"]) == 0
    assert not output.exists()
    with pytest.raises(ValueError, match="GPU indices do not exist.*45"):
        runner.main(common + ["--gpus", "0", "45"])
    assert not output.exists()
