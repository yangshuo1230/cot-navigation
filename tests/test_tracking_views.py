"""Pure logging/layout checks: no model, dataset, GPU, or W&B run is created."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cot_navigation.config import load_config
from cot_navigation.tracking.logger import Tracker, task_name, visible_metrics


def config():
    return load_config([Path(__file__).parents[1] / "configs/experiments/main_context.yaml"])


@pytest.mark.parametrize("knowledge,stage,protocol,task,count", [
    ("context_graph", "B", "direct", "context_navigation", 2),
    ("context_graph", "B", "filler", "context_navigation", 2),
    ("context_graph", "B", "path", "context_navigation", 3),
    ("context_graph", "B", "dfs_state", "context_navigation", 4),
    ("parametric_graph", "B", "dfs_state", "fixed_navigation", 5),
    ("parametric_graph", "A", "direct", "fixed_knowledge", 2),
])
def test_task_metric_selection(knowledge, stage, protocol, task, count):
    cfg = config()
    cfg.data.knowledge, cfg.train.stage, cfg.protocol.name = knowledge, stage, protocol
    assert task_name(cfg) == task
    assert len(visible_metrics(cfg)) == count


def test_full_local_metrics_slim_online_and_shared_step(tmp_path, monkeypatch):
    cfg = config()
    cfg.output_dir = str(tmp_path)
    cfg.tracking.mode = "offline"
    cfg.train.persist_outputs = True
    cfg.runtime.smoke = False
    cfg.data.knowledge = "context_graph"
    cfg.protocol.name = "direct"
    definitions, events, init_kwargs = [], [], {}
    fake_run = SimpleNamespace(
        mode="offline", url=None,
        define_metric=lambda *a, **kw: definitions.append((a, kw)),
        log=lambda row, **kw: events.append(row), finish=lambda: None,
    )
    def init(**kwargs):
        init_kwargs.update(kwargs)
        return fake_run
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    tracker = Tracker(cfg, "not-a-real-run", None, {})
    tracker.log(1, {"answer_loss": 0.7, "diagnostic/probe": 13, "bad": float("nan")})
    tracker.log(1, {"free/overall/answer_accuracy": 0.75})
    tracker.log(1, {"diagnostic/only": 2})
    tracker.finish()
    assert init_kwargs["job_type"] == "context_navigation"
    assert events == [{"optimizer_step": 1, "answer_loss": 0.7},
                      {"optimizer_step": 1, "free/overall/answer_accuracy": 0.75}]
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["diagnostic/probe"] == 13
    assert rows[2]["diagnostic/only"] == 2 and "bad" not in rows[0]
    assert (("*",), {"step_metric": "optimizer_step", "hidden": True}) in definitions
    assert sum(not kw["hidden"] for _, kw in definitions) == 2


def test_task_views_are_manual_filtered_and_use_existing_metrics():
    pytest.importorskip("wandb_workspaces")
    spec = importlib.util.spec_from_file_location("organize_wandb", Path(__file__).parents[1] / "scripts/organize_wandb.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    for task, count in [("context_navigation", 4), ("fixed_navigation", 5), ("fixed_knowledge", 2)]:
        workspace = script.build_workspace("test", "test", "batch-50k", task)
        assert workspace.auto_generate_panels is False
        assert len(workspace.sections[0].panels) == count
        assert all(p.x == "optimizer_step" for p in workspace.sections[0].panels)
        model = workspace._to_model()
        roundtrip = script.ws.Workspace._from_model(model)
        assert "batch-50k" in roundtrip.runset_settings.filters
        assert script.TASKS[task][1] in roundtrip.runset_settings.filters
        assert roundtrip.auto_generate_panels is False


def test_context_groups_are_selectable_and_sizes_are_not_averaged_together():
    pytest.importorskip("wandb_workspaces")
    spec = importlib.util.spec_from_file_location("organize_wandb", Path(__file__).parents[1] / "scripts/organize_wandb.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    workspace = script.build_workspace("test", "test", None, "context_navigation")
    raw = json.loads(workspace._to_model().model_dump_json(by_alias=True))
    runset = raw["spec"]["section"]["runSets"][0]
    assert all(f["key"]["name"] != "group" for f in runset["filters"]["filters"])
    assert runset["grouping"] == [
        {"section": "run", "name": "group"},
        {"section": "config", "name": "experiment.value.data.splits.train"},
        {"section": "config", "name": "experiment.value.data.shapes.balanced.layers"},
        {"section": "config", "name": "experiment.value.tracking.variant"},
        {"section": "config", "name": "experiment.value.protocol.name"},
    ]
    assert runset["runFeed"]["columnVisible"]["run:group"] is True
    assert "run:group" in runset["runFeed"]["columnOrder"]
    for section in raw["spec"]["section"]["panelBankConfig"]["sections"]:
        for panel in section["panels"]:
            assert not panel["config"].get("groupBy")
    roundtrip = script.ws.Workspace._from_model(workspace._to_model())
    assert "context_graph" in roundtrip.runset_settings.filters
    assert len(roundtrip.runset_settings.groupby) == 5
