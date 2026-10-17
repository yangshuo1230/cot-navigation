"""Create three small, task-specific W&B views without touching run histories.

Defaults to a local preview. --apply saves views after backing up their original
specifications. Re-running updates the same named views rather than duplicating
them. Install the project's dev dependencies to use this maintenance script.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import wandb
from wandb_workspaces import _graphql, workspaces as ws
from wandb_workspaces.reports import v2 as panels
from wandb_workspaces.workspaces import internal


TASKS = {
    "context_navigation": ("01 新图导航", "context_graph", None),
    "fixed_navigation": ("02 固定图导航", "parametric_graph", "B"),
    "fixed_knowledge": ("03 固定图知识阶段", "parametric_graph", "A"),
}
TITLES = {
    "answer_loss": "训练 · 答案 loss",
    "trace_loss": "训练 · CoT 轨迹 loss",
    "walk_loss": "训练 · 游走 loss",
    "free/overall/answer_accuracy": "验证 · 自由生成答案准确率",
    "free/trace/search_complete_rate": "验证 · 搜索完整且匹配参考轨迹的比例",
    "lr": "训练 · 学习率",
}


def build_workspace(entity: str, project: str, group: str | None, task: str) -> ws.Workspace:
    """Keep original metric names so historical runs work without backfilling."""
    title, knowledge, stage = TASKS[task]
    # Omit the batch filter entirely when unset. A disabled, null-valued Group
    # filter is not a selector and can leave the UI with no useful choice.
    filters = [ws.Config("experiment.data.knowledge") == knowledge]
    if group is not None:
        filters.insert(0, ws.Metric("Group") == group)
    if stage:
        filters.append(ws.Config("experiment.train.stage") == stage)
    metrics = (["walk_loss", "lr"] if task == "fixed_knowledge" else [
        "answer_loss", "trace_loss", "free/overall/answer_accuracy",
        "free/trace/search_complete_rate",
    ])
    if task == "fixed_navigation":
        metrics.insert(2, "walk_loss")
    charts = []
    for metric in metrics:
        # Knowledge has one common warm-up per seed, regardless of protocol.
        grouping = {} if task == "fixed_knowledge" else {
            "aggregate": True, "groupby_aggfunc": "mean",
            "groupby_rangefunc": "stddev",
        }
        # Context graphs now have several training-set sizes. Inherit the
        # table's batch/size/protocol groups rather than averaging all sizes
        # (and all batches) into four protocol-only curves.
        # Inherit all table groups, including variant and graph shape. Otherwise
        # curriculum/direct or wide/deep runs would be averaged into one curve.
        charts.append(panels.LinePlot(
            title=TITLES[metric], x="optimizer_step", y=[metric],
            title_x="Optimizer step", max_runs_to_show=100,
            smoothing_type="none", **grouping,
        ))
    groupby = [] if task == "fixed_knowledge" else [
        ws.Config("experiment.data.shapes.balanced.layers"),
        ws.Config("experiment.tracking.variant"), ws.Config("experiment.protocol.name")]
    columns = []
    if task == "context_navigation":
        groupby = [ws.Metric("Group"), ws.Config("experiment.data.splits.train"),
                   ws.Config("experiment.data.shapes.balanced.layers"),
                   ws.Config("experiment.tracking.variant"), ws.Config("experiment.protocol.name")]
        columns = ["run:displayName", "run:group", "config:experiment.value.data.splits.train",
                   "config:experiment.value.tracking.variant", "config:experiment.value.protocol.name"]
    elif group is None:
        groupby.insert(0, ws.Metric("Group"))
    return ws.Workspace(
        entity=entity, project=project, name=title, auto_generate_panels=False,
        settings=ws.WorkspaceSettings(x_axis="optimizer_step", max_runs=100),
        runset_settings=ws.RunsetSettings(
            filters=ws.And(*filters), groupby=groupby, pinned_columns=columns,
        ),
        sections=[ws.Section(
            name=title, panels=charts, is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=3),
        )],
    )


def read_views(api, entity: str, project: str) -> list[dict]:
    # Query the raw spec as well: personal workspace names end in "-w" and are
    # not round-tripped by Workspace.from_url in SDK 0.4.13 (which expects -v).
    query = '''query Views($entityName: String, $name: String) {
      project(name: $name, entityName: $entityName) {
        allViews(viewType: "project-view") {
          edges { node { id name displayName spec } }
        }
      }
    }'''
    result = _graphql.execute_graphql(api, query, {"entityName": entity, "name": project})
    return [edge["node"] for edge in result["project"]["allViews"]["edges"]]


def load_workspace(node: dict, entity: str, project: str) -> ws.Workspace:
    return ws.Workspace._from_model(internal.View(
        entity=entity, project=project, id=node["id"], name=node["name"],
        display_name=node["displayName"],
        spec=internal.WorkspaceViewspec.model_validate_json(node["spec"]),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="yangshuo1230-sjtu-hpc-center")
    parser.add_argument("--project", default="cot-graph-reasoning")
    parser.add_argument("--group", help="Optional batch filter; omitted shows selectable batch groups")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS),
                        help="Views to update; other views remain untouched")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--replace-context-view", help="Existing internal view name to tidy as the context view")
    parser.add_argument("--backup-dir", type=Path, default=Path("outputs/wandb_layout"))
    args = parser.parse_args()
    desired = {task: build_workspace(args.entity, args.project, args.group, task) for task in args.tasks}
    # Validate SDK serialization before any network mutation.
    for task, workspace in desired.items():
        workspace._to_model()
        print(f"{task}: {sum(len(s.panels) for s in workspace.sections)} charts; batch={args.group or 'all groups'}")
    if not args.apply:
        print("Preview only. Pass --apply to save; no run or training job is created.")
        return
    api = wandb.Api(timeout=30)
    before = read_views(api, args.entity, args.project)
    if args.replace_context_view and not any(n["name"] == args.replace_context_view for n in before):
        raise ValueError("Requested context view was not found; no views changed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    (args.backup_dir / f"{stamp}.before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    receipts = {}
    for task, workspace in desired.items():
        matches = [n for n in before if n["displayName"] == workspace.name]
        if task == "context_navigation" and args.replace_context_view:
            matches = [n for n in before if n["name"] == args.replace_context_view]
        if len(matches) > 1:
            raise ValueError(f"Duplicate view names: {workspace.name}")
        if matches:
            old = load_workspace(matches[0], args.entity, args.project)
            workspace._internal_name = old._internal_name
            workspace._internal_id = old._internal_id
            workspace._internal_runset_id = old._internal_runset_id
        workspace.save()
        receipts[task] = {"url": workspace.url, "name": workspace._internal_name,
                          "panels": sum(len(s.panels) for s in workspace.sections)}
    # Verify saved cloud specifications, not just the in-memory objects.
    after = {n["name"]: n for n in read_views(api, args.entity, args.project)}
    for task, receipt in receipts.items():
        saved = load_workspace(after[receipt["name"]], args.entity, args.project)
        assert saved.auto_generate_panels is False
        assert sum(len(s.panels) for s in saved.sections) == receipt["panels"]
        assert saved.runset_settings.filters == ws.Workspace._from_model(desired[task]._to_model()).runset_settings.filters
    (args.backup_dir / f"{stamp}.after.json").write_text(
        json.dumps(receipts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
