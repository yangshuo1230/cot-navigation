"""Local metrics plus an optional W&B run.

Online mode never pretends to have uploaded. Missing credentials follow
``tracking.online_failure``: raise, or retry once in offline mode and record
that downgrade. Secrets are not copied into ``wandb.config`` or the jsonl file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cot_navigation.config import AppConfig, to_plain


def _scrub(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(word in lowered for word in ("secret", "password", "credential", "api_key")):
                continue
            cleaned[key] = _scrub(item)
        return cleaned
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


class Tracker:
    def __init__(self, cfg: AppConfig, run_id: str, parent_run_id: str | None, env: dict, *, resume: bool = False):
        self.cfg = cfg
        self.persist = cfg.train.persist_outputs and not cfg.runtime.smoke
        self.path = Path(cfg.output_dir) / "metrics.jsonl"
        if self.persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = None
        self.closed = False
        self.mode = cfg.tracking.mode if self.persist else "disabled"
        rank = os.environ.get("RANK", "0")
        if rank not in {"0", ""}:
            self.mode = "disabled"
        if self.mode == "disabled":
            return
        import wandb

        requested = self.mode
        if requested == "online" and not _has_wandb_credentials(wandb):
            if cfg.tracking.online_failure == "offline":
                self.mode = "offline"
            else:
                raise RuntimeError(
                    "W&B mode is online but no credentials were found. "
                    "Set WANDB_API_KEY or choose tracking.mode=offline. Nothing was uploaded."
                )
        os.environ["WANDB_DIR"] = str(Path(cfg.output_dir))
        os.environ["WANDB_SILENT"] = "true"
        try:
            self.run = wandb.init(
                project=cfg.tracking.project,
                entity=cfg.tracking.entity or os.environ.get("WANDB_ENTITY") or None,
                name=cfg.tracking.name,
                group=cfg.tracking.group,
                tags=list(cfg.tracking.tags),
                notes=cfg.tracking.notes,
                id=run_id,
                resume=cfg.tracking.resume if resume else "never",
                mode="offline" if self.mode == "offline" else "online",
                dir=str(Path(cfg.output_dir)),
                config=_scrub({"experiment": to_plain(cfg), "env": _public_env(env), "parent_run_id": parent_run_id}),
                reinit=True,
            )
        except Exception as exc:
            if requested == "online" and cfg.tracking.online_failure == "offline":
                self.mode = "offline"
                self.run = wandb.init(
                    project=cfg.tracking.project,
                    entity=cfg.tracking.entity,
                    name=cfg.tracking.name,
                    id=run_id,
                    resume=cfg.tracking.resume if resume else "never",
                    mode="offline",
                    config={"parent_run_id": parent_run_id, "online_error": str(exc)},
                    reinit=True,
                )
            else:
                raise
        if requested == "online" and getattr(self.run, "mode", "online") != "online":
            if cfg.tracking.online_failure == "error":
                raise RuntimeError(
                    f"W&B stayed in mode {self.run.mode} after an online init. Refusing to claim an upload."
                )
        # The SDK owns its monotonically increasing history step. Several log
        # events may share one optimizer update without overwriting each other.
        self.run.define_metric("optimizer_step")
        self.run.define_metric("*", step_metric="optimizer_step")
        (Path(cfg.output_dir) / "wandb_status.json").write_text(
            json.dumps(
                {
                    "requested": requested,
                    "effective": self.mode,
                    "run_id": run_id,
                    "project": cfg.tracking.project,
                    "url": getattr(self.run, "url", None),
                    "parent_run_id": parent_run_id,
                },
                indent=2,
            )
            + "\n"
        )

    def log(self, step: int, metrics: dict) -> None:
        row = {"step": int(step), "optimizer_step": int(step)}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                continue
            row[key] = value
        if self.persist:
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        if self.run is not None:
            self.run.log(row, commit=True)

    def finish(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.run is not None:
            if self.cfg.tracking.upload_artifacts:
                import wandb

                artifact = wandb.Artifact(f"checkpoint-{self.run.id}", type="model")
                files = [self.path.parent / "resolved_config.yaml", self.path.parent / "env.json",
                         self.path.parent / "checkpoints/best.pt", self.path.parent / "checkpoints/latest.pt"]
                for path in files:
                    if path.is_file():
                        artifact.add_file(str(path), name=str(path.relative_to(self.path.parent)))
                if any(p.is_file() for p in files):
                    self.run.log_artifact(artifact)
            self.run.finish()


def _has_wandb_credentials(wandb) -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        key = wandb.api.api_key
    except Exception:
        return False
    return bool(key)


def _public_env(env: dict) -> dict:
    keep = ("python", "torch", "cuda", "gpu", "accelerator", "git", "optimizer", "parameters", "sdpa", "code_fingerprint", "triton", "torch_compile")
    return {key: env[key] for key in keep if key in env}
