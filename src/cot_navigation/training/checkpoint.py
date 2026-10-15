"""Atomic checkpoints.

A checkpoint stores the uncompiled parameter names, optimizer, schedule step,
Python/NumPy/Torch/CUDA RNG, sampler cursors, config, and manifest hashes.
``latest`` is replaced atomically. Periodic files beyond ``keep_last`` are removed.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.writing"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_training_checkpoint(directory: Path, payload: dict, keep_last: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    step = int(payload["step"])
    periodic = directory / f"step_{step:08d}.pt"
    atomic_torch_save(periodic, payload)
    latest = directory / "latest.pt"
    atomic_torch_save(latest, payload)
    periodic_files = sorted(directory.glob("step_*.pt"))
    if keep_last >= 0 and len(periodic_files) > keep_last:
        for old in periodic_files[: len(periodic_files) - keep_last]:
            old.unlink(missing_ok=True)
    return latest


def save_best(directory: Path, payload: dict) -> None:
    atomic_torch_save(directory / "best.pt", payload)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
