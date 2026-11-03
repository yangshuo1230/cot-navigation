"""Host and kernel probe. This does not mutate drivers or other users' processes."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import hashlib
from pathlib import Path


def _git(root: Path) -> dict:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = run(["git", "rev-parse", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None, "dirty_entries": None if dirty is None else len(dirty.splitlines())}


def probe_environment(probe_kernels: bool = False) -> dict:
    import torch

    root = Path(__file__).resolve().parents[3]
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    disk = shutil.disk_usage(root)
    info = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(mem / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        },
        "accelerator": {},
        "git": _git(root),
        "code_fingerprint": _source_fingerprint(root),
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:
        info["triton"] = f"unavailable: {exc.__class__.__name__}"
    info["torch_compile"] = bool(hasattr(torch, "compile"))
    if torch.cuda.is_available():
        info["gpu"] = []
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            info["gpu"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "free_mb": int(free / (1024**2)),
                    "total_mb": int(total / (1024**2)),
                }
            )
        info["accelerator"] = {"sdpa": probe_sdpa() if probe_kernels else "not_probed"}
    return info


def probe_sdpa() -> dict:
    import torch
    import torch.nn.functional as F

    results = {}
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        return {"error": str(exc)}
    # Environment inspection must not advance training RNG, including on resume.
    generator = torch.Generator(device="cuda").manual_seed(0)
    query = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(query, query, query, is_causal=True)
            torch.cuda.synchronize()
            results[backend.name] = "ok"
        except Exception as exc:
            results[backend.name] = exc.__class__.__name__ + ": " + str(exc).splitlines()[0][:180]
    return results


def _source_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted((root / "src/cot_navigation").rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def nvidia_smi_gpus() -> list[dict]:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,utilization.gpu,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows = []
    for line in text.strip().splitlines():
        index, used, free, util, uuid = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "memory_used_mb": float(used),
                "memory_free_mb": float(free),
                "util": float(util),
                "uuid": uuid,
            }
        )
    return rows


def compute_apps() -> list[dict]:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    apps = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, used = [part.strip() for part in line.split(",")]
        apps.append({"uuid": uuid, "pid": int(float(pid)), "used_mb": float(used)})
    return apps
