"""Eager versus optimized step benchmark.

Both sides consume the same token batch. Timing starts after CUDA synchronization
and a warmup. Filler prefill is not reported as autoregressive decode.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from cot_navigation.config import AppConfig
from cot_navigation.models.transformer import build_model
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.runtime.envcheck import compute_apps, nvidia_smi_gpus, probe_environment, probe_sdpa
from cot_navigation.training.loss import combine_losses


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pick_device(cfg: AppConfig) -> torch.device:
    if not torch.cuda.is_available() or cfg.runtime.device == "cpu":
        return torch.device("cpu")
    info = probe_environment(probe_kernels=False)
    gpus = info.get("gpu") or []
    if not gpus:
        return torch.device("cpu")
    best = max(gpus, key=lambda row: row["free_mb"])
    if best["free_mb"] < 8000:
        return torch.device("cpu")
    return torch.device(f"cuda:{best['index']}")


def _make_batch(cfg: AppConfig, device: torch.device, length: int, batch: int, vocab: int):
    tokens = torch.randint(1, vocab, (batch, length), device=device)
    answer = torch.zeros(batch, length, device=device)
    trace = torch.zeros(batch, length, device=device)
    # Supervise a late answer token and a short trace, and leave the tail as padding-like zeros.
    answer[:, length // 2] = 1
    trace[:, 1 : length // 2] = 1
    return {
        "tokens": tokens,
        "answer_mask": answer,
        "trace_mask": trace,
        "walk_mask": torch.zeros(batch, length, device=device),
        "lengths": torch.full((batch,), length, device=device, dtype=torch.long),
    }


def _steps(model, batch, optimizer, device, precision: str, n: int, n_task: int) -> None:
    for _ in range(n):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16" and device.type == "cuda"):
            logits = model(batch["tokens"])
        parts = combine_losses(logits, batch["tokens"], batch["answer_mask"], batch["trace_mask"], batch["walk_mask"], n_task, 0)
        parts["total_loss"].backward()
        optimizer.step()


def _time_mode(cfg: AppConfig, device: torch.device, mode: str, batch_size: int, length: int, warmup: int, steps: int) -> dict:
    tok = Tokenizer(cfg.data.node_vocab)
    model_cfg = cfg.model
    compile_enabled = mode == "optimized"
    attn = "sdpa" if mode == "optimized" else "math"
    model = build_model(model_cfg, len(tok), tok.pad)
    model.attn_impl = attn
    for block in model.blocks:
        block.attn.attn_impl = attn
    model.to(device)
    if compile_enabled:
        if device.type != "cuda":
            return {"mode": mode, "skipped": "compile requires CUDA"}
        model = torch.compile(model, mode=cfg.model.compile_mode)
    precision = "bf16" if mode == "optimized" and device.type == "cuda" else "fp32"
    fused = mode == "optimized" and device.type == "cuda"
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, fused=fused)
        opt_impl = "fused_adamw" if fused else "adamw"
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, foreach=device.type == "cuda")
        opt_impl = "foreach_adamw"
    batch = _make_batch(cfg, device, length, batch_size, len(tok))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    _sync(device)
    cold_t0 = time.perf_counter()
    _steps(model, batch, opt, device, precision, 1, batch_size)
    _sync(device)
    cold_s = time.perf_counter() - cold_t0
    _steps(model, batch, opt, device, precision, warmup, batch_size)
    _sync(device)
    t0 = time.perf_counter()
    _steps(model, batch, opt, device, precision, steps, batch_size)
    _sync(device)
    elapsed = time.perf_counter() - t0
    tokens = batch_size * length
    return {
        "mode": mode,
        "device": str(device),
        "precision": precision,
        "attn_impl": attn,
        "optimizer": opt_impl,
        "compile": compile_enabled,
        "batch": batch_size,
        "seq_len": length,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "cold_start_s": cold_s,
        "step_ms": (elapsed / steps) * 1000,
        "queries_per_s": batch_size * steps / elapsed,
        "tokens_per_s": tokens * steps / elapsed,
        "effective_tokens_per_s": tokens * steps / elapsed,
        "padding_tokens_per_s": 0.0,
        "peak_mem_mb": (torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else None,
        "note": "Repeated fixed batch. Padding is zero. Not comparable to a shorter sequence.",
        "attention_kernels": _attention_kernels(model, batch, device, precision) if mode == "optimized" and device.type == "cuda" else [],
    }


def _attention_kernels(model, batch, device: torch.device, precision: str) -> list[str]:
    """Names of CUDA ops from one extra forward. Not included in the timed steps."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            model(batch["tokens"])
        _sync(device)
    found = []
    for evt in prof.key_averages():
        low = evt.key.lower()
        if any(word in low for word in ("fmha", "flash", "mem_efficient", "efficient_attention", "cudnn", "scaled_dot", "triton")):
            found.append(evt.key)
    return found[:40]


def run_benchmark(cfg: AppConfig, batch_size: int = 8, length: int = 256, warmup: int = 3, steps: int = 10) -> dict:
    device = _pick_device(cfg)
    data_t0 = time.perf_counter()
    # The synthetic batch is built inside each mode; this measures host setup only.
    data_s = time.perf_counter() - data_t0
    report = {
        "env": probe_environment(probe_kernels=False),
        "sdpa": probe_sdpa() if device.type == "cuda" else "cpu",
        "data_setup_s": data_s,
        "shared_gpu": False,
        "modes": [],
    }
    if device.type == "cuda":
        index = device.index or 0
        free, total = torch.cuda.mem_get_info(index)
        # Other jobs may already occupy the card. We only proceed when several GB are free.
        report["selected_gpu"] = {"index": index, "free_mb": int(free / (1024**2)), "total_mb": int(total / (1024**2))}
        uuid = next((row["uuid"] for row in nvidia_smi_gpus() if row["index"] == index), None)
        others = [app["pid"] for app in compute_apps() if app["uuid"] == uuid and app["pid"] != os.getpid()]
        report["shared_gpu"] = bool(others)
        report["other_compute_pids"] = others
        report["data_setup_note"] = "data_setup_s is host setup only. The synthetic batch is built inside each mode, so this is not a dataloader measurement."
    report["modes"].append(_time_mode(cfg, device, "eager", batch_size, length, warmup, steps))
    if device.type == "cuda":
        report["modes"].append(_time_mode(cfg, device, "optimized", batch_size, length, warmup, steps))
    else:
        report["gpu_verified"] = False
        report["note"] = "CUDA was not used. No accelerator numbers were invented."
    if cfg.train.persist_outputs and not cfg.runtime.smoke:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report
