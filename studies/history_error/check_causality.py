"""Small in-memory causal/batching check; no results or checkpoints saved."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["WANDB_MODE"] = "disabled"
from pathlib import Path
import json
import torch
from run import build_events
from cot_navigation.analysis.scale_followup import make_adapter
torch.set_num_threads(2)
root = Path.cwd()
entry = next(e for e in json.loads((root / "studies/scale_mechanistic/results/selection.json").read_text())["models"] if e["id"] == "context_l10_dfs_state")
am = make_adapter(entry, root, "cuda:0")
events, _, _ = build_events(am, "train", 8, 512)
for e in events:
    with torch.inference_mode():
        _, hs, _ = am.forward([e["correct"], e["wrong"]])
        p = e["error_position"] - 1
        diffs = [(h[0, p].float() - h[1, p].float()).abs().max().item() for h in hs]
        print(e["query"], "same_batch_pre_error", diffs, flush=True)
        assert max(diffs) < 1e-4
