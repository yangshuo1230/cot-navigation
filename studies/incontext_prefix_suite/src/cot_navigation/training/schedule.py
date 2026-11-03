"""Warmup then cosine decay. ``step`` is the number of completed optimizer updates."""

from __future__ import annotations

import math


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, lr: float, min_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.base_lr = float(lr)
        self.min_lr = float(min_lr)
        self.updates = 0

    def lr_at(self, updates: int) -> float:
        if self.warmup_steps > 0 and updates < self.warmup_steps:
            return self.base_lr * float(updates + 1) / float(self.warmup_steps)
        span = max(1, self.max_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (updates - self.warmup_steps) / span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def set_lr(self) -> float:
        lr = self.lr_at(self.updates)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def step(self) -> float:
        """Call after ``optimizer.step``. The next update uses the new learning rate."""
        self.updates += 1
        return self.set_lr()

    def state_dict(self) -> dict:
        return {
            "updates": self.updates,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.updates = int(state["updates"])
        # Increasing training max_steps must not restart a decayed schedule.
        # Once its original horizon is reached, extension uses the saved floor.
        self.warmup_steps = int(state["warmup_steps"])
        self.max_steps = int(state["max_steps"])
        self.base_lr = float(state["base_lr"])
        self.min_lr = float(state["min_lr"])
        self.set_lr()
