from __future__ import annotations
from torch.optim.lr_scheduler import _LRScheduler
import math
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps; self.max_steps = max_steps; self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.warmup_steps > 0 and self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / self.warmup_steps
            return [base_lr * progress for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        decayed = (1 - self.eta_min) * cosine_decay + self.eta_min
        return [base_lr * decayed for base_lr in self.base_lrs]
    def _get_closed_form_lr(self):
        return self.get_lr()
