from __future__ import annotations
import time
import torch
import pytorch_lightning as pl


class ThroughputCallback(pl.Callback):
    """EMA of training throughput.

    Unit depends on `pl_module.task`:
    - lm / lm_large / cls : tokens/sec (counts batch["input_ids"].numel())
    - vision_cnn / vision_vit : images/sec (counts batch[0].size(0))
    """

    def __init__(self, ema_decay: float = 0.9):
        self.ema_decay = ema_decay
        self._ema: float | None = None
        self._last_time: float | None = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._last_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._last_time is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = max(time.time() - self._last_time, 1e-6)

        task = getattr(pl_module, "task", "lm")
        if task in ("vision_cnn", "vision_vit"):
            count = batch[0].size(0)
            metric_name = "throughput_images_per_sec"
        else:
            # text-based tasks (lm, lm_large, cls)
            count = batch["input_ids"].numel()
            metric_name = "throughput_tokens_per_sec"

        current = count / elapsed
        self._ema = current if self._ema is None else self.ema_decay * self._ema + (1 - self.ema_decay) * current
        pl_module.log(metric_name, self._ema, prog_bar=True)
