from __future__ import annotations
import time, torch
import pytorch_lightning as pl

class ThroughputCallback(pl.Callback):
    """Computes moving-average throughput (images/sec or tokens/sec) during training."""
    def __init__(self, ema_decay: float = 0.9):
        self.ema_decay = ema_decay
        self._ema = None
        self._last_time = None
        
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx: int):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._last_time = time.time()
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int):
        if self._last_time is None:
            return
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = max(time.time() - self._last_time, 1e-6)
        
        # Determine unit count
        if pl_module.task == "vision":
            batch_size = batch[0].size(0)
            units = batch_size
            metric_name = "throughput_images_per_sec"
        elif pl_module.task in ("cls", "lm"):
            input_ids = batch["input_ids"]
            units = input_ids.numel()  # tokens processed
            metric_name = "throughput_tokens_per_sec"
        else:
            return
        
        current = units / dt
        if self._ema is None:
            self._ema = current
        else:
            self._ema = self.ema_decay * self._ema + (1 - self.ema_decay) * current
            
        trainer.lightning_module.log(metric_name, self._ema, prog_bar=True, sync_dist=True)
