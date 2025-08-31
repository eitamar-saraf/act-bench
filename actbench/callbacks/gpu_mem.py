import pytorch_lightning as pl
import torch

class GPUMemoryCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e6
            pl_module.log("peak_gpu_mem_mb", mem, sync_dist=True)
            torch.cuda.reset_peak_memory_stats()