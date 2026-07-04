from __future__ import annotations
import pytorch_lightning as pl, torch

class PeakMemoryCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated()/1e6
            pl_module.log('peak_gpu_mem_mb', mem, on_step=True, sync_dist=True)
            torch.cuda.reset_peak_memory_stats()