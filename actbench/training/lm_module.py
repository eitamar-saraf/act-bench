from __future__ import annotations
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .scheduler import WarmupCosineAnnealingLR


class LanguageModelingModule(pl.LightningModule):
    """Causal language modeling trainer for the activation comparison study."""

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.01,
        warmup_steps: int = 200,
        max_steps: int = 10000,
        task: str = "lm",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        # `task` is what the throughput callback / collect script branch on.
        self.task = task

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids)

    def _next_token_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(token_ids)
        # Shift: predict token t+1 from positions [0..t]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = token_ids[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
        )

    def training_step(self, batch, batch_idx):
        loss = self._next_token_loss(batch["input_ids"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ppl", torch.exp(loss.detach()), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._next_token_loss(batch["input_ids"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_ppl", torch.exp(loss.detach()), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.adamw_betas,
            eps=self.hparams.adamw_eps,
            weight_decay=self.hparams.adamw_weight_decay,
        )
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=self.hparams.warmup_steps,
            max_steps=self.hparams.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
