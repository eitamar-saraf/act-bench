"""Lightning module for multiclass classification (text or image).

Used for AG News text classification, CIFAR-10 ResNet, and CIFAR-10 ViT.
All three reduce to: forward(x) -> logits, cross-entropy against integer labels.
"""
from __future__ import annotations
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score

from .scheduler import WarmupCosineAnnealingLR


class ClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        task: str,
        learning_rate: float = 1e-3,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.task = task
        self.num_classes = num_classes

        self.train_top1 = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_top1 = Accuracy(task="multiclass", num_classes=num_classes)
        # Macro F1 only meaningful when classes might be imbalanced (cls). For
        # CIFAR-10 it's redundant with top-1 but cheap, so we keep it everywhere.
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, inputs):
        return self.model(inputs)

    def _unpack(self, batch):
        if isinstance(batch, dict):
            # text: {input_ids, labels}
            return batch["input_ids"], batch["labels"]
        # vision: (images, labels)
        return batch[0], batch[1]

    def training_step(self, batch, batch_idx):
        inputs, labels = self._unpack(batch)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_top1.update(logits, labels)
        self.log("train_top1", self.train_top1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = self._unpack(batch)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.val_top1.update(logits, labels)
        self.val_f1.update(logits, labels)
        self.log("val_top1", self.val_top1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, sync_dist=True)
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
