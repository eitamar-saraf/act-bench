from __future__ import annotations
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from .scheduler import WarmupCosineAnnealingLR
from torchmetrics import Accuracy, F1Score, CalibrationError


class ActivationBenchmarkModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, task: str, learning_rate: float = 1e-3, adamw_betas: tuple[float, float] = (0.9, 0.999), adamw_eps: float = 1e-8, adamw_weight_decay: float = 0.01, warmup_steps: int = 500, max_steps: int = 10000):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.task = task
        
        if task == "vision":
            self.top1 = Accuracy(task='multiclass', num_classes=200)
            self.top5 = Accuracy(task='multiclass', num_classes=200, top_k=5)
            self.f1_macro = F1Score(task='multiclass', num_classes=200, average='macro')
            self.ece = CalibrationError(task='multiclass', num_classes=200, n_bins=15, norm='l1')
        elif task == "cls":
            self.cls_acc = Accuracy(task='multiclass', num_classes=4)
            self.f1_macro = F1Score(task='multiclass', num_classes=4, average='macro')
            self.ece = CalibrationError(task='multiclass', num_classes=4, n_bins=15, norm='l1')
            
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def _forward_and_loss(self, batch):
        
        if self.task == "vision":
            inputs, labels = batch
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, labels)
            return loss, logits, labels
        
        if self.task == "cls":
            inputs, labels = batch["input_ids"], batch["labels"]
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, labels)
            return loss, logits, labels
        
        if self.task == "lm":
            input_ids = batch["input_ids"]
            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            return loss, logits, input_ids
        
        raise ValueError(f"Unknown task: {self.task}")
    
    
    def training_step(self, batch, batch_idx):
        loss, logits, targets = self._forward_and_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)
        
        if self.task == "vision":
            self.log("train_top1", self.top1(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_top5", self.top5(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_f1_macro", self.f1_macro(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_ece", self.ece(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
        
        elif self.task == "cls":
            self.log("train_acc", self.cls_acc(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_f1_macro", self.f1_macro(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_ece", self.ece(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
        
        elif self.task == "lm":
            ppl = torch.exp(loss.detach())
            self.log("train_ppl", ppl, on_step=False, on_epoch=True, sync_dist=True)
            
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self._forward_and_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.task == "vision":
            self.log("val_top1", self.top1(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_top5", self.top5(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_f1_macro", self.f1_macro(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_ece", self.ece(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
        
        elif self.task == "cls":
            self.log("val_acc", self.cls_acc(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_f1_macro", self.f1_macro(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_ece", self.ece(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
        
        elif self.task == "lm":
            ppl = torch.exp(loss.detach())
            self.log("val_ppl", ppl, on_step=False, on_epoch=True, sync_dist=True)
        # collect reliability diagram data if requested
        if getattr(self, 'collect_reliability', False) and self.task in ("vision","cls"):
            if not hasattr(self, '_rel_logits'):
                self._rel_logits, self._rel_targets = [], []
            self._rel_logits.append(logits.detach().cpu())
            self._rel_targets.append(targets.detach().cpu())
        return loss
    
    
    def test_step(self, batch, batch_idx):
        loss, logits, targets = self._forward_and_loss(batch)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.task == "vision":
            self.log("test_top1", self.top1(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_top5", self.top5(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_f1_macro", self.f1_macro(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_ece", self.ece(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
        
        elif self.task == "cls":
            self.log("test_acc", self.cls_acc(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_f1_macro", self.f1_macro(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_ece", self.ece(logits, targets), on_step=False, on_epoch=True, sync_dist=True)
        
        elif self.task == "lm":
            ppl = torch.exp(loss.detach())
            self.log("test_ppl", ppl, on_step=False, on_epoch=True, sync_dist=True)
        if getattr(self, 'collect_reliability', False) and self.task in ("vision","cls"):
            if not hasattr(self, '_rel_logits_test'):
                self._rel_logits_test, self._rel_targets_test = [], []
            self._rel_logits_test.append(logits.detach().cpu())
            self._rel_targets_test.append(targets.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        # Only rank 0 writes diagram to avoid duplicates
        if getattr(self, 'collect_reliability', False) and self.global_rank == 0 and hasattr(self, '_rel_logits'):
            from actbench.analysis.reliability import compute_reliability, plot_reliability
            logits = torch.cat(self._rel_logits, dim=0)
            targets = torch.cat(self._rel_targets, dim=0)
            diag = compute_reliability(logits, targets, n_bins=15)
            log_dir = self.trainer.log_dir if self.trainer else 'logs'
            out_json = log_dir + '/reliability_val_epoch%03d.json' % self.current_epoch
            with open(out_json,'w') as f:
                f.write(diag.to_json())
            try:
                plot_reliability(diag, out_png=log_dir + '/reliability_val_epoch%03d.png' % self.current_epoch)
            except Exception:
                pass
            del self._rel_logits, self._rel_targets
    
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, betas=self.hparams.adamw_betas, eps=self.hparams.adamw_eps, weight_decay=self.hparams.adamw_weight_decay)
        sched = WarmupCosineAnnealingLR(optimizer, warmup_steps=self.hparams.warmup_steps, max_steps=self.hparams.max_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1
                    }
                }
