from __future__ import annotations
import pytorch_lightning as pl, torch, os, json

class GradientStatsCallback(pl.Callback):
    def __init__(self, log_every_n_steps: int | None = None, json_export: bool = True):
        self.log_every_n_steps = log_every_n_steps
        self.json_export = json_export
        self._acc = {}  # layer -> stat -> {sum,count}
        self._last_log_step = 0
    def _update(self, layer, stat, value):
        lay = self._acc.setdefault(layer, {})
        slot = lay.setdefault(stat, {'sum':0.0,'count':0})
        slot['sum'] += float(value)
        slot['count'] += 1
    def setup(self, trainer, pl_module, stage: str):
        if stage != 'fit': return
        def make_hook(name):
            def _hook(mod, gin, gout):
                # accumulate gradient norms AFTER backward
                total = 0.0
                rel = None
                for p in mod.parameters(recurse=False):
                    if p.grad is not None:
                        gnorm = p.grad.norm().item()
                        total += gnorm**2
                        if rel is None:
                            rel = gnorm / (p.data.norm().item() + 1e-8)
                total = total**0.5
                if total>0:
                    self._update(name,'grad_l2', total)
                if rel is not None:
                    self._update(name,'rel_update', rel)
            return _hook
        for name, module in pl_module.model.named_modules():
            if len(list(module.parameters(recurse=False)))>0:
                module.register_full_backward_hook(make_hook(name))
    def _log_now(self, pl_module, on_epoch=False):
        for layer, stats in self._acc.items():
            for stat, agg in stats.items():
                if agg['count']==0: continue
                val = agg['sum']/agg['count']
                pl_module.log(f'grad/{layer}/{stat}', val, sync_dist=True, on_step=False, on_epoch=on_epoch)
        self._last_log_step = pl_module.global_step
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.log_every_n_steps and (pl_module.global_step - self._last_log_step) >= self.log_every_n_steps:
            self._log_now(pl_module, on_epoch=False)
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking: return
        self._log_now(pl_module, on_epoch=True)
        if self.json_export and trainer.log_dir:
            export = {layer: {stat: (agg['sum']/max(1,agg['count'])) for stat,agg in stats.items()} for layer, stats in self._acc.items()}
            out_dir = os.path.join(trainer.log_dir, 'gradient_stats')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f'epoch_{pl_module.current_epoch:03d}.json'),'w') as f:
                json.dump(export, f, indent=2)