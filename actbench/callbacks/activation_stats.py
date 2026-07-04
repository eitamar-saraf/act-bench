from __future__ import annotations
import json
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn

from actbench.models.activations import Mish, ScaledTanh
from actbench.models.gated_ffn import SwiGLU


_DEFAULT_ACTIVATION_NAME_FRAGMENTS = (
    'relu',        # nn.ReLU, nn.LeakyReLU
    'leakyrelu',   # explicit, redundant with 'relu' but harmless
    'tanh',        # nn.Tanh AND ScaledTanh — substring match on lowercased class name
    'gelu',
    'silu',
    'mish',        # Mish
    'swiglu',      # SwiGLU (gated FFN — we record the FFN output moments)
)


def _post_moments(name: str, data: torch.Tensor, update_fn):
    mean = data.mean().item()
    std = data.std(unbiased=False).item() + 1e-8
    normed = (data - mean) / std
    update_fn(name, 'mean', mean)
    update_fn(name, 'std', std)
    update_fn(name, 'skew', (normed ** 3).mean().item())
    update_fn(name, 'kurtosis', (normed ** 4).mean().item())


class ActivationStatsCallback(pl.Callback):
    """Per-layer activation health stats, dumped per epoch as JSON.

    For each activation family we record the diagnostic that's most informative:
    - ReLU / LeakyReLU      → dead_fraction (post <= 0)
    - Tanh / ScaledTanh     → tanh_saturation_pre (|x|>4), tanh_saturation_post (|y|>0.99 for tanh, normalized for ScaledTanh), post moments
    - GELU / SiLU / Mish    → post moments (mean / std / skew / kurtosis)

    Aggregation: running sum + count per (layer, stat) → mean at log time.
    """

    def __init__(
        self,
        activations: tuple[str, ...] = _DEFAULT_ACTIVATION_NAME_FRAGMENTS,
        log_every_n_steps: int | None = None,
        json_export: bool = True,
    ):
        self.activations = activations
        self.log_every_n_steps = log_every_n_steps
        self.json_export = json_export
        self._hooks: list = []
        self._acc: dict = {}
        self._global_step_last_log = 0

    def _update(self, layer: str, stat: str, value: float):
        layer_acc = self._acc.setdefault(layer, {})
        slot = layer_acc.setdefault(stat, {'sum': 0.0, 'count': 0})
        slot['sum'] += float(value)
        slot['count'] += 1

    def setup(self, trainer, pl_module, stage: str):
        if stage != 'fit':
            return

        def make_hook(layer_name: str):
            def _hook(module, inp, out):
                # Skip when no input tensor requires grad — avoids backward-hook warning during eval.
                if not any(isinstance(t, torch.Tensor) and t.requires_grad for t in inp):
                    return
                data = out.detach().float().view(-1)
                if data.numel() == 0:
                    return

                if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                    self._update(layer_name, 'dead_fraction', (data <= 0).float().mean().item())
                    _post_moments(layer_name, data, self._update)

                if isinstance(module, nn.Tanh):
                    pre = inp[0].detach().float().view(-1)
                    self._update(layer_name, 'tanh_saturation_pre', (pre.abs() > 4).float().mean().item())
                    self._update(layer_name, 'tanh_saturation_post', (data.abs() > 0.99).float().mean().item())
                    _post_moments(layer_name, data, self._update)

                if isinstance(module, ScaledTanh):
                    pre = inp[0].detach().float().view(-1)
                    self._update(layer_name, 'tanh_saturation_pre', (pre.abs() > 4).float().mean().item())
                    # Normalize the |y| threshold by the scale so it stays comparable to plain tanh.
                    threshold = 0.99 * module.scale
                    self._update(layer_name, 'tanh_saturation_post', (data.abs() > threshold).float().mean().item())
                    _post_moments(layer_name, data, self._update)

                if isinstance(module, (nn.GELU, nn.SiLU, Mish)):
                    _post_moments(layer_name, data, self._update)

                if isinstance(module, SwiGLU):
                    _post_moments(layer_name, data, self._update)

            return _hook

        for name, module in pl_module.model.named_modules():
            lname = type(module).__name__.lower()
            if any(fragment in lname for fragment in self.activations):
                self._hooks.append(module.register_forward_hook(make_hook(name)))

    def _log_now(self, trainer, pl_module, on_epoch: bool = False):
        for layer, stats in self._acc.items():
            for stat, agg in stats.items():
                if agg['count'] == 0:
                    continue
                value = agg['sum'] / agg['count']
                pl_module.log(f'act/{layer}/{stat}', value, sync_dist=True, on_step=False, on_epoch=on_epoch)
        self._global_step_last_log = pl_module.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.log_every_n_steps and self.log_every_n_steps > 0:
            if (pl_module.global_step - self._global_step_last_log) >= self.log_every_n_steps:
                self._log_now(trainer, pl_module, on_epoch=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._log_now(trainer, pl_module, on_epoch=True)
        if self.json_export and trainer.log_dir:
            export = {
                layer: {stat: (agg['sum'] / max(1, agg['count'])) for stat, agg in stats.items()}
                for layer, stats in self._acc.items()
            }
            out_dir = os.path.join(trainer.log_dir, 'activation_stats')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f'epoch_{pl_module.current_epoch:03d}.json'), 'w') as f:
                json.dump(export, f, indent=2)
