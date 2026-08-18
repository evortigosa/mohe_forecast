# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Distributed Trainer Class
"""

import torch
from .Trainer import Trainer



class DTrainer(Trainer):
    """
    Trainer class with distributed-specific logic using Lightning Fabric.
    - model creation, data-loader creation, and optimizer creation must happen after fabric.launch()
    - model, optimizer, and loaders must be provided on CPU
    See https://lightning.ai/docs/fabric/stable/
    """

    def __init__(self, fabric, **kwargs,) -> None:
        super().__init__(**kwargs)
        # create the Fabric object
        self.fabric= fabric
        self._setup()


    def _set_model(self):
        pass


    def _set_optimizer(self):
        if getattr(self, "optimizer", None) is None:
            return

        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key]= self.fabric.to_device(value)


    def _set_loader(self, loader):
        if loader.__class__.__name__ == "_FabricDataLoader":   # already wrapped
            return loader
        return self.fabric.setup_dataloaders(loader, move_to_device=False)


    def _move_to_device(self, x):
        if isinstance(x, torch.Tensor):
            return self.fabric.to_device(x)
        return x


    def _setup(self):
        self.device= self.fabric.device
        # setup model and optimizer
        assert self.model is not None, "Model must be provided."
        assert self.optimizer is not None, "Optimizer must be provided."
        # keep a reference to the bare module BEFORE Fabric wraps it. Fabric/DDP wrap by reference and
        # move parameters in place, so this reference stays valid and on-device, and writing to it
        # (e.g. set_forecast_horizon -> n_outputs) is seen by the wrapped forward. More robust than
        # unwrapping via '.module', which does not resolve for single-device Fabric
        self._bare_model= self.model
        self.model, self.optimizer= self.fabric.setup(self.model, self.optimizer)
        # setup dataloaders
        if self.train_loader is not None:
            self.train_loader= self._set_loader(self.train_loader)
        if self.val_loader is not None:
            self.val_loader= self._set_loader(self.val_loader)
        if self.test_loader is not None:
            self.test_loader= self._set_loader(self.test_loader)


    def _unwrap_model(self):
        if getattr(self, "_bare_model", None) is not None:
            return self._bare_model
        return self.model.module if hasattr(self.model, "module") else self.model


    def _is_main_process(self) -> bool:
        return self.fabric.is_global_zero


    def _barrier(self):
        self.fabric.barrier()


    def _print(self, *args, **kwargs):
        self.fabric.print(*args, **kwargs)


    def _reduce_sum(self, x:torch.Tensor) -> torch.Tensor:
        return self.fabric.all_reduce(x, reduce_op="sum")


    def _gather_variable_batch(self, x:torch.Tensor) -> torch.Tensor:
        """
        Gather tensors with possibly different local batch sizes across ranks.
        Assumption:
        - the batch dimension is dim=0, and all non-batch dimensions are identical across ranks.
        Returns:
        - a tensor containing the concatenation of all valid local tensors from all ranks, on every rank.
        """
        if not hasattr(self, "fabric") or self.fabric.world_size == 1:
            # for single-GPU / non-distributed execution
            return x

        local_n= torch.tensor([x.size(0)], device=x.device, dtype=torch.long)
        # shape: [world_size, 1] -> [world_size]
        all_n= self.fabric.all_gather(local_n).reshape(-1)
        max_n= int(all_n.max().item())
        local_n_int= int(local_n.item())

        if local_n_int < max_n:
            pad_shape= list(x.shape)
            pad_shape[0]= max_n - local_n_int
            pad= torch.zeros(pad_shape, dtype=x.dtype, device=x.device,)
            x_pad= torch.cat([x, pad], dim=0)
        else:
            x_pad= x

        # shape: x_pad -> [max_n, ...]; gathered -> [world_size, max_n, ...]
        gathered= self.fabric.all_gather(x_pad)
        # if no rank dimension exists, return local valid part
        if gathered.ndim == x_pad.ndim:
            return gathered[:local_n_int]

        pieces= []
        for rank_idx, n in enumerate(all_n.tolist()):
            pieces.append(gathered[rank_idx, :int(n)])

        return torch.cat(pieces, dim=0)


    def _backward(self, loss, clip_grad=None):
        self.fabric.backward(loss)
        if clip_grad is not None:
            self.fabric.clip_gradients(self.model, self.optimizer, max_norm=clip_grad)


    def _reduce_moe_stats(self, metric_dicts):
        """
        Reduce a list of per-rank MoE-metric dicts across ranks and recompute the derived quantities
        from the global sufficient statistics. All dicts are packed into a single tensor so the whole
        list is reduced with one all-reduce (one sync point), regardless of how many MoE layers there are.
        - Why recompute instead of averaging: hard_fraction / soft_fraction are per-expert normalizations,
        entropy is a mean over tokens, dead_experts is a global zero-count, and the CVs are std/mean --
        all nonlinear in the local statistics, so averaging the per-rank derived values is wrong (e.g. an
        expert that is idle on one rank but busy on another would be miscounted as partially "dead").
        The additive sufficient statistics (token_counts, prob_mass, num_tokens, and the entropy
        numerator = entropy * num_tokens) sum cleanly across ranks; every derived metric is then rebuilt
        with the exact formulas used in LoadBalancingLoss.compute_metrics, so world_size == 1 reproduces
        the single-device result bit-for-bit.
        """
        if len(metric_dicts) == 0:
            return []

        dev= self.device
        # pack the additive sufficient statistics of every dict into one [D, 2E+2] tensor
        rows= []
        for m in metric_dicts:
            tc= m["token_counts"].to(dev).flatten().float()   # [E] hard slot counts
            pm= m["prob_mass"].to(dev).flatten().float()      # [E] soft probability mass
            nt= m["num_tokens"].to(dev).reshape(1).float()    # [1] token (row) count
            en= m["entropy"].to(dev).reshape(1).float() * nt  # [1] entropy numerator (mean * count)
            rows.append(torch.cat([tc, pm, nt, en], dim=0))
        packed= torch.stack(rows, dim=0)                      # [D, 2E+2], identical shape on every rank

        # the only collective: sum the sufficient statistics across ranks
        packed= self._reduce_sum(packed)
        # unpack and rebuild every derived metric from the global sufficient statistics
        reduced= []
        for row in packed:
            e= (row.numel() - 2) // 2
            g_tc= row[:e]                                     # global token_counts [E]
            g_pm= row[e:2 * e]                                # global prob_mass    [E]
            g_nt= row[2 * e]                                  # global num_tokens   (scalar)
            g_en= row[2 * e + 1]                              # global entropy numerator (scalar)

            hard_fraction= g_tc / g_tc.sum().clamp_min(1.0)
            soft_fraction= g_pm / g_pm.sum().clamp_min(1e-12)
            reduced.append({
                "hard_fraction": hard_fraction,
                "soft_fraction": soft_fraction,
                "token_counts": g_tc,
                "prob_mass": g_pm,
                "entropy": g_en / g_nt.clamp_min(1.0),
                "dead_experts": (g_tc == 0).sum(),
                "cv_hard": hard_fraction.std(unbiased=False) / hard_fraction.mean().clamp_min(1e-12),
                "cv_soft": soft_fraction.std(unbiased=False) / soft_fraction.mean().clamp_min(1e-12),
                "num_tokens": g_nt,
            })
        return reduced


    def _reduce_moe_metrics(self, metrics):
        """
        Aggregate MoE routing diagnostics across distributed ranks. Reduces the sufficient statistics
        (token_counts, prob_mass, num_tokens, entropy numerator) and recomputes every derived quantity
        globally. Handles both structures the base trainer passes in:
        - the flat global-metrics dict, and
        - the per-layer {layer_id: metrics_dict} mapping (reduced in a single collective).
        The structure/None branching depends only on the model architecture (which is identical on every
        rank), never on data values, so all ranks always issue the same collectives in the same order.
        """
        # nothing to reduce for no-MoE steps or single-process runs; also avoids issuing a collective
        # (returning the input unchanged matches the base-class no-op exactly for world_size == 1)
        if metrics is None or self.fabric.world_size == 1:
            return metrics

        # per-layer mapping {layer_id: metrics_dict}: reduce every layer together in one all-reduce
        # (an empty mapping -- e.g. a model with no MoE layers -- issues no collective on any rank)
        if all(isinstance(v, dict) for v in metrics.values()):
            if len(metrics) == 0:
                return metrics
            layer_ids= sorted(metrics.keys())  # sorted -> identical order on all ranks
            reduced= self._reduce_moe_stats([metrics[i] for i in layer_ids])
            return {i: r for i, r in zip(layer_ids, reduced)}

        # flat global-metrics dict
        return self._reduce_moe_stats([metrics])[0]


    def _save(self, state_file, state_path):
        self.fabric.save(state_path, state_file)


    def _load(self, state_path, map_location="cpu", weights_only=True):
        return self.fabric.load(state_path)


    def train(self, *args, **kwargs) -> None:
        """
        Distributed training entry point. Precision is delegated to Fabric (so use_bf16 is forced off
        for the base loop).
        - get_moe_metrics: when enabled, the per-step routing diagnostics are aggregated across ranks
        by the _reduce_moe_metrics override before being tracked, so the ExpertUsageTracker on every
        rank holds the correct global statistics.
        """
        finfo= (
            f"accelerator={self.fabric.accelerator.__class__.__name__.replace('Accelerator', '').lower()}, "
            f"devices={self.fabric.world_size}, precision={self.fabric._precision.precision}, "
            f"strategy={self.fabric.strategy.__class__.__name__.replace('Strategy', '').lower()}"
        )
        self._set_log("info", f"train | Fabric {finfo}")

        kwargs["use_bf16"]= None  # not necessary, fabric handles precision
        super().train(*args, **kwargs)
